# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic, ActorCriticVQ, FrontRESActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # FrontRES Stage-2 正则化：惩罚过大的 Δq，防止修正幅度失控
        # L_reg = lambda_reg * ||Δq_mean||^2，与 PPO loss 通过 PCGrad 协调梯度方向
        lambda_reg_init: float = 0.0,
        lambda_reg_decay: float = 1.0,
        lambda_reg_min: float = 0.0,
        # PCGrad：检测 PPO 梯度与正则化梯度的冲突并投影消解
        use_pcgrad: bool = False,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer — for FrontRESActorCritic only train residual_actor + critic + std;
        # GMT parameters have requires_grad=False so policy.parameters() already excludes them,
        # but we make it explicit here to match the MOSAIC optimizer pattern.
        if isinstance(policy, FrontRESActorCritic):
            trainable = list(policy.residual_actor.parameters()) + list(policy.critic.parameters())
            if hasattr(policy, "std"):
                trainable.append(policy.std)
            elif hasattr(policy, "log_std"):
                trainable.append(policy.log_std)
            self.optimizer = optim.Adam(trainable, lr=learning_rate)
            print("[PPO] FrontRESActorCritic detected: optimizer restricted to residual_actor + critic")
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # FrontRES 正则化参数
        self.lambda_reg_init    = lambda_reg_init
        self.lambda_reg_current = lambda_reg_init
        self.lambda_reg_decay   = lambda_reg_decay
        self.lambda_reg_min     = lambda_reg_min
        self.use_pcgrad         = use_pcgrad
        self.use_frontres_reg   = (lambda_reg_init > 0.0) and isinstance(policy, FrontRESActorCritic)
        # smooth_loss: 时序平滑约束，对 rollout buffer 中相邻帧 Δq 的差分做 L2 惩罚
        # L_smooth = lambda_smooth * mean_t(||Δq_t - Δq_{t-1}||^2)
        # 默认 0.0（禁用），待 Stage 2 超参调优后启用
        self.lambda_smooth = 0.0

        if self.use_frontres_reg:
            print(f"[PPO] FrontRES regularization enabled: λ_reg={lambda_reg_init} "
                  f"(decay={lambda_reg_decay}, min={lambda_reg_min})")
        if self.use_pcgrad:
            print("[PPO] PCGrad enabled for FrontRES Stage-2 training")
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_reg_loss = 0
        mean_smooth_loss = 0
        mean_kl_divergence = 0

        # ── smooth_loss + delta_q diagnostics: 在 rollout buffer 全局计算 ──────
        # 必须在循环前计算：mini-batch 会打乱时序顺序，无法正确计算相邻帧差分。
        # storage.actions shape: (T, B, A)，T = num_transitions_per_env，存储的是 Δq 采样值
        smooth_loss = torch.tensor(0.0, device=self.device)
        delta_q_norm_mean = torch.tensor(0.0, device=self.device)
        delta_q_norm_std  = torch.tensor(0.0, device=self.device)
        if self.use_frontres_reg:
            delta_q = self.storage.actions                  # (T, B, A)
            per_step_norm = delta_q.norm(dim=-1)            # (T, B)
            delta_q_norm_mean = per_step_norm.mean()
            delta_q_norm_std  = per_step_norm.std()
            if self.lambda_smooth > 0.0:
                diff = delta_q[1:] - delta_q[:-1]          # (T-1, B, A)
                smooth_loss = diff.pow(2).mean()
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        # -- VQ loss
        if isinstance(self.policy, ActorCriticVQ):
            mean_vq_loss = 0
            mean_vq_perplexity = 0
        else:
            mean_vq_loss = None
            mean_vq_perplexity = None
        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    mean_kl_divergence += kl_mean.item()

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ====== FrontRES Stage-2 正则化 ======
            # mu_batch 此时是 Δq_mean（FrontRESActorCritic 的 distribution.mean），
            # 用其 L2 范数作为正则化项，拉近修正量与零（即拉近 q' 与 q_ref 的距离）。
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.use_frontres_reg and self.lambda_reg_current > 0.0:
                reg_loss = mu_batch.pow(2).mean()

            # ====== 组合 loss（或 PCGrad）======
            if self.use_pcgrad and self.use_frontres_reg and self.lambda_reg_current > 0.0:
                # PCGrad：分别求各任务梯度，投影消解冲突后合并
                loss = None  # 不直接 backward，由 _pcgrad_step 处理
            else:
                loss = ppo_loss + self.lambda_reg_current * reg_loss + self.lambda_smooth * smooth_loss

            # VQ loss（加到 ppo_loss 统一路径，PCGrad 暂不支持 VQ）
            if isinstance(self.policy, ActorCriticVQ):
                vq_loss = self.policy.vq_loss
                loss += vq_loss
                vq_perplexity = self.policy.vq_perplexity.item()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            if self.use_pcgrad and self.use_frontres_reg and self.lambda_reg_current > 0.0:
                # PCGrad 路径：不调用 loss.backward()，由 _pcgrad_step 接管
                # smooth_loss 与 reg_loss 目标一致（均约束 Δq 幅度），合并为一个任务
                task_losses = {
                    "ppo": ppo_loss,
                    "reg": self.lambda_reg_current * reg_loss + self.lambda_smooth * smooth_loss,
                }
                self._pcgrad_step(task_losses)
            else:
                # 标准路径
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if self.use_frontres_reg:
                mean_reg_loss += reg_loss.item()
                mean_smooth_loss += smooth_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # -- VQ loss
            if isinstance(self.policy, ActorCriticVQ):
                mean_vq_loss += vq_loss.item()
                mean_vq_perplexity += vq_perplexity
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- VQ loss
        if isinstance(self.policy, ActorCriticVQ):
            mean_vq_loss /= num_updates
            mean_vq_perplexity /= num_updates
        # -- lambda_reg 衰减
        if self.use_frontres_reg:
            self.lambda_reg_current = max(
                self.lambda_reg_min,
                self.lambda_reg_current * self.lambda_reg_decay)
            mean_reg_loss    /= num_updates
            mean_smooth_loss /= num_updates

        # -- Clear the storage
        self.storage.clear()

        # kl_divergence is only accumulated when schedule="adaptive"; guard against div-by-zero
        if self.schedule == "adaptive" and self.desired_kl is not None:
            mean_kl_divergence /= num_updates

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_divergence": mean_kl_divergence,
        }
        if self.use_frontres_reg:
            loss_dict["reg"]              = mean_reg_loss
            loss_dict["smooth"]           = mean_smooth_loss
            loss_dict["lambda_reg"]       = self.lambda_reg_current
            loss_dict["delta_q_norm_mean"] = delta_q_norm_mean.item()
            loss_dict["delta_q_norm_std"]  = delta_q_norm_std.item()
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if isinstance(self.policy, ActorCriticVQ):
            loss_dict["vq"] = mean_vq_loss
            loss_dict["vq_perplexity"] = mean_vq_perplexity
        return loss_dict

    """
    Helper functions
    """

    def _pcgrad_step(self, task_losses: dict) -> None:
        """
        PCGrad (Project Conflicting Gradients) 优化步骤。

        对每对任务 (i, j)：若 ∇Li · ∇Lj < 0（梯度冲突），
        则将 ∇Li 投影到 ∇Lj 的法平面，消除冲突分量。
        最终梯度 = 各任务投影后梯度之和。

        参考：Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

        Args:
            task_losses: {task_name: loss_tensor}，每个 loss 必须共享同一计算图
                         （本方法内部通过 retain_graph=True 多次反向传播）。
        """
        params = [p for p in self.policy.parameters() if p.requires_grad]
        names  = list(task_losses.keys())
        n      = len(names)

        if n == 0:
            return

        # ── Step 1: 分别计算每个任务的梯度 ──────────────────────────────────
        task_grads: dict[str, list[torch.Tensor]] = {}
        for idx, (name, loss) in enumerate(task_losses.items()):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            # 除最后一次外保留计算图，以便后续任务也能 backward
            loss.backward(retain_graph=(idx < n - 1))
            task_grads[name] = [
                p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for p in params
            ]

        # ── Step 2: PCGrad 投影 ────────────────────────────────────────────
        # proj_grads[i] 是任务 i 经过投影后的梯度列表（从原始梯度出发逐步修正）
        proj_grads: dict[str, list[torch.Tensor]] = {
            name: [g.clone() for g in grads]
            for name, grads in task_grads.items()
        }

        # 随机化任务顺序，减少系统性偏差
        perm = list(names)
        random.shuffle(perm)

        for i_name in perm:
            for j_name in perm:
                if i_name == j_name:
                    continue
                # 展平梯度向量用于点积计算
                g_i = torch.cat([g.reshape(-1) for g in proj_grads[i_name]])
                g_j = torch.cat([g.reshape(-1) for g in task_grads[j_name]])

                dot = torch.dot(g_i, g_j)
                if dot < 0:  # 梯度冲突：投影消除
                    norm_gj_sq = torch.dot(g_j, g_j).clamp(min=1e-8)
                    scale = dot / norm_gj_sq
                    for k in range(len(params)):
                        proj_grads[i_name][k] = proj_grads[i_name][k] - scale * task_grads[j_name][k]

        # ── Step 3: 合并投影后梯度并更新 ─────────────────────────────────
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        for k, p in enumerate(params):
            p.grad = sum(proj_grads[name][k] for name in names)

        if self.is_multi_gpu:
            self.reduce_parameters()

        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
