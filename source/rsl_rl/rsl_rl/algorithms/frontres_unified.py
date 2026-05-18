from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, FrontRESActorCritic, ResidualActorCritic
from rsl_rl.storage import RolloutStorage


class FrontRESUnified:
    """FrontRES PPO + supervised ΔSE3 training.

    This class intentionally owns only the pieces FrontRES needs:
    on-policy PPO, the online ΔSE3 supervised auxiliary loss, optional reference
    velocity estimation, and the split-env FrontRES mask.  MOSAIC teacher BC and
    off-policy expert BC are not implemented here by design.
    """

    policy: ActorCritic

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
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        obs_normalizer: Optional[torch.nn.Module] = None,
        privileged_obs_normalizer: Optional[torch.nn.Module] = None,
        use_estimate_ref_vel: bool = False,
        ref_vel_estimator_checkpoint_path: Optional[str] = None,
        ref_vel_estimator_type: str = "mlp",
        lambda_supervised: float = 0.0,
        lambda_supervised_min: float = 0.05,
        lambda_supervised_decay: float = 0.997,
        supervised_trigger_cosine_sim: float = 0.85,
        supervised_rpy_loss_weight: float = 1.0,
        supervised_conf_loss_weight: float = 0.05,
        supervised_direction_loss_weight: float = 0.1,
        supervised_valid_loss_weight: float = 4.0,
        ppo_actor_warmup_iterations: int = 0,
        ppo_actor_ramp_iterations: int = 0,
        ppo_advantage_focal_power: float = 0.0,
        frontres_active_task_dims: list[int] | None = None,
        diagnose_gradient_conflict: bool = True,
        hybrid: bool = True,
        use_ppo: bool = True,
        gradient_accumulation_steps: int = 1,
        **disabled_mosaic_kwargs,
    ):
        self._assert_no_mosaic_branches(disabled_mosaic_kwargs)
        if not hybrid:
            raise ValueError("FrontRESUnified supports only hybrid=True PPO+supervised training.")
        if not use_ppo:
            raise ValueError("FrontRESUnified requires use_ppo=True.")
        if gradient_accumulation_steps != 1:
            raise ValueError("FrontRESUnified does not use MOSAIC gradient accumulation.")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

        if rnd_cfg is not None:
            raise ValueError("FrontRESUnified does not support RND.")
        if symmetry_cfg is not None:
            raise ValueError("FrontRESUnified does not support symmetry augmentation.")

        self.obs_normalizer = obs_normalizer
        self.privileged_obs_normalizer = privileged_obs_normalizer

        self.use_estimate_ref_vel = use_estimate_ref_vel
        self.ref_vel_estimator = None
        self.ref_vel_estimator_obs_shape = None
        if use_estimate_ref_vel:
            if ref_vel_estimator_checkpoint_path is None:
                raise ValueError("ref_vel_estimator_checkpoint_path must be provided when use_estimate_ref_vel=True")
            self._load_ref_vel_estimator(ref_vel_estimator_checkpoint_path, ref_vel_estimator_type)

        self.policy = policy.to(self.device)

        trainable_params = self._collect_trainable_params(policy)
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)

        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        self.use_ppo = True
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

        self.lambda_supervised = lambda_supervised
        self.lambda_supervised_min = lambda_supervised_min
        self.lambda_supervised_decay_rate = lambda_supervised_decay
        self.supervised_trigger_cosine_sim = supervised_trigger_cosine_sim
        self.supervised_rpy_loss_weight = supervised_rpy_loss_weight
        self.supervised_conf_loss_weight = supervised_conf_loss_weight
        self.supervised_direction_loss_weight = supervised_direction_loss_weight
        self.supervised_valid_loss_weight = supervised_valid_loss_weight
        self.ppo_actor_warmup_iterations = int(ppo_actor_warmup_iterations)
        self.ppo_actor_ramp_iterations = int(ppo_actor_ramp_iterations)
        self.ppo_advantage_focal_power = float(ppo_advantage_focal_power)
        self.frontres_active_task_dims = frontres_active_task_dims
        self.diagnose_gradient_conflict = bool(diagnose_gradient_conflict)
        self.ppo_actor_weight = 1.0
        self._supervised_decay_triggered = False
        self._supervised_cosine_ema = 0.0
        self._supervised_ema_alpha = 0.05

        self.is_frontres_unified = True
        self._print_init_summary()

    @staticmethod
    def _assert_no_mosaic_branches(kwargs: dict) -> None:
        forbidden_nonzero = {
            "teacher_checkpoint_path": None,
            "teacher_policy": None,
            "teacher_policy_cfg": None,
            "teacher_obs_source_mapping": None,
            "teacher_critic_checkpoint_path": None,
            "expert_trajectory_path": None,
        }
        for key, disabled_value in forbidden_nonzero.items():
            if kwargs.get(key, disabled_value) is not disabled_value:
                raise ValueError(f"FrontRESUnified does not support MOSAIC branch '{key}'.")

        for key in ("lambda_teacher_init", "lambda_teacher_min", "lambda_off_policy", "lambda_off_policy_min"):
            if float(kwargs.get(key, 0.0) or 0.0) != 0.0:
                raise ValueError(f"FrontRESUnified requires {key}=0.0.")

    def _load_ref_vel_estimator(self, checkpoint_path: str, estimator_type: str) -> None:
        print(f"[FrontRESUnified] Loading reference velocity estimator from: {checkpoint_path}")
        if estimator_type == "mlp":
            from rsl_rl.modules import VelocityEstimator

            self.ref_vel_estimator = VelocityEstimator.load(checkpoint_path, device=self.device)
        elif estimator_type == "transformer":
            from rsl_rl.modules import VelocityEstimatorTransformer

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.ref_vel_estimator = VelocityEstimatorTransformer(
                feature_dim=checkpoint.get("feature_dim", 61),
                history_length=checkpoint.get("history_length", 5),
                d_model=checkpoint.get("d_model", 128),
                nhead=checkpoint.get("nhead", 4),
                num_layers=checkpoint.get("num_layers", 2),
            ).to(self.device)
            self.ref_vel_estimator.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"Unknown ref_vel_estimator_type: {estimator_type}. Must be 'mlp' or 'transformer'.")

        self.ref_vel_estimator.eval()
        for param in self.ref_vel_estimator.parameters():
            param.requires_grad = False
        self.ref_vel_estimator_obs_shape = (self.ref_vel_estimator.num_obs,)
        print("[FrontRESUnified] Reference velocity estimator loaded and frozen")

    @staticmethod
    def _collect_trainable_params(policy):
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            params = list(policy.residual_actor.parameters())
            params.extend(policy.critic.parameters())
            has_trainable_std = False
            if hasattr(policy, "std") and getattr(policy.std, "requires_grad", False):
                params.append(policy.std)
                has_trainable_std = True
            elif hasattr(policy, "log_std") and getattr(policy.log_std, "requires_grad", False):
                params.append(policy.log_std)
                has_trainable_std = True
            suffix = " + policy std" if has_trainable_std else " (fixed policy std)"
            print(f"[FrontRESUnified] Optimizer updates residual_actor + critic{suffix}")
            return params
        print("[FrontRESUnified] Optimizer updates full policy")
        return policy.parameters()

    def _print_init_summary(self):
        print("=" * 80)
        print("  FrontRESUnified ▸ PPO + Supervised ΔSE3")
        print(f"  L = L_PPO + λ_sup({self.lambda_supervised:.2f})·L_supervised")
        print("=" * 80)
        print(f"  LR={self.learning_rate}  clip={self.clip_param}  ent_coef={self.entropy_coef}")
        print(f"  epochs={self.num_learning_epochs}  mini_batches={self.num_mini_batches}")
        print(f"  Supervised  λ={self.lambda_supervised:.3f} → {self.lambda_supervised_min}"
              f"  decay={self.lambda_supervised_decay_rate}"
              f"  trigger_cos={self.supervised_trigger_cosine_sim}"
              f"  rpy_w={self.supervised_rpy_loss_weight}"
              f"  conf_w={self.supervised_conf_loss_weight}"
              f"  dir_w={self.supervised_direction_loss_weight}"
              f"  valid_w={self.supervised_valid_loss_weight}")
        print(f"  PPO actor warmup={self.ppo_actor_warmup_iterations}"
              f"  ramp={self.ppo_actor_ramp_iterations}"
              f"  adv_focal_power={self.ppo_advantage_focal_power}")
        print("  MOSAIC teacher/off-policy branches: disabled by construction")
        print("=" * 80)

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        teacher_obs_shape=None,
        ref_vel_estimator_obs_shape=None,
    ):
        if training_type != "frontres":
            raise ValueError(f"FrontRESUnified storage must use training_type='frontres', got {training_type!r}.")
        self.ref_vel_estimator_obs_shape = ref_vel_estimator_obs_shape
        self.storage = RolloutStorage(
            "frontres",
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            None,
            self.device,
            teacher_obs_shape=None,
            ref_vel_estimator_obs_shape=ref_vel_estimator_obs_shape,
        )

    def act(self, obs, critic_obs, teacher_obs=None, ref_vel_estimator_obs=None, motion_groups=None):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
            estimator_input = ref_vel_estimator_obs if ref_vel_estimator_obs is not None else obs
            with torch.no_grad():
                estimated_ref_vel = self.ref_vel_estimator(estimator_input)
                self.last_estimated_ref_vel = estimated_ref_vel.clone()
                obs_augmented = torch.cat([obs, estimated_ref_vel], dim=-1)
        else:
            obs_augmented = obs
            self.last_estimated_ref_vel = None

        self.transition.actions = self.policy.act(obs_augmented).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.ref_vel_estimator_observations = ref_vel_estimator_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch)

    def update(self):
        loss_dict = self._update_ppo_supervised()
        self._step_supervised_lambda(loss_dict.get("supervised_cos_sim", 0.0))
        return loss_dict

    def _step_supervised_lambda(self, cos_sim: float):
        if self.lambda_supervised <= self.lambda_supervised_min:
            return
        self._supervised_cosine_ema = (
            (1.0 - self._supervised_ema_alpha) * self._supervised_cosine_ema
            + self._supervised_ema_alpha * cos_sim
        )
        if not self._supervised_decay_triggered:
            if self._supervised_cosine_ema >= self.supervised_trigger_cosine_sim:
                self._supervised_decay_triggered = True
                print(f"[FrontRESUnified] Supervised λ decay triggered: "
                      f"cos_sim_ema={self._supervised_cosine_ema:.3f} >= "
                      f"{self.supervised_trigger_cosine_sim:.3f}")

        if self._supervised_decay_triggered:
            self.lambda_supervised = max(
                self.lambda_supervised * self.lambda_supervised_decay_rate,
                self.lambda_supervised_min,
            )

    def _update_ppo_supervised(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_supervised_loss = 0.0
        mean_supervised_cos_sim = 0.0
        grad_conflict_cos = 0.0
        grad_conflict_norm_ratio = 0.0
        grad_conflict_count = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

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
            _teacher_obs_batch,
            _teacher_mu_batch,
            _teacher_sigma_batch,
            ref_vel_estimator_obs_batch,
            _motion_groups_batch,
            frontres_mask_batch,
            supervised_target_batch,
            frontres_actor_gate_batch,
        ) in generator:
            original_batch_size = obs_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                with torch.no_grad():
                    estimator_input = ref_vel_estimator_obs_batch if ref_vel_estimator_obs_batch is not None else obs_batch
                    estimated_ref_vel_batch = self.ref_vel_estimator(estimator_input)
                    obs_batch_augmented = torch.cat([obs_batch, estimated_ref_vel_batch], dim=-1)
            else:
                obs_batch_augmented = obs_batch

            self.policy.update_distribution(obs_batch_augmented)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                self._adapt_learning_rate(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)

            surrogate_loss, value_loss = self._compute_ppo_losses(
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                advantages_batch,
                target_values_batch,
                returns_batch,
                value_batch,
                frontres_mask_batch,
                frontres_actor_gate_batch,
            )
            supervised_loss, sup_cos_sim = self._compute_supervised_loss(
                mu_batch, supervised_target_batch, original_batch_size)

            oracle_mix = float(getattr(self, "oracle_mix", 0.0))
            ppo_weight = float(getattr(self, "ppo_actor_weight", 1.0)) * (1.0 - oracle_mix)
            if self.diagnose_gradient_conflict and ppo_weight > 0.0 and self.lambda_supervised > 0.0:
                _gc, _ratio = self._compute_actor_grad_conflict(
                    surrogate_loss, supervised_loss)
                if _gc is not None:
                    grad_conflict_cos += _gc
                    grad_conflict_norm_ratio += _ratio
                    grad_conflict_count += 1
            loss = (
                ppo_weight * surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.lambda_supervised * supervised_loss
            )

            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                self._warn_skip("non-finite loss", loss)
                continue

            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()

            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in self.policy.parameters() if p.requires_grad):
                self._warn_skip("NaN gradient detected")
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_supervised_loss += supervised_loss.item()
            mean_supervised_cos_sim += sup_cos_sim

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_supervised_loss /= num_updates
        mean_supervised_cos_sim /= num_updates
        if grad_conflict_count > 0:
            grad_conflict_cos /= grad_conflict_count
            grad_conflict_norm_ratio /= grad_conflict_count
        else:
            grad_conflict_cos = 0.0
            grad_conflict_norm_ratio = 0.0

        self.storage.clear()
        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_off_policy": 0.0,
            "bc_teacher": 0.0,
            "lambda_off_policy": 0.0,
            "lambda_teacher": 0.0,
            "supervised_loss": mean_supervised_loss,
            "supervised_cos_sim": mean_supervised_cos_sim,
            "lambda_supervised": self.lambda_supervised,
            "ppo_actor_weight": float(getattr(self, "ppo_actor_weight", 1.0)),
            "grad_cos_ppo_supervised": grad_conflict_cos,
            "grad_norm_ratio_ppo_to_supervised": grad_conflict_norm_ratio,
        }

    def _compute_actor_grad_conflict(self, surrogate_loss, supervised_loss):
        params = [
            p for p in self.policy.residual_actor.parameters()
            if p.requires_grad
        ] if hasattr(self.policy, "residual_actor") else []
        if not params:
            return None, 0.0

        ppo_grads = torch.autograd.grad(
            surrogate_loss, params, retain_graph=True, allow_unused=True)
        sup_grads = torch.autograd.grad(
            supervised_loss, params, retain_graph=True, allow_unused=True)

        dot = torch.tensor(0.0, device=self.device)
        ppo_norm_sq = torch.tensor(0.0, device=self.device)
        sup_norm_sq = torch.tensor(0.0, device=self.device)
        for gp, gs in zip(ppo_grads, sup_grads):
            if gp is None or gs is None:
                continue
            dot = dot + (gp * gs).sum()
            ppo_norm_sq = ppo_norm_sq + gp.square().sum()
            sup_norm_sq = sup_norm_sq + gs.square().sum()

        denom = (ppo_norm_sq.sqrt() * sup_norm_sq.sqrt()).clamp(min=1e-12)
        if ppo_norm_sq <= 0 or sup_norm_sq <= 0:
            return None, 0.0
        cos = (dot / denom).detach().item()
        ratio = (ppo_norm_sq.sqrt() / sup_norm_sq.sqrt().clamp(min=1e-12)).detach().item()
        return cos, ratio

    def _adapt_learning_rate(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch):
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_ppo_losses(
        self,
        actions_log_prob_batch,
        old_actions_log_prob_batch,
        advantages_batch,
        target_values_batch,
        returns_batch,
        value_batch,
        frontres_mask_batch,
        frontres_actor_gate_batch=None,
    ):
        has_mask = frontres_mask_batch is not None
        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

        advantages = torch.squeeze(advantages_batch)
        focal_power = max(0.0, float(getattr(self, "ppo_advantage_focal_power", 0.0)))
        if focal_power > 0.0:
            focal = advantages.abs().pow(focal_power).clamp(max=25.0)
        else:
            focal = torch.ones_like(advantages)
        surrogate = -advantages * ratio * focal
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * focal
        surrogate_terms = torch.max(surrogate, surrogate_clipped)

        if has_mask:
            if frontres_actor_gate_batch is not None:
                mask_flat = (frontres_mask_batch * frontres_actor_gate_batch).view(-1)
            else:
                mask_flat = frontres_mask_batch.view(-1)
            surrogate_loss = (surrogate_terms * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        else:
            surrogate_loss = surrogate_terms.mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_terms = torch.max(value_losses, value_losses_clipped)
        else:
            value_terms = (returns_batch - value_batch).pow(2)

        if has_mask:
            n = frontres_mask_batch.sum().clamp(min=1.0)
            value_loss = (value_terms * frontres_mask_batch).sum() / n
        else:
            value_loss = value_terms.mean()
        return surrogate_loss, value_loss

    def _compute_supervised_loss(self, mu_batch, supervised_target_batch, original_batch_size):
        supervised_loss = torch.tensor(0.0, device=self.device)
        sup_cos_sim = 0.0
        if supervised_target_batch is None or self.lambda_supervised <= 0:
            return supervised_loss, sup_cos_sim

        mu_dim = mu_batch.shape[-1]
        sup_dim = supervised_target_batch.shape[-1]
        if mu_dim < sup_dim:
            return supervised_loss, sup_cos_sim

        raw_pred = mu_batch[:original_batch_size]
        target = supervised_target_batch[:original_batch_size]

        if hasattr(self.policy, "num_task_corrections") and self.policy.num_task_corrections > 0:
            pred = torch.cat([
                torch.tanh(raw_pred[:, :3]) * self.policy.max_delta_pos,
                torch.tanh(raw_pred[:, 3:6]) * self.policy.max_delta_rpy,
            ], dim=-1)
        else:
            pred = raw_pred

        if hasattr(self.policy, "max_delta_pos") and hasattr(self.policy, "max_delta_rpy"):
            target = torch.cat([
                target[:, :3].clamp(-self.policy.max_delta_pos, self.policy.max_delta_pos),
                target[:, 3:].clamp(-self.policy.max_delta_rpy, self.policy.max_delta_rpy),
            ], dim=-1)

        active_dims = getattr(self, "frontres_active_task_dims", None)
        if (
            active_dims is not None
            and hasattr(self.policy, "num_task_corrections")
            and self.policy.num_task_corrections > 0
        ):
            mask = torch.zeros(pred.shape[-1], device=self.device, dtype=pred.dtype)
            for idx in active_dims:
                idx = int(idx)
                if 0 <= idx < pred.shape[-1]:
                    mask[idx] = 1.0
            pred = pred * mask.view(1, -1)
            target = target * mask.view(1, -1)

        target_detached = target.detach()
        target_norm = target_detached.norm(dim=-1)
        valid = target_norm > 1e-4

        pos_valid = target_detached[:, :3].norm(dim=-1) > 1e-4
        rpy_valid = target_detached[:, 3:].norm(dim=-1) > 1e-4
        pos_weight = torch.ones_like(target_norm)
        rpy_weight = torch.ones_like(target_norm)
        valid_weight = float(self.supervised_valid_loss_weight)
        if pos_valid.any():
            pos_weight[pos_valid] = valid_weight
        if rpy_valid.any():
            rpy_weight[rpy_valid] = valid_weight
        pos_weight = pos_weight / pos_weight.mean().clamp(min=1e-6)
        rpy_weight = rpy_weight / rpy_weight.mean().clamp(min=1e-6)

        pos_err = nn.functional.huber_loss(
            pred[:, :3], target_detached[:, :3], reduction="none").mean(dim=-1)
        rpy_err = nn.functional.huber_loss(
            pred[:, 3:], target_detached[:, 3:], reduction="none").mean(dim=-1)
        pos_sup = (pos_err * pos_weight).mean()
        rpy_sup = (rpy_err * rpy_weight).mean()
        supervised_loss = pos_sup + self.supervised_rpy_loss_weight * rpy_sup

        if self.supervised_direction_loss_weight > 0:
            direction_loss = torch.zeros((), device=self.device)
            if pos_valid.any():
                direction_loss = direction_loss + (
                    1.0 - nn.functional.cosine_similarity(
                        pred[pos_valid, :3], target_detached[pos_valid, :3], dim=-1).mean()
                )
            if rpy_valid.any():
                direction_loss = direction_loss + (
                    1.0 - nn.functional.cosine_similarity(
                        pred[rpy_valid, 3:], target_detached[rpy_valid, 3:], dim=-1).mean()
                )
            supervised_loss = supervised_loss + self.supervised_direction_loss_weight * direction_loss

        if (
            hasattr(self.policy, "num_task_corrections")
            and self.policy.num_task_corrections > 0
            and raw_pred.shape[-1] >= 8
            and self.supervised_conf_loss_weight > 0
        ):
            target_conf = valid.view(-1, 1).to(raw_pred.dtype)
            conf_sup = nn.functional.binary_cross_entropy_with_logits(
                raw_pred[:, 6:8], target_conf.expand(-1, 2).detach())
            supervised_loss = supervised_loss + self.supervised_conf_loss_weight * conf_sup

        with torch.no_grad():
            if valid.any():
                sup_cos_sim = nn.functional.cosine_similarity(
                    pred[valid], target[valid], dim=-1).mean().item()
        return supervised_loss, sup_cos_sim

    def _warn_skip(self, reason: str, loss: torch.Tensor | None = None):
        skip_count = getattr(self, "_nan_skip_count", 0) + 1
        self._nan_skip_count = skip_count
        if skip_count <= 5 or skip_count % 100 == 0:
            suffix = f" ({loss.item():.4g})" if loss is not None else ""
            print(f"[FrontRESUnified] WARNING: {reason}{suffix}, skipping update (skip #{skip_count})")

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel
