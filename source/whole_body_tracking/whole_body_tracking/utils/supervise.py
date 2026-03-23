# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import SuperviseLearning

class SuperviseStorage:
    """
    预留的独立经验池类。
    接口与 rsl_rl.storage.RolloutStorage 完全保持一致，方便你理解后进行修改或无缝替换。
    """
    class Transition:
        def __init__(self):
            self.observations = None
            self.target_actions = None
            self.dones = None

        def clear(self):
            self.observations = None
            self.target_actions = None
            self.dones = None

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        # 核心数据缓冲区
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.target_actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow.")
        self.observations[self.step].copy_(transition.observations)
        self.target_actions[self.step].copy_(transition.target_actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def generator(self):
        """生成器：按时间步返回批次数据，保持与原生 generator 一致的调用体验"""
        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.target_actions[i], self.dones[i]

class SuperviseTrainer:
    """Supervised learning algorithm for training FrontRES to output delta_q."""

    def __init__(
        self,
        policy: SuperviseLearning,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None, # Distributed training parameters
    ):
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Policy (FrontRES Student)
        self.policy = policy
        self.policy.to(self.device)

        # Create the optimizer
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)

        # Storage
        self.storage = None
        self.transition = SuperviseStorage.Transition()
        self.last_hidden_states = None

        # Training parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.num_updates = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """初始化内置存储区。此函数将被 OnPolicyRunner 自动调用。"""
        self.storage = SuperviseStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            action_shape=action_shape,
            device=self.device
        )

    def act(self, obs: torch.Tensor, target_delta_q: torch.Tensor) -> torch.Tensor:
        """采样动作并记录环境传来的标答 (delta q)。"""
        # Compute the actions
        actions = self.policy.act(obs).detach()
        
        # 使用标准的 transition 流水线记录当前步的数据
        self.transition.observations = obs
        self.transition.target_actions = target_delta_q
        
        return actions

    def process_env_step(self, rewards, dones, infos) -> None:
        """将数据正式存入内置 Buffer 中。"""
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        pass

    def update(self) -> dict:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            # 直接按时间步迭代我们内置的 Buffer
            for obs, target_actions, dones in self.storage.generator():
                # Inference of the FrontRES student
                predicted_actions = self.policy.act_inference(obs)

                # 计算行为克隆 Loss：预测的 delta_q vs 真实的 delta_q
                behavior_loss = self.loss_fn(predicted_actions, target_actions)

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict
