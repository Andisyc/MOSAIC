# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======================================================================================
# 思路与逻辑梳理 (注释) - 简化版 (仅针对第一阶段监督学习)
#
# 核心架构:
#   1. GMT (ONNX模型): 一个预训练的专家模型，它可以根据给定的参考运动q生成高质量的动作。
#   2. student (FrontRES): 一个前端网络。在第一阶段，它的目标是学习预测一个“真实”的残差 Δ_q_gt。
#      这个真实的残差来自于GMT在仿真中的实际表现。
#
# 监督学习训练流程 (在外部训练脚本如 `supervise.py` 中实现):
#   1. [数据准备] 从数据集中获取一个参考运动 q_ref。
#   2. [获取专家数据] 调用 `get_gmt_action(q_ref)` 来获得 GMT 专家在原始 q_ref 上的动作 a_gmt。
#   3. [与环境交互] 在仿真环境中执行 a_gmt，得到实际的模拟结果 q_sim。
#   4. [计算监督目标] 调用 `get_supervision_target(q_sim, q_ref)`，计算出真实的残差
#      Δ_q_gt = q_sim - q_ref。这就是我们希望 `student` 网络学会预测的目标。
#   5. [模型预测] 将 q_ref (或其他观测) 输入到 `student` 网络中，通过调用 `forward(obs)` 得到
#      预测的残差 Δ_q_pred。
#   6. [计算损失] 计算 Δ_q_pred 和 Δ_q_gt 之间的损失 (例如 MSELoss)。
#   7. [反向传播] 根据损失更新 `student` 网络的权重。
#
#  *注*: 在这个简化的第一阶段中，我们不关心 "q_repaired" 或完整的推理流程。
#  我们只专注于训练 `student` 网络来准确预测 `Δ_q_gt`。
# =======================================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class SuperviseLearning(nn.Module):
    """
    A module for supervised learning aimed at training a network to predict Δq
    """
    is_recurrent = False
    is_encoding = False  # 适配 OnPolicyRunner 接口

    def __init__(
        self,
        num_actor_obs,    # 对应工厂模式传入的 policy 观测维度
        num_critic_obs,   # 对应工厂模式传入的 critic 观测维度
        num_actions,      # Δq dim
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        gmt_path: str = None,  # Path to load the ONNX model
        **kwargs,
    ):
        """
        Args:
            num_actor_obs (int): input dim of FrontRES
            num_critic_obs (int): unused
            num_actions (int): output dim of FrontRES
            gmt_path (str): GMT ONNX
        """
        if kwargs:
            print(
                "SuperviseLearning.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_name = activation          # keep original string for ActorCritic
        activation = resolve_nn_activation(activation)

        # ========== GMT Tracker (专家模型, .pt checkpoint) ==========
        # Uses the same loading logic as FrontRESActorCritic (Stage 2) so the
        # architecture is always inferred correctly from the checkpoint itself.
        self.gmt_policy: ActorCritic | None = None
        self.gmt_normalizer: EmpiricalNormalization | None = None
        if gmt_path:
            print(f"[SuperviseLearning] Loading GMT policy from: {gmt_path}")
            checkpoint = torch.load(gmt_path, map_location="cpu", weights_only=False)
            sd = checkpoint["model_state_dict"]

            # ---- infer architecture ----
            has_skip = "actor.actor_layer1.weight" in sd
            if has_skip:
                layer1_in  = sd["actor.actor_layer1.weight"].shape[1]
                layer1_out = sd["actor.actor_layer1.weight"].shape[0]
                rem0_in    = sd["actor.actor_remaining.0.weight"].shape[1]
                ref_vel_dim = rem0_in - layer1_out
                gmt_actor_in  = layer1_in + ref_vel_dim
                gmt_critic_in = sd["critic.0.weight"].shape[1]
                rem_keys = [k for k in sd if k.startswith("actor.actor_remaining.") and k.endswith(".weight")]
                last_key = max(rem_keys, key=lambda k: int(k.split(".")[2]))
                gmt_num_actions = sd[last_key].shape[0]
                extra_cfg: dict = {"ref_vel_skip_first_layer": True, "ref_vel_dim": ref_vel_dim}
            else:
                gmt_actor_in  = sd["actor.0.weight"].shape[1]
                gmt_critic_in = sd["critic.0.weight"].shape[1]
                act_keys = [k for k in sd if k.startswith("actor.") and k.endswith(".weight")]
                last_key = max(act_keys, key=lambda k: int(k.split(".")[1]))
                gmt_num_actions = sd[last_key].shape[0]
                extra_cfg = {}

            # hidden dims (all layers except the last output layer)
            if has_skip:
                actor_weight_keys = sorted(
                    [k for k in sd if k.startswith("actor.actor_remaining.") and k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[2]))
            else:
                actor_weight_keys = sorted(
                    [k for k in sd if k.startswith("actor.") and k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[1]))
            actor_hidden_dims = [sd[k].shape[0] for k in actor_weight_keys[:-1]]

            critic_weight_keys = sorted(
                [k for k in sd if k.startswith("critic.") and k.endswith(".weight")],
                key=lambda k: int(k.split(".")[1]))
            critic_hidden_dims = [sd[k].shape[0] for k in critic_weight_keys[:-1]]

            noise_std_type = "scalar" if "std" in sd else "log"
            init_noise_std = (sd["std"][0].item() if "std" in sd
                              else torch.exp(sd["log_std"][0]).item())


            self.gmt_policy = ActorCritic(
                num_actor_obs=gmt_actor_in,
                num_critic_obs=gmt_critic_in,
                num_actions=gmt_num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation_name,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                **extra_cfg,
            )
            self.gmt_policy.load_state_dict(sd)
            self.gmt_policy.eval()
            for p in self.gmt_policy.parameters():
                p.requires_grad = False
            print(f"[SuperviseLearning] GMT policy loaded and frozen "
                  f"(actor_in={gmt_actor_in}, actions={gmt_num_actions})")

            # ---- load frozen obs normalizer ----
            if "obs_norm_state_dict" in checkpoint:
                obs_norm_sd = checkpoint["obs_norm_state_dict"]
                norm_dim = obs_norm_sd["_mean"].shape[1]
                self.gmt_normalizer = EmpiricalNormalization(shape=[norm_dim], until=1.0e8)
                self.gmt_normalizer.load_state_dict(obs_norm_sd)
                self.gmt_normalizer.eval()
                self.gmt_normalizer.until = 0  # freeze statistics
                print(f"[SuperviseLearning] GMT obs normalizer loaded and frozen (dim={norm_dim})")
            else:
                print("[SuperviseLearning] WARNING: no obs_norm_state_dict in GMT checkpoint")

        # ========== student (FrontRES 网络) ==========
        student_layers = [] # 这个 MLP 就是学习预测残差Δq的学生网络
        student_layers.append(nn.Linear(num_actor_obs, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        print(f"[SuperviseLearning] Student MLP: {self.student}")

        # 临时存储分布状态 (兼容 rsl_rl 内部流程)
        self._student_pred = None

    def reset(self, dones=None, hidden_states=None):
        pass

    @property
    def action_mean(self):
        return self._student_pred

    @property
    def action_std(self):
        return torch.zeros_like(self._student_pred) if self._student_pred is not None else None

    @property
    def entropy(self):
        return torch.zeros_like(self._student_pred[:, 0]) if self._student_pred is not None else None

    def forward(self, observations):
        """
        Run the student and return the predicted Δ q
        Args:
            observations (torch.Tensor): student obs
        Returns:
            torch.Tensor: the predicted Δ q
        """
        return self.student(observations)
        
    def update_distribution(self, observations):
        """updating interface to adapt to RL Runner"""
        self._student_pred = self.student(observations)

    def act(self, observations, **kwargs):
        """
        standard interface, invoke at Runner Rollout
        """
        self.update_distribution(observations)
        return self._student_pred

    def get_actions_log_prob(self, actions):
        """Provide a false Log Prob"""
        return torch.zeros_like(actions[:, 0])

    def act_inference(self, observations, **kwargs):
        return self.act(observations)
        
    def evaluate(self, critic_observations, **kwargs):
        """
        Provide a false Critic evaluation interface
        """
        return torch.zeros((critic_observations.shape[0], 1), device=critic_observations.device)

    @torch.no_grad()
    def get_gmt_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run GMT (PyTorch .pt) inference on a batch of raw observations.
        obs is normalised internally by gmt_normalizer before being fed to gmt_policy.
        """
        if self.gmt_policy is None:
            raise RuntimeError("GMT policy is not loaded. Cannot compute GMT action.")

        device = obs.device
        # Normalize with GMT's frozen normalizer (same as Stage 2)
        if self.gmt_normalizer is not None:
            obs = self.gmt_normalizer(obs.to(self.gmt_normalizer._mean.device))
        return self.gmt_policy.act_inference(obs.to(device))

    @staticmethod
    def get_supervision_target(q_sim: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """
        Δ_q_gt = q_ref - q_sim
        """
        delta_q = q_ref - q_sim
        return delta_q

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student network."""
        if any("actor" in key for key in state_dict.keys()):
            student_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    student_state_dict[key.replace("actor.", "")] = value
            self.student.load_state_dict(student_state_dict, strict=strict)
        else:
            super().load_state_dict(state_dict, strict=strict)
        return True

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
