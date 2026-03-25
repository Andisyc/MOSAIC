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
import onnxruntime as ort

from rsl_rl.utils import resolve_nn_activation


class SuperviseLearning(nn.Module):
    """
    一个用于监督学习的模块, 旨在训练一个网络(student)来预测由专家模型(GMT)
    在仿真中产生的运动残差。
    这个模块仅包含第一阶段监督训练所需的核心功能。
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
            num_actor_obs (int): 学生网络(FrontRES)的输入维度。
            num_critic_obs (int): 未使用, 仅用于适配 rsl_rl 的统一初始化。
            num_actions (int): 学生网络(FrontRES)的输出维度, 'delta_q_pred'的维度。
            gmt_path (str): 预训练的 GMT ONNX 模型的路径。
        """
        if kwargs:
            print(
                "SuperviseLearning.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # ========== GMT Tracker (专家模型) ==========
        # GMT模型用于在训练循环中生成目标数据 (q_sim)
        self.gmt_session = None
        if gmt_path:
            print(f"[SuperviseLearning] Loading GMT model from: {gmt_path}")
            try:
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if torch.cuda.is_available()
                    else ["CPUExecutionProvider"]
                )
                self.gmt_session = ort.InferenceSession(gmt_path, providers=providers)
                self.gmt_input_name = self.gmt_session.get_inputs()[0].name
                self.gmt_output_name = self.gmt_session.get_outputs()[0].name
                print(f"[SuperviseLearning] GMT model loaded. Input: '{self.gmt_input_name}', Output: '{self.gmt_output_name}'")
            except Exception as e:
                print(f"[SuperviseLearning] Failed to load GMT model from {gmt_path}: {e}")
                self.gmt_session = None

        # ========== student (FrontRES 网络) ==========
        # 这个 MLP 就是学习预测残差Δq的学生网络
        student_layers = []
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
        运行student网络并返回预测的残差Δq。这是监督学习训练的核心。
        Args:
            observations (torch.Tensor): 此处的输入是为学生网络准备的观测。
        Returns:
            torch.Tensor: 预测的残差Δ_q_pred。
        """
        return self.student(observations)
        
    def update_distribution(self, observations):
        """用于适配 RL Runner 的策略分布更新方法"""
        self._student_pred = self.student(observations)

    def act(self, observations, **kwargs):
        """
        标准策略获取接口。在 Runner 的 Rollout 阶段被调用。
        注: 如果您在外部编写了自定义的 SupervisedRunner, 请显式提取 q_ref 并调用 self.get_gmt_action()
        进行仿真环境的步进，而非直接使用此方法的预测输出。
        """
        self.update_distribution(observations)
        return self._student_pred

    def get_actions_log_prob(self, actions):
        """提供一个虚假的 Log Prob，防止在某些统一流程下导致崩溃"""
        return torch.zeros_like(actions[:, 0])

    def act_inference(self, observations, **kwargs):
        """
        act_inference 是 act 的别名，用于提供兼容的API。
        """
        return self.act(observations)
        
    def evaluate(self, critic_observations, **kwargs):
        """
        提供虚假的 Critic 评估接口, 完美适配 OnPolicyRunner 的设计模式
        """
        return torch.zeros((critic_observations.shape[0], 1), device=critic_observations.device)

    @torch.no_grad()
    def get_gmt_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        在给定的完整观测 `obs` 上运行 GMT 专家模型，以获得专家动作。
        这个方法主要在外部的训练脚本中被调用，用于生成 `q_sim`，进而计算出监督学习的目标 `Δ_q_gt`。
        """
        if not self.gmt_session:
            raise RuntimeError("GMT model is not loaded. Cannot compute GMT action.")

        # 健壮的批量推理: 转换至 cpu 并剥离梯度
        gmt_input_np = obs.detach().cpu().numpy().astype("float32")
        gmt_input = {self.gmt_input_name: gmt_input_np}
        gmt_action_np = self.gmt_session.run([self.gmt_output_name], gmt_input)[0]
        return torch.from_numpy(gmt_action_np).to(obs.device)

    @staticmethod
    def get_supervision_target(q_sim: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """
        [在训练脚本或自定义Runner中调用]
        计算监督学习的目标: Δ_q_gt = q_ref - q_sim
        这个Δ_q_gt是student网络需要学习预测的ground truth。
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
