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
        for i in range(self.num_transitions_per_env):
            prev_i = max(0, i - 1)
            yield (
                self.observations[i],
                self.target_actions[i],
                self.dones[i],
                self.target_actions[prev_i],  # previous step's Δq_gt for temporal gate
                self.dones[prev_i],           # previous step's done flag for boundary reset
            )

class SuperviseTrainer:
    """Supervised learning algorithm for training FrontRES to output delta_q."""

    def __init__(
        self,
        policy: SuperviseLearning,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "huber",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,  # Distributed training parameters
        lower_limb_indices: list | None = None,  # Joint indices of lower limbs (hip/knee/ankle)
        lower_limb_weight: float = 2.0,          # Static weight multiplier for lower limb joints
        jump_threshold: float = 0.2,             # rad — temporal gate: joints whose Δq_gt jumps
                                                 # more than this between steps are gated out
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

        # Loss type
        if loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type

        # Static lower-limb joint weights (built in init_storage once num_actions is known)
        self.lower_limb_indices = lower_limb_indices or []
        self.lower_limb_weight  = lower_limb_weight
        self.static_weights: torch.Tensor | None = None

        # Temporal gate threshold (radians)
        self.jump_threshold = jump_threshold

        self.num_updates = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """initialize buffer"""
        self.storage = SuperviseStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            action_shape=action_shape,
            device=self.device
        )

        # Build static lower-limb weight tensor now that num_actions is known
        num_actions = action_shape[0]
        self.static_weights = torch.ones(num_actions, device=self.device)
        if self.lower_limb_indices:
            self.static_weights[self.lower_limb_indices] = self.lower_limb_weight
            print(f"[SuperviseTrainer] Lower-limb static weight {self.lower_limb_weight}× "
                  f"applied to {len(self.lower_limb_indices)} joints: {self.lower_limb_indices}")

    def act(self, obs: torch.Tensor, target_delta_q: torch.Tensor) -> torch.Tensor:
        """record delta_q_gt"""
        # Compute the actions
        actions = self.policy.act(obs).detach()
        
        # 使用标准的 transition 流水线记录当前步的数据
        self.transition.observations = obs
        self.transition.target_actions = target_delta_q
        
        return actions

    def process_env_step(self, rewards, dones, infos) -> None:
        """store data into buffer"""
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

        # Accumulators for diagnostic metrics
        sum_pred_norm          = 0.0
        sum_gt_norm            = 0.0
        sum_cos_sim            = 0.0
        sum_valid_ratio        = 0.0
        sum_cascade_gate_ratio = 0.0
        sum_joint_mae          = None  # (num_actions,) tensor, accumulated then averaged

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            # 直接按时间步迭代我们内置的 Buffer
            for obs, target_actions, dones, prev_target_actions, prev_dones in self.storage.generator():
                # Inference of the FrontRES student
                predicted_actions = self.policy.forward(obs)

                # ── Mask 1: sample-level terminal mask ────────────────────────────
                # When done=True the robot has fallen/reset; Δq_gt from that state
                # is unreliable (the reset joint positions are arbitrary).
                # valid: (B, 1) float — 1 = live transition, 0 = terminal
                valid = 1.0 - dones.float()   # (B, 1)

                # ── Mask 2: joint-level temporal gate ─────────────────────────────
                # A sudden jump in Δq_gt between consecutive steps indicates a
                # tracking failure (penetration / floating) that hasn't triggered
                # done=True yet.  Gate out those joints to avoid polluting training.
                #
                # Episode-boundary fix: if the previous step was terminal (done=True),
                # the env was reset and prev_target_actions comes from a fallen/reset
                # state.  Setting prev = current makes jump = 0 so the gate stays open
                # for the first valid step of the new episode.
                prev_done_mask = prev_dones.float()                             # (B, 1) 1=prev was terminal
                safe_prev = (prev_target_actions * (1.0 - prev_done_mask)
                             + target_actions    *          prev_done_mask)     # (B, A)
                jump = (target_actions - safe_prev).abs()                       # (B, A)
                joint_valid = (jump < self.jump_threshold).float()              # (B, A)

                # ── Mask 3: lower-limb cascade mask (sample-level) ────────────────
                # Balance in a humanoid is a global property: if ANY lower-limb
                # joint fails the temporal gate, the entire sample is discarded.
                # Upper-limb joint failures only gate that specific joint.
                if self.lower_limb_indices:
                    lower_stable = joint_valid[:, self.lower_limb_indices]      # (B, |J_lower|)
                    any_lower_fail = lower_stable.min(dim=-1, keepdim=True).values < 0.5  # (B, 1)
                    m_cascade = (~any_lower_fail).float() * valid               # (B, 1)
                else:
                    m_cascade = valid                                            # (B, 1)

                # ── Effective weight: static lower-limb bias × joint gate × cascade mask ──
                # static_weights: (A,), joint_valid: (B, A), m_cascade: (B, 1)
                eff_w = self.static_weights.unsqueeze(0) * joint_valid * m_cascade  # (B, A)
                n_eff = eff_w.sum().clamp(min=1.0)

                # ── Per-joint Huber / MSE loss, then weighted mean ─────────────────
                if self.loss_type == "huber":
                    per_joint_loss = nn.functional.huber_loss(
                        predicted_actions, target_actions, reduction="none")      # (B, A)
                else:
                    per_joint_loss = nn.functional.mse_loss(
                        predicted_actions, target_actions, reduction="none")      # (B, A)

                behavior_loss = (per_joint_loss * eff_w).sum() / n_eff

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # --- Diagnostic metrics (no grad needed) ---
                with torch.no_grad():
                    # 1. valid_ratio: fraction of non-terminal steps (fall rate proxy)
                    sum_valid_ratio += valid.mean().item()

                    # 2. cascade_gate_ratio: fraction of live samples passing lower-limb
                    #    cascade gate (↓ = more pre-fall failures; useful to tune jump_threshold)
                    n_live = valid.sum().clamp(min=1.0)
                    sum_cascade_gate_ratio += m_cascade.sum() / n_live

                    # 3 & 4. pred/gt norms — used to compute delta_q_norm_ratio
                    sum_pred_norm += predicted_actions.norm(dim=-1).mean().item()
                    sum_gt_norm   += target_actions.norm(dim=-1).mean().item()

                    # 5. cosine_similarity: direction alignment ∈ [-1, 1], target → 1.0
                    sum_cos_sim += nn.functional.cosine_similarity(
                        predicted_actions, target_actions, dim=-1).mean().item()

                    # 6 & 7. Per-joint MAE for joint_mae_mean and joint_mae_max
                    joint_mae = (predicted_actions - target_actions).abs().mean(dim=0)  # (A,)
                    if sum_joint_mae is None:
                        sum_joint_mae = joint_mae
                    else:
                        sum_joint_mae = sum_joint_mae + joint_mae

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

        # --- Construct the loss dictionary ---
        assert sum_joint_mae is not None, "Storage was empty — no batches processed."
        mean_pred_norm = sum_pred_norm / cnt
        mean_gt_norm   = sum_gt_norm   / cnt
        mean_joint_mae = sum_joint_mae / cnt  # (num_actions,)
        loss_dict = {
            # Primary convergence signal
            "behavior":           mean_behavior_loss,
            # Trivial-solution guard: pred/gt ≈ 1.0 when calibrated; → 0 means network collapsed
            "delta_q_norm_ratio": mean_pred_norm / (mean_gt_norm + 1e-8),
            # Direction alignment ∈ [-1, 1], target → 1.0
            "cosine_similarity":  sum_cos_sim / cnt,
            # Data quality
            "valid_ratio":        sum_valid_ratio / cnt,         # fraction of non-terminal steps
            "cascade_gate_ratio": sum_cascade_gate_ratio / cnt, # fraction of live samples passing lower-limb cascade
            # Per-joint error
            "joint_mae_mean":     mean_joint_mae.mean().item(),
            "joint_mae_max":      mean_joint_mae.max().item(),
        }

        return loss_dict
