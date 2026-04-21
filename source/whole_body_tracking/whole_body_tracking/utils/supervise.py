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
        # Split between joint outputs (Δq, full masking) and aux outputs (Δz, terminal mask only).
        # Must match policy.num_actions (= robot DOFs, e.g. 29 for G1).
        num_joint_outputs: int = 29,
        z_loss_weight: float = 0.5,              # Weight for auxiliary Δz loss vs Δq loss
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

        # Δq / Δz split
        self.num_joint_outputs = num_joint_outputs
        self.z_loss_weight = z_loss_weight

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

        # Build static lower-limb weight tensor for the Δq part only.
        # action_shape[0] = num_joint_outputs + num_z_outputs (e.g. 30).
        # static_weights only covers the first num_joint_outputs dims (Δq).
        self.static_weights = torch.ones(self.num_joint_outputs, device=self.device)
        if self.lower_limb_indices:
            self.static_weights[self.lower_limb_indices] = self.lower_limb_weight
            print(f"[SuperviseTrainer] Lower-limb static weight {self.lower_limb_weight}× "
                  f"applied to {len(self.lower_limb_indices)} joints: {self.lower_limb_indices}")
        num_z = action_shape[0] - self.num_joint_outputs
        print(f"[SuperviseTrainer] Output split: {self.num_joint_outputs} Δq + {num_z} Δz "
              f"(z_loss_weight={self.z_loss_weight})")

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
        sum_joint_mae          = None  # (num_joint_outputs,) tensor
        # Δq / Δz loss separation (for logging)
        sum_dq_loss            = 0.0
        sum_dz_loss            = 0.0
        # Δz diagnostics
        sum_dz_pred_abs        = 0.0
        sum_dz_gt_abs          = 0.0
        sum_dz_mae             = 0.0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            # 直接按时间步迭代我们内置的 Buffer
            for obs, target_actions, dones, prev_target_actions, prev_dones in self.storage.generator():
                # Inference of the FrontRES student
                predicted_actions = self.policy.forward(obs)

                # ── Split into Δq part (joint corrections) and Δz part (z correction) ──
                nj = self.num_joint_outputs
                pred_dq   = predicted_actions[:, :nj]          # (B, 29) Δq predictions
                pred_dz   = predicted_actions[:, nj:]           # (B, num_z) Δz predictions
                target_dq = target_actions[:, :nj]              # (B, 29) Δq ground truth
                target_dz = target_actions[:, nj:]              # (B, num_z) Δz ground truth

                # ── Mask 1: sample-level terminal mask ────────────────────────────
                valid = 1.0 - dones.float()   # (B, 1)

                # ── Mask 2: joint-level temporal gate (Δq only) ───────────────────
                # Applied only to the Δq part; Δz is in meters and can change faster.
                prev_done_mask  = prev_dones.float()
                safe_prev_dq    = (prev_target_actions[:, :nj] * (1.0 - prev_done_mask)
                                   + target_dq                  *          prev_done_mask)
                jump            = (target_dq - safe_prev_dq).abs()              # (B, 29)
                joint_valid     = (jump < self.jump_threshold).float()          # (B, 29)

                # ── Mask 3: lower-limb cascade mask (sample-level) ────────────────
                if self.lower_limb_indices:
                    lower_stable    = joint_valid[:, self.lower_limb_indices]   # (B, |J_lower|)
                    any_lower_fail  = lower_stable.min(dim=-1, keepdim=True).values < 0.5
                    m_cascade       = (~any_lower_fail).float() * valid          # (B, 1)
                else:
                    m_cascade = valid

                # ── Δq loss: static lower-limb bias × joint gate × cascade mask ───
                eff_w  = self.static_weights.unsqueeze(0) * joint_valid * m_cascade  # (B, 29)
                n_eff  = eff_w.sum().clamp(min=1.0)
                if self.loss_type == "huber":
                    per_joint_loss = nn.functional.huber_loss(pred_dq, target_dq, reduction="none")
                else:
                    per_joint_loss = nn.functional.mse_loss(pred_dq, target_dq, reduction="none")
                dq_loss = (per_joint_loss * eff_w).sum() / n_eff

                # ── Δz loss: terminal mask only (no temporal gate on z) ───────────
                if pred_dz.shape[-1] > 0:
                    if self.loss_type == "huber":
                        dz_per = nn.functional.huber_loss(pred_dz, target_dz, reduction="none")
                    else:
                        dz_per = nn.functional.mse_loss(pred_dz, target_dz, reduction="none")
                    n_valid_z  = valid.sum().clamp(min=1.0)
                    dz_loss    = (dz_per * valid).sum() / n_valid_z
                    behavior_loss = dq_loss + self.z_loss_weight * dz_loss
                else:
                    dz_loss       = torch.zeros(1, device=self.device)
                    behavior_loss = dq_loss

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                sum_dq_loss += dq_loss.item()
                sum_dz_loss += dz_loss.item() if pred_dz.shape[-1] > 0 else 0.0
                cnt += 1

                # --- Diagnostic metrics (no grad needed) ---
                with torch.no_grad():
                    # 1. valid_ratio
                    sum_valid_ratio += valid.mean().item()

                    # 2. cascade_gate_ratio
                    n_live = valid.sum().clamp(min=1.0)
                    sum_cascade_gate_ratio += m_cascade.sum() / n_live

                    # 3 & 4. pred/gt norms for Δq
                    sum_pred_norm += pred_dq.norm(dim=-1).mean().item()
                    sum_gt_norm   += target_dq.norm(dim=-1).mean().item()

                    # 5. cosine_similarity on Δq
                    sum_cos_sim += nn.functional.cosine_similarity(
                        pred_dq, target_dq, dim=-1).mean().item()

                    # 6 & 7. Per-joint MAE (Δq only)
                    joint_mae = (pred_dq - target_dq).abs().mean(dim=0)        # (nj,)
                    if sum_joint_mae is None:
                        sum_joint_mae = joint_mae
                    else:
                        sum_joint_mae = sum_joint_mae + joint_mae

                    # 8–10. Δz diagnostics
                    if pred_dz.shape[-1] > 0:
                        sum_dz_pred_abs += pred_dz.abs().mean().item()
                        sum_dz_gt_abs   += target_dz.abs().mean().item()
                        sum_dz_mae      += (pred_dz - target_dz).abs().mean().item()

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
        # Structural check: does the policy have z outputs?
        # sum_dz_pred_abs > 0.0 would be False early in training when Δz ≈ 0, giving wrong result.
        has_dz = getattr(self.policy, 'num_z_outputs', 0) > 0

        loss_dict = {
            # -- Primary convergence signal
            "behavior":           mean_behavior_loss,
            # -- Δq component (joint corrections, full masking)
            "loss_dq":            sum_dq_loss / cnt,
            # -- Δq quality
            "delta_q_norm_ratio": mean_pred_norm / (mean_gt_norm + 1e-8),
            "cosine_similarity":  sum_cos_sim / cnt,
            # -- Data quality
            "valid_ratio":        sum_valid_ratio / cnt,
            "cascade_gate_ratio": sum_cascade_gate_ratio / cnt,
            # -- Per-joint Δq error
            "joint_mae_mean":     mean_joint_mae.mean().item(),
            "joint_mae_max":      mean_joint_mae.max().item(),
        }
        # -- Δz component (only logged when num_z_outputs > 0)
        if has_dz:
            loss_dict["loss_dz"]       = sum_dz_loss / cnt
            loss_dict["dz_pred_abs"]   = sum_dz_pred_abs / cnt   # mean |Δz_pred| (m)
            loss_dict["dz_gt_abs"]     = sum_dz_gt_abs   / cnt   # mean |Δz_gt|   (m)
            loss_dict["dz_mae"]        = sum_dz_mae       / cnt   # mean |Δz_pred − Δz_gt| (m)

        return loss_dict
