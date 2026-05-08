"""
This module provides domain randomization by applying perturbations to the reference motion.

All perturbations use an Ornstein-Uhlenbeck (OU) process so that per-env states evolve
smoothly between steps (β≈0.905 at τ=0.2s, f_cutoff≈0.8Hz) rather than being IID
white noise (equivalent to 25Hz — far outside the 0–3Hz band of real mocap errors).

States are reset to zero on episode boundaries via reset_envs().
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils import configclass


@configclass
class MotionPerturbationCfg:
    """Configuration for motion perturbations applied to the reference motion sequence (q_ref)."""

    # -- Foot slip perturbation
    foot_slip_prob: float = 0.0
    """Probability of applying foot slip perturbation (acts as an enable switch; OU is always on when > 0)."""
    foot_slip_ratio: float = 0.0
    """Steady-state RMS magnitude of the X-axis OU drift (metres)."""
    foot_slip_height: float = 0.05
    """(Legacy, kept for config compatibility.) Foot height threshold — no longer used by OU path."""

    # -- Body float perturbation
    float_prob: float = 0.0
    """Enable switch for Z+ (float) OU perturbation."""
    float_ratio: float = 0.0
    """Steady-state RMS magnitude of the Z-float component (metres)."""

    # -- Body sink perturbation
    sink_prob: float = 0.0
    """Enable switch for Z- (sink) OU perturbation."""
    sink_ratio: float = 0.0
    """Steady-state RMS magnitude of the Z-sink component (metres)."""

    # -- Root orientation tilt perturbation
    root_tilt_prob: float = 0.0
    """Enable switch for roll/pitch OU tilt perturbation."""
    root_tilt_max_rad: float = 0.0
    """Steady-state RMS tilt angle (radians). GMT fails around 3-5 deg (~0.05-0.09 rad)."""

    # -- Lateral drift perturbation (Y direction)
    lateral_drift_prob: float = 0.0
    """Enable switch for lateral Y-axis OU drift."""
    lateral_drift_std: float = 0.0
    """Steady-state RMS magnitude of the Y drift (metres)."""

    # -- Joint angle noise perturbation
    joint_noise_prob: float = 0.0
    """Enable switch for joint-angle OU noise."""
    joint_noise_std: float = 0.0
    """Steady-state RMS noise added to reference joint angles (radians)."""
    joint_noise_joint_indices: list | None = None
    """Joint indices to perturb. None = all joints."""

    # -- OU process parameters
    ou_time_constant: float = 0.2
    """OU time constant τ (seconds).  β = exp(-dt/τ), dt=0.02s.
    τ=0.2s → β≈0.905, f_cutoff≈0.8Hz — covers camera jitter (1-2Hz) without IID step jumps."""


class MotionPerturber:
    """
    Applies smooth domain-randomization perturbations to the reference motion.

    Each perturbation axis has an independent per-environment OU state that evolves
    every physics step.  The OU process:
        state_{t+1} = β * state_t + sqrt(1-β²) * max_magnitude * N(0,1)
    has steady-state std = max_magnitude (i.e. the cfg magnitude fields are RMS values,
    not hard bounds).  States are zeroed at episode boundaries via reset_envs().
    """

    def __init__(self, cfg: MotionPerturbationCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # OU decay: β = exp(-dt/τ),  dt=0.02s (50 Hz physics sim)
        tau = getattr(cfg, 'ou_time_constant', 0.2)
        self._beta = math.exp(-0.02 / tau)

        # Per-env OU states — all start at zero (no perturbation at episode start)
        self._z_state     = torch.zeros(num_envs, device=device)   # float/sink Z offset (m)
        self._x_state     = torch.zeros(num_envs, device=device)   # X slip offset (m)
        self._y_state     = torch.zeros(num_envs, device=device)   # lateral Y offset (m)
        self._roll_state  = torch.zeros(num_envs, device=device)   # root roll (rad)
        self._pitch_state = torch.zeros(num_envs, device=device)   # root pitch (rad)
        self._joint_state: torch.Tensor | None = None              # (num_envs, num_joints)

        # Baseline env mask: these envs always receive zero perturbation.
        # Set once by set_baseline_envs(); enforced inside _ou_step().
        self._baseline_mask: torch.Tensor | None = None  # bool [num_envs]

    # ── public API ────────────────────────────────────────────────────────────

    def set_baseline_envs(self, env_ids: torch.Tensor) -> None:
        """Permanently disable perturbation for designated baseline envs.

        Must be called once during runner setup (before the first env.step).
        Unlike reset_envs(), this cannot be undone by the OU process — _ou_step()
        forces the state to zero for these envs on every physics step.

        This is the fix for ep_len_gmt ≈ 10: even after reset_envs() zeros the
        OU state, the very next _ou_step() produces noise = sqrt(1-β²)*max_mag*randn.
        At dr_scale=4, this single-step noise (std ≈ 0.085 m) has ~43% chance of
        triggering ee_body_pos termination (threshold 0.25 m) within 10 steps.
        """
        self._baseline_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._baseline_mask[env_ids] = True

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Zero OU states for terminated or resampled environments."""
        if env_ids.numel() == 0:
            return
        self._z_state[env_ids]     = 0.0
        self._x_state[env_ids]     = 0.0
        self._y_state[env_ids]     = 0.0
        self._roll_state[env_ids]  = 0.0
        self._pitch_state[env_ids] = 0.0
        if self._joint_state is not None:
            self._joint_state[env_ids] = 0.0

    def apply_perturbations(
        self,
        root_pos_ref: torch.Tensor,
        left_foot_pos_ref: torch.Tensor,
        right_foot_pos_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply X/Y/Z OU perturbations to the reference root position.

        Args:
            root_pos_ref: Reference root position of shape (num_envs, 3).
            left_foot_pos_ref: (unused, kept for API compatibility)
            right_foot_pos_ref: (unused, kept for API compatibility)

        Returns:
            Perturbed root position tensor of shape (num_envs, 3).
        """
        perturbed = root_pos_ref.clone()

        # X: foot-slip-style X offset
        if self.cfg.foot_slip_prob > 0.0:
            self._x_state = self._ou_step(self._x_state, self.cfg.foot_slip_ratio)
            perturbed[:, 0] += self._x_state

        # Y: lateral drift
        if self.cfg.lateral_drift_prob > 0.0:
            self._y_state = self._ou_step(self._y_state, self.cfg.lateral_drift_std)
            perturbed[:, 1] += self._y_state

        # Z: float/sink merged into a single zero-mean OU state
        _z_max = max(self.cfg.float_ratio, self.cfg.sink_ratio)
        if _z_max > 0.0 and (self.cfg.float_prob > 0.0 or self.cfg.sink_prob > 0.0):
            self._z_state = self._ou_step(self._z_state, _z_max)
            perturbed[:, 2] += self._z_state

        return perturbed

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        """Apply roll/pitch OU tilt to the anchor quaternion. Input/output shape: (num_envs, 4) as (w,x,y,z)."""
        if self.cfg.root_tilt_prob <= 0.0:
            return root_quat

        self._roll_state  = self._ou_step(self._roll_state,  self.cfg.root_tilt_max_rad)
        self._pitch_state = self._ou_step(self._pitch_state, self.cfg.root_tilt_max_rad)

        roll, pitch = self._roll_state, self._pitch_state
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)

        # Tilt quaternion = q_roll * q_pitch  (w,x,y,z convention)
        # q_roll=(cr,sr,0,0), q_pitch=(cp,0,sp,0) → product:
        tw, tx, ty, tz = cr * cp, sr * cp, cr * sp, sr * sp

        w2, x2, y2, z2 = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        return torch.stack([
            tw * w2 - tx * x2 - ty * y2 - tz * z2,
            tw * x2 + tx * w2 + ty * z2 - tz * y2,
            tw * y2 - tx * z2 + ty * w2 + tz * x2,
            tw * z2 + tx * y2 - ty * x2 + tz * w2,
        ], dim=-1)

    def apply_joint_perturbation(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Apply joint-angle OU noise to the reference joint positions. Input/output shape: (num_envs, num_joints)."""
        if self.cfg.joint_noise_prob <= 0.0:
            return joint_pos

        num_joints = joint_pos.shape[1]
        if self._joint_state is None:
            self._joint_state = torch.zeros(self.num_envs, num_joints, device=self.device)

        noise_std = self.cfg.joint_noise_std * math.sqrt(1.0 - self._beta ** 2)
        self._joint_state = self._beta * self._joint_state + noise_std * torch.randn_like(self._joint_state)
        if self._baseline_mask is not None:
            self._joint_state[self._baseline_mask] = 0.0

        perturbed = joint_pos.clone()
        if self.cfg.joint_noise_joint_indices is not None:
            idx = torch.tensor(self.cfg.joint_noise_joint_indices, device=self.device, dtype=torch.long)
            perturbed[:, idx] += self._joint_state[:, idx]
        else:
            perturbed += self._joint_state
        return perturbed

    # ── internal ──────────────────────────────────────────────────────────────

    def _ou_step(self, state: torch.Tensor, max_magnitude: float) -> torch.Tensor:
        """Single OU update; steady-state std of the process = max_magnitude.

        Baseline envs (set via set_baseline_envs) are always returned as zero,
        regardless of max_magnitude or the previous state.
        """
        noise_std = max_magnitude * math.sqrt(1.0 - self._beta ** 2)
        new_state = self._beta * state + noise_std * torch.randn_like(state)
        if self._baseline_mask is not None:
            new_state[self._baseline_mask] = 0.0
        return new_state
