"""
Domain randomization for reference motion via OU + IID step-jump perturbations.

OU: smooth, temporally correlated drift (camera jitter, slow drift).
    GMT absorbs OU on Z (leg suspension), partially on XY (step relocation).
IID: per-step independent jumps (frame drops, ICP failure, feature jumps).
    IID on Z/RP breaks dynamic continuity → GMT fails → FrontRES rescues.

Per-axis design:
  Z:      IID dominant (OU signal absorbed by leg suspension)
  XY:     OU + IID (OU gives cumulative drift signal, IID gives rescue signal)
  RP:     OU + IID (tilt always destabilizing, both signals strong)
  Yaw:    IID dominant (OU yaw change absorbed by turning)

dr_scale controls IID magnitude same as OU magnitude.
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

    # -- IID step-jump probabilities (per-axis, per-step)
    iid_prob_z:  float = 0.0   # Z axis:  float/sink step jump probability
    iid_prob_xy: float = 0.0   # XY axis: lateral drift step jump probability
    iid_prob_rp: float = 0.0   # Roll/Pitch: tilt step jump probability
    iid_prob_ya: float = 0.0   # Yaw: orientation step jump probability

    # -- IID step-jump magnitudes (std, scaled by dr_scale at runtime)
    iid_std_z:  float = 0.05   # Z jump std (metres) — e.g. 5cm before scale
    iid_std_xy: float = 0.03   # XY jump std (metres)
    iid_std_rp: float = 0.05   # Roll/Pitch jump std (radians) — ~2.9°
    iid_std_ya: float = 0.05   # Yaw jump std (radians)

    # -- Local root-frame artifact bursts
    local_root_artifact_prob: float = 0.0
    """Probability per env per step of starting a short root XY/Yaw artifact burst."""
    local_root_artifact_min_steps: int = 3
    """Minimum duration of a local root artifact burst in sim steps."""
    local_root_artifact_max_steps: int = 8
    """Maximum duration of a local root artifact burst in sim steps."""
    local_root_artifact_xy_std: float = 0.0
    """Std of fixed XY offset sampled at burst start (metres), scaled by dr_scale."""
    local_root_artifact_yaw_std: float = 0.0
    """Std of fixed yaw offset sampled at burst start (radians), scaled by dr_scale."""


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

        # Short-window artifact state.  Unlike OU drift, this is a piecewise
        # constant local error: the same wrong root XY/Yaw is held for a few
        # frames, creating an executable discontinuity rather than a harmless
        # global reference-frame shift.
        self._artifact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._artifact_xy = torch.zeros(num_envs, 2, device=device)
        self._artifact_yaw = torch.zeros(num_envs, device=device)

        # Baseline env mask: these envs always receive zero perturbation.
        self._baseline_mask: torch.Tensor | None = None  # bool [num_envs]

        # DR scale (set by runner each iteration, multiplied into IID magnitudes)
        self._dr_scale: float = 0.0

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
        self._artifact_steps[env_ids] = 0
        self._artifact_xy[env_ids] = 0.0
        self._artifact_yaw[env_ids] = 0.0

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Zero OU states for terminated or resampled environments."""
        if env_ids.numel() == 0:
            return
        self._z_state[env_ids]     = 0.0
        self._x_state[env_ids]     = 0.0
        self._y_state[env_ids]     = 0.0
        self._roll_state[env_ids]  = 0.0
        self._pitch_state[env_ids] = 0.0
        self._artifact_steps[env_ids] = 0
        self._artifact_xy[env_ids] = 0.0
        self._artifact_yaw[env_ids] = 0.0
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

        # ── Superimpose IID step-jumps ──────────────────────────────────────
        self._update_local_root_artifact(root_pos_ref, left_foot_pos_ref, right_foot_pos_ref)
        perturbed[:, :2] += self._artifact_xy
        perturbed = self._apply_iid_xy(perturbed)
        perturbed = self._apply_iid_z(perturbed)

        return perturbed

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        """Apply roll/pitch OU tilt + IID jitter to the anchor quaternion."""
        result = root_quat

        # OU tilt
        if self.cfg.root_tilt_prob > 0.0:
            self._roll_state  = self._ou_step(self._roll_state,  self.cfg.root_tilt_max_rad)
            self._pitch_state = self._ou_step(self._pitch_state, self.cfg.root_tilt_max_rad)
            roll, pitch = self._roll_state, self._pitch_state
            cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
            cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
            tw, tx, ty, tz = cr * cp, sr * cp, cr * sp, sr * sp
            w2, x2, y2, z2 = result[:, 0], result[:, 1], result[:, 2], result[:, 3]
            result = torch.stack([
                tw * w2 - tx * x2 - ty * y2 - tz * z2,
                tw * x2 + tx * w2 + ty * z2 - tz * y2,
                tw * y2 - tx * z2 + ty * w2 + tz * x2,
                tw * z2 + tx * y2 - ty * x2 + tz * w2,
            ], dim=-1)

        # IID jitter (roll/pitch/yaw)
        result = self._apply_iid_quat(result)
        result = self._apply_artifact_yaw(result)
        return result

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

    # ── IID step-jump perturbations (superimposed on OU) ──────────────────

    def _iid_jump(self, prob: float, std: float, num_envs: int) -> torch.Tensor:
        """Per-env IID step jump: each env gets a jump with probability `prob`.
        Magnitude = std × dr_scale × N(0,1).  Returns (num_envs,) tensor."""
        if prob <= 0.0 or std <= 0.0 or self._dr_scale <= 0.0:
            return torch.zeros(num_envs, device=self.device)
        mask = torch.rand(num_envs, device=self.device) < prob
        if self._baseline_mask is not None:
            mask = mask & ~self._baseline_mask
        if not mask.any():
            return torch.zeros(num_envs, device=self.device)
        jump = torch.randn(num_envs, device=self.device) * std * self._dr_scale
        jump[~mask] = 0.0
        return jump

    def _apply_iid_xy(self, root_pos: torch.Tensor) -> torch.Tensor:
        """Superimpose IID XY jitter on root position.  Modifies in-place."""
        jx = self._iid_jump(self.cfg.iid_prob_xy, self.cfg.iid_std_xy, root_pos.shape[0])
        jy = self._iid_jump(self.cfg.iid_prob_xy, self.cfg.iid_std_xy, root_pos.shape[0])
        root_pos[:, 0] += jx
        root_pos[:, 1] += jy
        return root_pos

    def _apply_iid_z(self, root_pos: torch.Tensor) -> torch.Tensor:
        """Superimpose IID Z jitter on root position.  Modifies in-place."""
        jz = self._iid_jump(self.cfg.iid_prob_z, self.cfg.iid_std_z, root_pos.shape[0])
        root_pos[:, 2] += jz
        return root_pos

    def _apply_iid_quat(self, root_quat: torch.Tensor) -> torch.Tensor:
        """Superimpose IID roll/pitch/yaw jitter on root quaternion."""
        jr = self._iid_jump(self.cfg.iid_prob_rp, self.cfg.iid_std_rp, root_quat.shape[0])
        jp = self._iid_jump(self.cfg.iid_prob_rp, self.cfg.iid_std_rp, root_quat.shape[0])
        jy = self._iid_jump(self.cfg.iid_prob_ya, self.cfg.iid_std_ya, root_quat.shape[0])
        if jr.abs().max() == 0 and jp.abs().max() == 0 and jy.abs().max() == 0:
            return root_quat
        # Build jump quaternion: q_jump = q_yaw * q_pitch * q_roll
        cy, sy = torch.cos(jy * 0.5), torch.sin(jy * 0.5)
        cp, sp = torch.cos(jp * 0.5), torch.sin(jp * 0.5)
        cr, sr = torch.cos(jr * 0.5), torch.sin(jr * 0.5)
        # q_yaw * q_pitch
        tw1 = cy * cp; tx1 = -sy * sp; ty1 = cy * sp; tz1 = sy * cp
        # * q_roll
        tw = tw1 * cr - tx1 * sr
        tx = tw1 * sr + tx1 * cr
        ty = ty1 * cr + tz1 * sr
        tz = -ty1 * sr + tz1 * cr
        w2, x2, y2, z2 = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        return torch.stack([
            tw * w2 - tx * x2 - ty * y2 - tz * z2,
            tw * x2 + tx * w2 + ty * z2 - tz * y2,
            tw * y2 - tx * z2 + ty * w2 + tz * x2,
            tw * z2 + tx * y2 - ty * x2 + tz * w2,
        ], dim=-1)

    def _update_local_root_artifact(
        self,
        root_pos_ref: torch.Tensor,
        left_foot_pos_ref: torch.Tensor,
        right_foot_pos_ref: torch.Tensor,
    ) -> None:
        """Update short-window root XY/Yaw artifact state.

        We intentionally do not translate the full reference body together with
        the anchor.  The anchor jumps while body/contact references remain tied
        to the original motion, which exposes the contact/heading inconsistency
        that a visual pose toolchain can create.
        """
        prob = float(getattr(self.cfg, "local_root_artifact_prob", 0.0))
        xy_std = float(getattr(self.cfg, "local_root_artifact_xy_std", 0.0))
        yaw_std = float(getattr(self.cfg, "local_root_artifact_yaw_std", 0.0))
        if prob <= 0.0 or (xy_std <= 0.0 and yaw_std <= 0.0) or self._dr_scale <= 0.0:
            self._artifact_steps.zero_()
            self._artifact_xy.zero_()
            self._artifact_yaw.zero_()
            return

        active = self._artifact_steps > 0
        self._artifact_steps[active] -= 1
        ended = self._artifact_steps <= 0
        self._artifact_xy[ended] = 0.0
        self._artifact_yaw[ended] = 0.0
        if self._baseline_mask is not None:
            self._artifact_steps[self._baseline_mask] = 0
            self._artifact_xy[self._baseline_mask] = 0.0
            self._artifact_yaw[self._baseline_mask] = 0.0

        inactive = self._artifact_steps <= 0
        min_foot_z = torch.minimum(left_foot_pos_ref[:, 2], right_foot_pos_ref[:, 2])
        contact_like = min_foot_z < 0.12
        # Contact frames are most informative, but do not make the signal depend
        # on a brittle height threshold.  Non-contact frames still receive a
        # smaller probability so every motion family can produce artifacts.
        effective_prob = torch.where(
            contact_like,
            torch.full((self.num_envs,), prob, device=self.device),
            torch.full((self.num_envs,), prob * 0.25, device=self.device),
        )
        start = inactive & (torch.rand(self.num_envs, device=self.device) < effective_prob)
        if self._baseline_mask is not None:
            start = start & ~self._baseline_mask
        if not start.any():
            return

        min_steps = max(1, int(getattr(self.cfg, "local_root_artifact_min_steps", 3)))
        max_steps = max(min_steps, int(getattr(self.cfg, "local_root_artifact_max_steps", 8)))
        num_start = int(start.sum().item())
        durations = torch.randint(min_steps, max_steps + 1, (num_start,), device=self.device)
        self._artifact_steps[start] = durations
        if xy_std > 0.0:
            self._artifact_xy[start] = (
                torch.randn(num_start, 2, device=self.device) * xy_std * self._dr_scale
            )
        if yaw_std > 0.0:
            self._artifact_yaw[start] = (
                torch.randn(num_start, device=self.device) * yaw_std * self._dr_scale
            )

    def _apply_artifact_yaw(self, root_quat: torch.Tensor) -> torch.Tensor:
        """Right-multiply the active local artifact yaw onto root_quat."""
        yaw = self._artifact_yaw
        if yaw.abs().max() == 0:
            return root_quat
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        tw = cy
        tx = torch.zeros_like(yaw)
        ty = torch.zeros_like(yaw)
        tz = sy
        w2, x2, y2, z2 = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        return torch.stack([
            tw * w2 - tx * x2 - ty * y2 - tz * z2,
            tw * x2 + tx * w2 + ty * z2 - tz * y2,
            tw * y2 - tx * z2 + ty * w2 + tz * x2,
            tw * z2 + tx * y2 - ty * x2 + tz * w2,
        ], dim=-1)

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
