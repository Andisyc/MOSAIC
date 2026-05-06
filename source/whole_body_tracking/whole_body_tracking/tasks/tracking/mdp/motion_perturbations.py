"""
This module provides domain randomization by applying perturbations to the reference motion.
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
    """Probability of applying foot slip perturbation."""
    foot_slip_ratio: float = 0.0
    """The magnitude of the foot slip as a ratio of the foot's original height."""
    foot_slip_height: float = 0.05
    """The foot height above which the perturbation can be applied."""

    # -- Body float perturbation
    float_prob: float = 0.0
    """Probability of applying float perturbation."""
    float_ratio: float = 0.0
    """The magnitude of the float as a ratio of the body's original height."""

    # -- Body sink perturbation
    sink_prob: float = 0.0
    """Probability of applying sink perturbation."""
    sink_ratio: float = 0.0
    """The magnitude of the sink as a ratio of the body's original height."""

    # -- Root orientation tilt perturbation
    root_tilt_prob: float = 0.0
    """Probability of applying root orientation tilt perturbation."""
    root_tilt_max_rad: float = 0.0
    """Maximum tilt angle in radians applied to root pitch/roll. GMT fails around 3-5 deg (~0.05-0.09 rad)."""

    # -- Lateral drift perturbation (Y direction)
    lateral_drift_prob: float = 0.0
    """Probability of applying lateral (Y-axis) drift perturbation."""
    lateral_drift_std: float = 0.0
    """Standard deviation of Gaussian Y displacement (metres). Covers side-to-side body drift
    artifacts absent from foot_slip (X-only) and float/sink (Z-only). Gives FrontRES a
    supervised training signal for the Δpos[1] output dimension."""

    # -- Joint angle noise perturbation
    joint_noise_prob: float = 0.0
    """Probability of applying joint angle noise perturbation."""
    joint_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to reference joint angles (radians)."""
    joint_noise_joint_indices: list | None = None
    """Joint indices to perturb. None = all joints. Use lower-limb-only indices to avoid polluting upper-limb q_ref."""


class MotionPerturber:
    """
    Applies perturbations to the reference motion data.

    This class is responsible for applying various domain randomizations directly
    to the kinematic reference motion sequence (q_ref). These perturbations simulate
    common motion artifacts or challenging scenarios.
    """

    def __init__(self, cfg: MotionPerturbationCfg, num_envs: int, device: str):
        """
        Initialize the motion perturber.

        Args:
            cfg: Configuration for motion perturbations.
            num_envs: Number of parallel environments.
            device: The device on which to perform the computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    def apply_perturbations(
        self,
        root_pos_ref: torch.Tensor,
        left_foot_pos_ref: torch.Tensor,
        right_foot_pos_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply all configured perturbations to the reference motion.

        Args:
            root_pos_ref: Reference root position of shape (num_envs, 3).
            left_foot_pos_ref: Reference left foot position of shape (num_envs, 3).
            right_foot_pos_ref: Reference right foot position of shape (num_envs, 3).

        Returns:
            The perturbed root position tensor of shape (num_envs, 3).
        """
        perturbed_root_pos = root_pos_ref.clone()

        # Apply foot slip (X direction)
        if self.cfg.foot_slip_prob > 0.0:
            perturbed_root_pos = self._apply_foot_slip(perturbed_root_pos, left_foot_pos_ref, right_foot_pos_ref)

        # Apply lateral drift (Y direction)
        if self.cfg.lateral_drift_prob > 0.0:
            perturbed_root_pos = self._apply_lateral_drift(perturbed_root_pos)

        # Apply float (Z positive)
        if self.cfg.float_prob > 0.0:
            perturbed_root_pos = self._apply_float(perturbed_root_pos)

        # Apply sink (Z negative)
        if self.cfg.sink_prob > 0.0:
            perturbed_root_pos = self._apply_sink(perturbed_root_pos)

        return perturbed_root_pos

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        """Apply root orientation tilt to the anchor quaternion. Input/output shape: (num_envs, 4) as (w,x,y,z)."""
        if self.cfg.root_tilt_prob > 0.0:
            return self._apply_root_tilt(root_quat)
        return root_quat

    def apply_joint_perturbation(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Apply joint angle noise to the reference joint positions. Input/output shape: (num_envs, num_joints)."""
        if self.cfg.joint_noise_prob > 0.0:
            return self._apply_joint_noise(joint_pos)
        return joint_pos

    def _apply_foot_slip(self, root_pos: torch.Tensor, left_foot_pos: torch.Tensor, right_foot_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates foot slip by applying a horizontal displacement to the root.

        This is applied when a foot is in the air, simulating a slip during swing phase.
        """
        slip_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.foot_slip_prob
        if torch.sum(slip_envs) == 0:
            return root_pos

        left_higher = left_foot_pos[:, 2] > self.cfg.foot_slip_height
        right_higher = right_foot_pos[:, 2] > self.cfg.foot_slip_height

        # Apply slip only when one foot is in the air
        can_slip = torch.logical_and(slip_envs, torch.logical_xor(left_higher, right_higher))

        slip_dir = torch.zeros_like(root_pos)
        slip_dir[torch.logical_and(can_slip, left_higher), 0] = 1.0
        slip_dir[torch.logical_and(can_slip, right_higher), 0] = -1.0

        slip_magnitude = self.cfg.foot_slip_ratio * torch.randn_like(root_pos[:, 0])
        root_pos[can_slip, 0] += slip_magnitude[can_slip] * slip_dir[can_slip, 0]

        return root_pos

    def _apply_lateral_drift(self, root_pos: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian Y-axis lateral drift to the root position.

        Simulates side-to-side body position artifacts (e.g., lateral sway in
        motion capture data) that are orthogonal to the forward direction.
        Provides FrontRES with a supervised training signal for the Δpos[1] output.
        """
        drift_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.lateral_drift_prob
        if not torch.any(drift_envs):
            return root_pos

        drift = torch.zeros_like(root_pos)
        drift[:, 1] = torch.randn(self.num_envs, device=self.device) * self.cfg.lateral_drift_std
        root_pos[drift_envs] += drift[drift_envs]
        return root_pos

    def _apply_float(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates the body floating by adding a vertical displacement.
        """
        float_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.float_prob
        if torch.sum(float_envs) == 0:
            return root_pos

        float_displacement = torch.zeros_like(root_pos)
        float_displacement[:, 2] = self.cfg.float_ratio * torch.rand_like(root_pos[:, 2])
        root_pos[float_envs] += float_displacement[float_envs]

        return root_pos

    def _apply_sink(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates the body sinking by subtracting a vertical displacement.
        """
        sink_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.sink_prob
        if torch.sum(sink_envs) == 0:
            return root_pos

        sink_displacement = torch.zeros_like(root_pos)
        sink_displacement[:, 2] = self.cfg.sink_ratio * torch.rand_like(root_pos[:, 2])
        root_pos[sink_envs] -= sink_displacement[sink_envs]

        return root_pos

    def _apply_root_tilt(self, quat: torch.Tensor) -> torch.Tensor:
        """Apply a random pitch/roll tilt to the root orientation quaternion (w,x,y,z).

        Rotates the reference root orientation by a random angle up to root_tilt_max_rad
        around a random horizontal axis. This creates gravitational torque that GMT
        cannot absorb through joint-angle corrections alone.
        """
        tilt_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.root_tilt_prob
        if not torch.any(tilt_envs):
            return quat

        # Random tilt direction in horizontal plane and magnitude
        tilt_dir = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        tilt_mag = torch.rand(self.num_envs, device=self.device) * self.cfg.root_tilt_max_rad

        # Build tilt quaternion: axis=(cos(dir), sin(dir), 0), angle=tilt_mag
        half = tilt_mag * 0.5
        sin_h = torch.sin(half)
        tilt_quat = torch.stack([
            torch.cos(half),
            sin_h * torch.cos(tilt_dir),
            sin_h * torch.sin(tilt_dir),
            torch.zeros_like(tilt_mag),
        ], dim=-1)  # (num_envs, 4)

        # Quaternion multiply: q_out = tilt_quat * quat  (w,x,y,z convention)
        perturbed = quat.clone()
        q1 = tilt_quat[tilt_envs]
        q2 = quat[tilt_envs]
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        perturbed[tilt_envs, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        perturbed[tilt_envs, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        perturbed[tilt_envs, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        perturbed[tilt_envs, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return perturbed

    def _apply_joint_noise(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to reference joint angles.

        Only perturbs cfg.joint_noise_joint_indices (lower limbs when set).
        Upper-limb joints are excluded: large noise there only pollutes q_ref
        without giving FrontRES a meaningful correction signal.
        """
        noise_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.joint_noise_prob
        if not torch.any(noise_envs):
            return joint_pos

        perturbed = joint_pos.clone()
        indices = self.cfg.joint_noise_joint_indices
        if indices is None:
            noise = torch.randn(self.num_envs, joint_pos.shape[1], device=self.device) * self.cfg.joint_noise_std
            perturbed[noise_envs] += noise[noise_envs]
        else:
            idx = torch.tensor(indices, device=self.device, dtype=torch.long)
            noise_full = torch.zeros_like(joint_pos)
            noise_full[:, idx] = torch.randn(self.num_envs, len(indices), device=self.device) * self.cfg.joint_noise_std
            perturbed[noise_envs] += noise_full[noise_envs]

        return perturbed
