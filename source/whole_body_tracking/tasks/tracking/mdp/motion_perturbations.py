"""
This module provides domain randomization by applying perturbations to the reference motion.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

from omni.isaac.core.utils.torch.math import quat_rotate, quat_mul, quat_from_angle_axis
from rsl_rl.utils.config_class import configclass


@configclass
class MotionPerturbationCfg:
    """Configuration for motion perturbations."""

    # -- Foot slip perturbation
    foot_slip_prob: float = 0.0
    """Probability of applying foot slip perturbation."""
    foot_slip_ratio: float = 0.0
    """The magnitude of the foot slip as a ratio of the foot's original height."""
    foot_slip_height: float = 0.05
    """The foot height above which the perturbation can be applied."""

    # -- Body drag perturbation
    body_drag_prob: float = 0.0
    """Probability of applying body drag perturbation."""
    body_drag_ratio: float = 0.0
    """The magnitude of the body drag as a ratio of the body's original velocity."""

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
        root_vel_ref: torch.Tensor,
        left_foot_pos_ref: torch.Tensor,
        right_foot_pos_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply all configured perturbations to the reference motion.

        Args:
            root_pos_ref: Reference root position of shape (num_envs, 3).
            root_vel_ref: Reference root velocity of shape (num_envs, 3).
            left_foot_pos_ref: Reference left foot position of shape (num_envs, 3).
            right_foot_pos_ref: Reference right foot position of shape (num_envs, 3).


        Returns:
            The perturbed root position tensor of shape (num_envs, 3).
        """
        perturbed_root_pos = root_pos_ref.clone()

        # Apply foot slip
        if self.cfg.foot_slip_prob > 0.0:
            perturbed_root_pos = self._apply_foot_slip(perturbed_root_pos, left_foot_pos_ref, right_foot_pos_ref)

        # Apply body drag
        if self.cfg.body_drag_prob > 0.0:
            perturbed_root_pos = self._apply_body_drag(perturbed_root_pos, root_vel_ref)

        # Apply float
        if self.cfg.float_prob > 0.0:
            perturbed_root_pos = self._apply_float(perturbed_root_pos)
        
        # Apply sink
        if self.cfg.sink_prob > 0.0:
            perturbed_root_pos = self._apply_sink(perturbed_root_pos)

        return perturbed_root_pos

    def _apply_foot_slip(self, root_pos: torch.Tensor, left_foot_pos: torch.Tensor, right_foot_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates foot slip by applying a horizontal displacement to the root.

        This is applied when a foot is in the air, simulating a slip during swing phase.
        """
        # Determine which envs to apply the perturbation
        slip_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.foot_slip_prob
        if torch.sum(slip_envs) == 0:
            return root_pos

        # Determine which foot is slipping
        left_higher = left_foot_pos[:, 2] > self.cfg.foot_slip_height
        right_higher = right_foot_pos[:, 2] > self.cfg.foot_slip_height
        
        # Apply slip only when one foot is in the air
        can_slip = torch.logical_and(slip_envs, torch.logical_xor(left_higher, right_higher))
        
        # Calculate slip direction based on which foot is higher
        slip_dir = torch.zeros_like(root_pos)
        slip_dir[torch.logical_and(can_slip, left_higher), 0] = 1.0  # Slip left
        slip_dir[torch.logical_and(can_slip, right_higher), 0] = -1.0 # Slip right

        # Apply slip
        slip_magnitude = self.cfg.foot_slip_ratio * torch.randn_like(root_pos[:, 0])
        root_pos[can_slip, 0] += slip_magnitude[can_slip] * slip_dir[can_slip, 0]
        
        return root_pos

    def _apply_body_drag(self, root_pos: torch.Tensor, root_vel: torch.Tensor) -> torch.Tensor:
        """
        Simulates body drag by applying a displacement opposite to the direction of motion.
        """
        drag_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.body_drag_prob
        if torch.sum(drag_envs) == 0:
            return root_pos

        # Calculate drag displacement
        drag_displacement = -root_vel * self.cfg.body_drag_ratio
        
        # Apply drag
        root_pos[drag_envs] += drag_displacement[drag_envs]

        return root_pos

    def _apply_float(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates the body floating by adding a vertical displacement.
        """
        float_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.float_prob
        if torch.sum(float_envs) == 0:
            return root_pos
        
        # Calculate float displacement
        float_displacement = torch.zeros_like(root_pos)
        float_displacement[:, 2] = self.cfg.float_ratio * torch.rand_like(root_pos[:, 2])

        # Apply float
        root_pos[float_envs] += float_displacement[float_envs]

        return root_pos
    
    def _apply_sink(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Simulates the body sinking by subtracting a vertical displacement.
        """
        sink_envs = torch.rand(self.num_envs, device=self.device) < self.cfg.sink_prob
        if torch.sum(sink_envs) == 0:
            return root_pos
        
        # Calculate sink displacement
        sink_displacement = torch.zeros_like(root_pos)
        sink_displacement[:, 2] = self.cfg.sink_ratio * torch.rand_like(root_pos[:, 2])

        # Apply sink
        root_pos[sink_envs] -= sink_displacement[sink_envs]

        return root_pos
