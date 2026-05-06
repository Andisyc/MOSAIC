from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_rotate_inverse, quat_mul, quat_inv

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def ref_base_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference base linear velocity in the robot's base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Get reference anchor linear velocity in world frame
    ref_lin_vel_w = command.anchor_lin_vel_w

    # Transform to robot's base frame using inverse quaternion rotation
    ref_lin_vel_b = quat_rotate_inverse(command.anchor_quat_w, ref_lin_vel_w)

    return ref_lin_vel_b.view(env.num_envs, -1)


def ref_projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference projected gravity in the reference motion's base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # World frame gravity vector [0, 0, -1]
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_w[:, 2] = -1.0

    # Transform to reference motion's base frame using inverse quaternion rotation
    ref_gravity_b = quat_rotate_inverse(command.anchor_quat_w, gravity_w)

    return ref_gravity_b.view(env.num_envs, -1)

def body_pos_relative_w(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.body_pos_relative_w.reshape(command.body_pos_relative_w.size(0), -1)

def body_quat_relative_w(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.body_quat_relative_w.reshape(command.body_quat_relative_w.size(0), -1)

def selected_keypoints_pos_w_heading(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.selected_keypoints_pos_w_heading.reshape(command.selected_keypoints_pos_w_heading.size(0), -1)

def anchor_root_height_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    Root z-height error: robot root z minus reference anchor z (world frame). Shape (N, 1).

    Positive  -> robot is ABOVE reference (sink artifact).
    Negative  -> robot is BELOW reference (float artifact: reference drifted upward).

    Sign convention matches delta_z_gt in get_supervision_target_delta_q_z so that
    FrontRES learns the identity mapping: output Δz ≈ this input value.
    env_origins cancel (both anchor_pos_w and root_pos_w include them).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    z_ref = command.anchor_pos_w[:, 2:3]          # (N, 1) reference root z (world)
    z_sim = command.robot.data.root_pos_w[:, 2:3]  # (N, 1) robot root z (world)
    return z_sim - z_ref                           # (N, 1)


def anchor_root_pos_error_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    Root position error in world frame: robot_root_pos_w - anchor_pos_w. Shape (N, 3).

    Positive x/y/z means robot is ahead/left/above the reference anchor.
    Sign convention: FrontRES should output this correction to bring anchor to robot.
    env_origins cancel (both anchor_pos_w and root_pos_w are in world frame).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_robot  = command.robot.data.root_pos_w   # (N, 3) robot root world pos
    pos_anchor = command.anchor_pos_w            # (N, 3) reference anchor world pos
    return pos_robot - pos_anchor                # (N, 3)


def anchor_root_rpy_error_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    Root orientation error as RPY (roll, pitch, yaw). Shape (N, 3).

    q_rel = quat_inv(q_anchor) * q_robot: rotation to align anchor frame with robot frame.
    RPY uses ZYX intrinsic convention, quaternions in (w,x,y,z) format.
    Sign convention: FrontRES should output this correction to bring anchor orientation to robot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    q_anchor = command.anchor_quat_w              # (N, 4) wxyz
    q_robot  = command.robot.data.root_quat_w     # (N, 4) wxyz
    q_rel    = quat_mul(quat_inv(q_anchor), q_robot)  # (N, 4)
    w, x, y, z = q_rel[:, 0], q_rel[:, 1], q_rel[:, 2], q_rel[:, 3]
    roll  = torch.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    pitch = torch.asin((2.0 * (w*y - z*x)).clamp(-1.0, 1.0))
    yaw   = torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=-1)  # (N, 3)


def get_supervision_target_delta_q(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    Computes the supervision target delta_q = q_ref - q_sim.
    Kept for backward compatibility. New code should use get_supervision_target_delta_q_z.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    q_ref = command.joint_pos
    q_sim = command.robot_joint_pos
    return q_ref - q_sim


def get_supervision_target_delta_q_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    Computes combined supervision target [delta_q (29), delta_z (1)] for FrontRES Stage 1.

    delta_q = q_ref - q_sim   — joint angle corrections (rad), shape (N, 29)
    delta_z = z_sim - z_ref   — root z correction (m),        shape (N, 1)

    Sign convention for delta_z matches the Stage 2 application:
      anchor_z_corrected = anchor_z + delta_z
    We want the corrected reference to reach the robot, so delta_z = z_sim - z_ref.
    Negative delta_z means the reference is floating above the robot (float artifact).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    q_ref = command.joint_pos       # (N, 29) reference joint positions
    q_sim = command.robot_joint_pos # (N, 29) simulated joint positions
    delta_q_gt = q_ref - q_sim      # (N, 29)

    # anchor_pos_w already includes env_origins; root_pos_w is also world-frame
    # -> env_origins cancel in the difference
    z_ref = command.anchor_pos_w[:, 2:3]                   # (N, 1) world z of reference
    z_sim = command.robot.data.root_pos_w[:, 2:3]          # (N, 1) world z of robot
    delta_z_gt = z_sim - z_ref                             # (N, 1) negative when ref floats above robot

    return torch.cat([delta_q_gt, delta_z_gt], dim=-1)     # (N, 30)


def get_supervision_target_task_space(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    6-dim task-space supervision target for FrontRES Stage 1.
    [Δx, Δy, Δz, Δroll, Δpitch, Δyaw] — SE(3) ordering.

    Target = anti-DR correction: the SE(3) delta that UNDOES the DR perturbation.
    Uses the perturber's known delta as ground truth (available because we control
    the simulation).  When no perturber is active the delta is zero (identity).

    Position:  target_pos = -(perturbed_pos - clean_pos)  = clean_pos - perturbed_pos
    Orientation: quat_inv(perturbed_quat) * clean_quat → convert to RPY

    This replaces the old identity-mapping target (robot - anchor) which taught
    FrontRES to "chase the DR-perturbed robot" rather than "undo the DR".
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Position: undo DR perturbation (world-frame delta)
    delta_pos = -command.anchor_dr_delta_pos  # (N, 3)  anti-DR

    # Orientation: convert the quaternion correction to RPY
    q_corr = command.anchor_dr_delta_quat_correction  # (N, 4) wxyz
    w, x, y, z = q_corr[:, 0], q_corr[:, 1], q_corr[:, 2], q_corr[:, 3]
    roll  = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw   = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    delta_rpy = torch.stack([roll, pitch, yaw], dim=-1)  # (N, 3)

    return torch.cat([delta_pos, delta_rpy], dim=-1)     # (N, 6)
