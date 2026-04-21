from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_rotate_inverse

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
    # → env_origins cancel in the difference
    z_ref = command.anchor_pos_w[:, 2:3]                   # (N, 1) world z of reference
    z_sim = command.robot.data.root_pos_w[:, 2:3]          # (N, 1) world z of robot
    delta_z_gt = z_sim - z_ref                             # (N, 1) negative when ref floats above robot

    return torch.cat([delta_q_gt, delta_z_gt], dim=-1)     # (N, 30)
