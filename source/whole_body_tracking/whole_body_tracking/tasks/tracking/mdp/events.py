from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize the gravity vector."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # sample random gravity values
    gravity = torch.zeros(len(env_ids), 3, device=env.device)
    gravity[:, 0] = math_utils.sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=env.device)
    gravity[:, 1] = math_utils.sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=env.device)
    gravity[:, 2] = math_utils.sample_uniform(z_range[0], z_range[1], (len(env_ids),), device=env.device)

    # set the new gravity values
    env.scene.physics_sim_view.set_gravity(gravity, env_ids=env_ids)


def randomize_actuator_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float],
    damping_range: tuple[float, float],
):
    """Randomize the stiffness and damping of the actuators."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    # Get original actuator properties
    actuator_props = asset.actuator_props_cfg.to_dict()
    original_stiffness = torch.tensor([prop.stiffness for prop in actuator_props["control_cfg"]], device=env.device)
    original_damping = torch.tensor([prop.damping for prop in actuator_props["control_cfg"]], device=env.device)

    # sample random stiffness and damping multipliers
    stiffness_multipliers = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1], (len(env_ids), len(joint_ids)), device=env.device
    )
    damping_multipliers = math_utils.sample_uniform(
        damping_range[0], damping_range[1], (len(env_ids), len(joint_ids)), device=env.device
    )

    # apply the multipliers to the original values
    new_stiffness = original_stiffness[joint_ids] * stiffness_multipliers
    new_damping = original_damping[joint_ids] * damping_multipliers

    # set the new actuator properties
    asset.write_actuator_stiffness_to_sim(new_stiffness, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_actuator_damping_to_sim(new_damping, joint_ids=joint_ids, env_ids=env_ids)


def add_payload_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
):
    """Add a random payload mass to a specified body."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    # sample random mass values
    masses = math_utils.sample_uniform(mass_range[0], mass_range[1], (len(env_ids),), device=env.device)

    # get the current mass of the bodies
    body_masses = asset.root_physx_view.get_masses(clone=True)

    # add the payload mass
    body_masses[env_ids[:, None], body_ids] += masses.unsqueeze(1)

    # set the new masses
    asset.root_physx_view.set_masses(body_masses, env_ids=env_ids)
