"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    See scripts/batch_csv_to_npz.py for the detailed rationale.
    """

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        if isinstance(user_site, str) and p == user_site:
            return True
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, required=True, help="the path to the motion file.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot platform name used to replay this motion (e.g. g1, h1_2).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Robot platform registry
##
from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


def create_replay_scene_cfg(robot_cfg: ArticulationCfg):
    """Create a scene configuration for replaying motions with the specified robot."""

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    # motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    motion_file = args_cli.motion_file

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    # Resolve robot platform from registry
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg_class = create_replay_scene_cfg(robot_platform.cfg)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
