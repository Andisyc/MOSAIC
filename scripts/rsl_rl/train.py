# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import os
import argparse
import sys
import faulthandler
import signal

os.environ.setdefault("WANDB_SILENT", "true")
# Redirect wandb local run files to HDD to avoid filling /home
os.environ.setdefault("WANDB_DIR", "/hdd0/yuxuancheng/MOSAIC")
os.environ.setdefault("WANDB_CACHE_DIR", "/hdd0/yuxuancheng/MOSAIC/.wandb_cache")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if WORLD_SIZE > 1:
    base = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"isaaclab_kit_{os.getuid()}")
    rank_dir = os.path.join(base, f"rank{RANK}")
    os.environ.setdefault("OMNI_USER_DIR", rank_dir)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(rank_dir, "cache"))
    os.environ.setdefault("XDG_DATA_HOME",  os.path.join(rank_dir, "data"))
    os.environ.setdefault("XDG_CONFIG_HOME",os.path.join(rank_dir, "config"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument(
    "--max_iterations", 
    type=int, 
    default=None, 
    help="RL Policy training iterations."
)

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--supervised_warmup_iterations",
    type=int,
    default=None,
    help="Override FrontRES supervised warmup iterations before PPO starts.",
)
parser.add_argument(
    "--supervised_warmup_steps_per_iter",
    type=int,
    default=None,
    help="Override simulation steps collected per FrontRES supervised warmup iteration.",
)
parser.add_argument(
    "--supervised_warmup_max_envs_per_step",
    type=int,
    default=None,
    help="Maximum env samples kept from each warmup step for supervised SGD.",
)
parser.add_argument(
    "--is_full_resume",
    type=lambda x: str(x).lower() in ("true", "1", "yes", "y"),
    default=None,
    help=(
        "Resume mode for FrontRES checkpoints. True resumes actor+critic+optimizer+iteration; "
        "False treats the checkpoint as initialization and resets critic/optimizer/iteration."
    ),
)
parser.add_argument(
    "--frontres_debug_training",
    action="store_true",
    default=False,
    help="Enable the shortened FrontRES debug schedule for reward/DR tuning.",
)

# single motion for testing
# motion_path = '/home/chengyuxuan/MOSAIC/motion_npz/dance1_subject1.npz'

parser.add_argument("--motion", type=str, default=None, help="motion or motion file path.") # required=True, 

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print("[DEBUG] Registered SIGUSR1 stack dump. Use: kill -USR1 <pid>", flush=True)

if WORLD_SIZE > 1:
    args_cli.distributed = True
if args_cli.distributed:
    args_cli.device = f"cuda:{LOCAL_RANK}"
if args_cli.distributed and RANK != 0:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args_cli.video = False

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import torch
from datetime import datetime
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@configclass
class _NoOpCfg:
    pass


def _sanitize_env_cfg_for_training(env_cfg) -> None:
    """Avoid IsaacLab startup callbacks failing on None managers/debug visuals."""

    # Some FrontRES configs intentionally disable managers with None, but
    # IsaacLab manager callbacks still assume cfg has a __dict__ during reset.
    for field in ("events", "curriculum"):
        if hasattr(env_cfg, field) and getattr(env_cfg, field) is None:
            setattr(env_cfg, field, _NoOpCfg())

    # Headless/multi-GPU training does not need debug visualization.  Leaving
    # these enabled can register callbacks before Articulation data is ready.
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None and hasattr(motion_cfg, "debug_vis"):
        motion_cfg.debug_vis = False
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    # Large FrontRES runs can create hundreds of thousands of rigid bodies and
    # contacts.  If the PhysX GPU capacities are left at defaults, Omniverse can
    # fail to create the tensor simulation view, which later appears as missing
    # Articulation._data.  Scale the common capacities with num_envs.
    physx = getattr(getattr(env_cfg, "sim", None), "physx", None)
    if physx is not None and hasattr(env_cfg, "scene"):
        num_envs = int(getattr(env_cfg.scene, "num_envs", 0) or 0)

        def _raise_capacity(name: str, value: int) -> None:
            if hasattr(physx, name):
                current = getattr(physx, name)
                if current is None or int(current) < int(value):
                    setattr(physx, name, int(value))

        _raise_capacity("gpu_max_rigid_contact_count", max(2**23, num_envs * 4096))
        _raise_capacity("gpu_max_rigid_patch_count", max(15 * 2**17, num_envs * 512))
        _raise_capacity("gpu_found_lost_pairs_capacity", max(2**24, num_envs * 2048))
        _raise_capacity("gpu_found_lost_aggregate_pairs_capacity", max(2**23, num_envs * 1024))
        _raise_capacity("gpu_total_aggregate_pairs_capacity", max(2**23, num_envs * 1024))
        _raise_capacity("gpu_collision_stack_size", max(2**26, num_envs * 8192))
        _raise_capacity("gpu_heap_capacity", max(2**27, num_envs * 16384))
        _raise_capacity("gpu_temp_buffer_capacity", max(2**26, num_envs * 8192))

        print(
            "[INFO] PhysX GPU capacities prepared for "
            f"{num_envs} envs: "
            f"contact={getattr(physx, 'gpu_max_rigid_contact_count', 'N/A')}, "
            f"patch={getattr(physx, 'gpu_max_rigid_patch_count', 'N/A')}, "
            f"pairs={getattr(physx, 'gpu_found_lost_pairs_capacity', 'N/A')}",
            flush=True,
        )


def _configure_frontres_motion_perturbations(env_cfg, agent_cfg) -> None:
    """Align motion perturbation channels with the FrontRES action mask."""
    if not hasattr(env_cfg, "motion_perturbations"):
        return
    mode = str(getattr(agent_cfg, "frontres_perturbation_channels", "all")).lower()
    pt = env_cfg.motion_perturbations

    if mode in ("all", "composite", "full"):
        # Explicitly mirror the agent-side full-output test settings into the
        # environment perturbation config.  Without this, "all" silently falls
        # back to whatever the task cfg default happens to contain, making it
        # hard to tell whether the joint test is really exercising every
        # controllable channel.
        for name in (
            "float_prob",
            "float_ratio",
            "sink_prob",
            "sink_ratio",
            "foot_slip_prob",
            "foot_slip_ratio",
            "lateral_drift_prob",
            "lateral_drift_std",
            "root_tilt_prob",
            "root_tilt_max_rad",
            "joint_noise_prob",
            "joint_noise_std",
            "iid_prob_z",
            "iid_std_z",
            "iid_prob_xy",
            "iid_std_xy",
            "iid_prob_rp",
            "iid_std_rp",
            "iid_prob_ya",
            "iid_std_ya",
            "local_root_artifact_prob",
            "local_root_artifact_xy_std",
            "local_root_artifact_yaw_std",
        ):
            if hasattr(agent_cfg, name) and hasattr(pt, name):
                setattr(pt, name, type(getattr(pt, name))(getattr(agent_cfg, name)))
        for name in ("local_root_artifact_min_steps", "local_root_artifact_max_steps"):
            if hasattr(agent_cfg, name) and hasattr(pt, name):
                setattr(pt, name, int(getattr(agent_cfg, name)))
        print(
            "[INFO] FrontRES perturbation alignment: all "
            f"(float={pt.float_prob}/{pt.float_ratio}, sink={pt.sink_prob}/{pt.sink_ratio}, "
            f"foot_slip={pt.foot_slip_prob}/{pt.foot_slip_ratio}, "
            f"lateral={pt.lateral_drift_prob}/{pt.lateral_drift_std}, "
            f"root_tilt={pt.root_tilt_prob}/{pt.root_tilt_max_rad}, "
            f"iid_xy={pt.iid_prob_xy}/{pt.iid_std_xy}, iid_z={pt.iid_prob_z}/{pt.iid_std_z}, "
            f"iid_rp={pt.iid_prob_rp}/{pt.iid_std_rp}, iid_yaw={pt.iid_prob_ya}/{pt.iid_std_ya}, "
            f"local_artifact={pt.local_root_artifact_prob}/"
            f"{pt.local_root_artifact_xy_std}/{pt.local_root_artifact_yaw_std}/"
            f"{pt.local_root_artifact_min_steps}-{pt.local_root_artifact_max_steps})",
            flush=True,
        )
        return
    if mode not in ("xy_yaw", "xy-yaw", "xyyaw", "z_rp", "z-rp", "zrp", "rp_z", "rp-z", "rpz", "vertical_contact"):
        raise ValueError(
            "frontres_perturbation_channels must be one of "
            "{'all', 'composite', 'full', 'xy_yaw', 'z_rp', 'rp_z', 'vertical_contact'}; got "
            f"{mode!r}."
        )

    # Disable all generic channels first, then re-enable only the channels
    # controllable by the selected FrontRES task-space action mask.
    pt.float_prob = 0.0
    pt.float_ratio = 0.0
    pt.sink_prob = 0.0
    pt.sink_ratio = 0.0
    pt.foot_slip_prob = 0.0
    pt.foot_slip_ratio = 0.0
    pt.lateral_drift_prob = 0.0
    pt.lateral_drift_std = 0.0
    pt.root_tilt_prob = 0.0
    pt.root_tilt_max_rad = 0.0
    pt.joint_noise_prob = 0.0
    pt.joint_noise_std = 0.0
    pt.iid_prob_z = 0.0
    pt.iid_std_z = 0.0
    pt.iid_prob_rp = 0.0
    pt.iid_std_rp = 0.0

    if mode in ("xy_yaw", "xy-yaw", "xyyaw"):
        # X/Y/Yaw are injected as short local root artifacts, not global drift:
        # a brief anchor jump breaks contact/heading consistency and gives both
        # supervised warmup and PPO a clear executable signal.
        pt.iid_prob_xy = float(getattr(agent_cfg, "iid_prob_xy", pt.iid_prob_xy))
        pt.iid_std_xy = float(getattr(agent_cfg, "iid_std_xy", pt.iid_std_xy))
        pt.iid_prob_ya = float(getattr(agent_cfg, "iid_prob_ya", pt.iid_prob_ya))
        pt.iid_std_ya = float(getattr(agent_cfg, "iid_std_ya", pt.iid_std_ya))
        pt.local_root_artifact_prob = float(getattr(
            agent_cfg, "local_root_artifact_prob", getattr(pt, "local_root_artifact_prob", 0.0)))
        pt.local_root_artifact_min_steps = int(getattr(
            agent_cfg, "local_root_artifact_min_steps", getattr(pt, "local_root_artifact_min_steps", 3)))
        pt.local_root_artifact_max_steps = int(getattr(
            agent_cfg, "local_root_artifact_max_steps", getattr(pt, "local_root_artifact_max_steps", 8)))
        pt.local_root_artifact_xy_std = float(getattr(
            agent_cfg, "local_root_artifact_xy_std", getattr(pt, "local_root_artifact_xy_std", 0.0)))
        pt.local_root_artifact_yaw_std = float(getattr(
            agent_cfg, "local_root_artifact_yaw_std", getattr(pt, "local_root_artifact_yaw_std", 0.0)))
        print(
            "[INFO] FrontRES perturbation alignment: xy_yaw "
            f"(iid_xy={pt.iid_prob_xy}/{pt.iid_std_xy}, iid_yaw={pt.iid_prob_ya}/{pt.iid_std_ya}; "
            f"local_artifact={pt.local_root_artifact_prob}/"
            f"{pt.local_root_artifact_xy_std}/{pt.local_root_artifact_yaw_std}/"
            f"{pt.local_root_artifact_min_steps}-{pt.local_root_artifact_max_steps}; "
            "z/rp/joint disabled)",
            flush=True,
        )
        return

    # Z/Roll/Pitch experiment: only vertical float/sink and root tilt/IID
    # perturbations are enabled, matching active dims [dz, droll, dpitch].
    pt.float_prob = float(getattr(agent_cfg, "float_prob", 0.3))
    pt.float_ratio = float(getattr(agent_cfg, "float_ratio", 0.05))
    pt.sink_prob = float(getattr(agent_cfg, "sink_prob", 0.3))
    pt.sink_ratio = float(getattr(agent_cfg, "sink_ratio", 0.04))
    pt.root_tilt_prob = float(getattr(agent_cfg, "root_tilt_prob", 0.3))
    pt.root_tilt_max_rad = float(getattr(agent_cfg, "root_tilt_max_rad", 0.05))
    pt.iid_prob_z = float(getattr(agent_cfg, "iid_prob_z", pt.iid_prob_z))
    pt.iid_std_z = float(getattr(agent_cfg, "iid_std_z", pt.iid_std_z))
    pt.iid_prob_rp = float(getattr(agent_cfg, "iid_prob_rp", pt.iid_prob_rp))
    pt.iid_std_rp = float(getattr(agent_cfg, "iid_std_rp", pt.iid_std_rp))
    print(
        "[INFO] FrontRES perturbation alignment: z_rp "
        f"(float={pt.float_prob}/{pt.float_ratio}, sink={pt.sink_prob}/{pt.sink_ratio}, "
        f"root_tilt={pt.root_tilt_prob}/{pt.root_tilt_max_rad}, "
        f"iid_z={pt.iid_prob_z}/{pt.iid_std_z}, iid_rp={pt.iid_prob_rp}/{pt.iid_std_rp}; "
        "xy/yaw/local/joint disabled)",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point") # 
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations)
    if args_cli.supervised_warmup_iterations is not None:
        agent_cfg.supervised_warmup_iterations = args_cli.supervised_warmup_iterations
    if args_cli.supervised_warmup_steps_per_iter is not None:
        agent_cfg.supervised_warmup_steps_per_iter = args_cli.supervised_warmup_steps_per_iter
    if args_cli.supervised_warmup_max_envs_per_step is not None:
        agent_cfg.supervised_warmup_max_envs_per_step = args_cli.supervised_warmup_max_envs_per_step
    if args_cli.is_full_resume is not None:
        agent_cfg.is_full_resume = args_cli.is_full_resume
    if args_cli.frontres_debug_training:
        agent_cfg.frontres_debug_training = True

    # set seeds (explicit rank offset for distributed to avoid identical sampling across ranks)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    base_seed = int(agent_cfg.seed)
    rank = int(os.environ.get("RANK", "0"))

    # stride avoids overlaps if some components use multiple RNG draws per step
    seed_stride = int(os.environ.get("SEED_STRIDE", "1000"))
    env_seed = base_seed + rank * seed_stride
    env_cfg.seed = env_seed

    # also seed common RNGs to keep per-rank randomness independent
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)

    # specify device
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print(f"[INFO] Distributed seeding: base_seed={base_seed}, rank={rank}, env_seed={env_seed} (stride={seed_stride})")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # load in motion sequence
    env_cfg.commands.motion.motion = args_cli.motion
    _configure_frontres_motion_perturbations(env_cfg, agent_cfg)
    _sanitize_env_cfg_for_training(env_cfg)

    # specify directory for logging experiments
    # log_root_path 根据 experiment_name 自动派生，避免不同训练阶段的 checkpoint 混入同一目录。
    # experiment_name 由各 RunnerCfg 定义（如 "g1_flat_frontres_finetune"、"g1_flat_supervised"）

    # 两台服务器上的 MOSAIC 根目录（不含实验子目录）
    candidate_base_paths = [
        "/workspace/",
        "/hdd0/yuxuancheng/MOSAIC/",
        "/hdd1/cyx/MOSAIC/",
        "/ssd1/cyx/MOSAIC/"
    ]

    # 自动选择第一个真实存在的路径
    base_path = None
    for path in candidate_base_paths:
        if os.path.exists(path):
            base_path = path
            break

    if base_path is None:
        raise FileNotFoundError("No feasible MOSAIC file ")

    # 拼接实验名称
    log_root_path = os.path.join(base_path, agent_cfg.experiment_name)
    # log_root_path = f"/hdd0/yuxuancheng/MOSAIC/{agent_cfg.experiment_name}"
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # If student_checkpoint_path is set as an absolute path in the config, use it directly.
        # This bypasses get_checkpoint_path() which only looks inside the current experiment's
        # log_root_path — cross-experiment loading (e.g. Stage 1 → Stage 2) requires this.
        _direct = getattr(agent_cfg, "student_checkpoint_path", None)
        if _direct is not None and os.path.isfile(str(_direct)):
            resume_path = str(_direct)
            print(f"[INFO]: Loading model checkpoint from direct path: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if int(os.environ.get("RANK", "0")) == 0:
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()
