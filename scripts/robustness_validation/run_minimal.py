"""Minimal validation smoke test: 1 env, 1 epsilon, 1 push, 1 trial."""
import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion",     type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── After Isaac Sim is running ──────────────────────────────────────────
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # noqa

print("[SMOKE] Isaac Sim running, importing modules...")

# ── Minimal env build ───────────────────────────────────────────────────
TASK = "Tracking-Flat-G1-Wo-State-Estimation-v0"

task_entry = gym.envs.registry[TASK]
env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]

# Monkey-patch: replace None events
_orig = getattr(env_cfg_cls, '__post_init__', None)

def _safe_post_init(self_):
    if _orig is not None:
        _orig(self_)
    if getattr(self_, 'events', None) is None:
        from isaaclab.utils import configclass
        @configclass
        class _NoOp:
            pass
        self_.events = _NoOp()
    if getattr(self_, 'curriculum', None) is None:
        from isaaclab.utils import configclass
        @configclass
        class _NoOp:
            pass
        self_.curriculum = _NoOp()

env_cfg_cls.__post_init__ = _safe_post_init

env_cfg = env_cfg_cls()
env_cfg.scene.num_envs = 1
env_cfg.commands.motion.motion = args_cli.motion   # MotionCommandCfg 的正确字段名

# Disable timeout
if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None

# Zero init randomisation
motion_cfg = getattr(env_cfg.commands, "motion", None)
if motion_cfg is not None:
    zero = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.),
            "roll": (0., 0.), "pitch": (0., 0.), "yaw": (0., 0.)}
    for attr in ("pose_range", "velocity_range"):
        if hasattr(motion_cfg, attr):
            setattr(motion_cfg, attr, zero)

print("[SMOKE] Creating env...")
env = gym.make(TASK, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
print(f"[SMOKE] Env created: {env.num_envs} envs")

# ── Load GMT ────────────────────────────────────────────────────────────
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

agent_cfg = G1FlatPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
runner._move_normalizer_to_device(args_cli.device)
runner.load(args_cli.checkpoint, load_optimizer=False, load_critic=False)
print("[SMOKE] GMT loaded")

# ── Run one trial ───────────────────────────────────────────────────────
policy = runner.get_inference_policy(device=args_cli.device)
obs, _ = env.get_observations()

# Disable motion perturbations for clean baseline
env_unwrapped = env.unwrapped
cmd = env_unwrapped.command_manager._terms.get("motion")
if cmd is not None and hasattr(cmd, 'perturber'):
    cfg = cmd.perturber.cfg
    cfg.float_prob = 0.0
    cfg.sink_prob = 0.0
    cfg.root_tilt_prob = 0.0
    cfg.joint_noise_prob = 0.0

print("[SMOKE] Running 300 steps...")
for step in range(300):
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    if step % 50 == 0:
        print(f"   step {step:3d}: done={dones.item()}")

print("[SMOKE] OK — 300 steps completed without crash")
env.close()
simulation_app.close()
print("[SMOKE] Done.")
