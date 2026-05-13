"""Tiny validation pipeline: a couple of epsilons/pushes, save results, plot."""
import argparse
import faulthandler
import os
import sys
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)


def _sanitize_python_path_for_isaac() -> None:
    """Avoid loading binary packages from the user's site-packages into Isaac."""

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(path: str) -> bool:
        if not path:
            return False
        if isinstance(user_site, str) and path == user_site:
            return True
        return "/.local/lib/python" in path and "site-packages" in path

    sys.path[:] = [path for path in sys.path if not _is_user_site_path(path)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

from isaaclab.app import AppLauncher

TASK = "General-Tracking-Flat-G1-Wo-State-Estimation-v0"
OUTPUT_DIR = "verify/robustness_validation_minimal"

parser = argparse.ArgumentParser(description="Minimal robustness validation smoke test.")
parser.add_argument("--motion",     type=str, required=True)
parser.add_argument("--checkpoint", "--resume_path", dest="checkpoint", type=str, required=True)
parser.add_argument("--task",       type=str, default=TASK)
parser.add_argument("--file_glob",  type=str, default="*.npz")
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--num_trials", type=int, default=2)
parser.add_argument("--epsilons",   type=str, default="0.0,0.05")
parser.add_argument("--push_velocities", type=str, default="1.0")
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--observe_steps", type=int, default=60)
parser.add_argument("--push_offset_min", type=int, default=0)
parser.add_argument("--push_offset_max", type=int, default=10)
parser.add_argument("--ou_tau", type=float, default=0.5)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--video",      action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument(
    "--startup_timeout",
    type=int,
    default=30,
    help="Dump Python stack traces every N seconds while diagnosing startup hangs. Use 0 to disable.",
)
parser.add_argument(
    "--keep_events",
    action="store_true",
    default=False,
    help="Keep event/curriculum managers instead of disabling them like play.py.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.startup_timeout > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(args_cli.startup_timeout, repeat=True)


def _parse_float_list(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        parser.error(f"Expected comma-separated floats, got {value!r}: {exc}")


def _resolve_motion_input(path: str, file_glob: str) -> tuple[str, str, str]:
    motion_path = Path(path).expanduser()
    if motion_path.is_file():
        return str(motion_path.parent), motion_path.name, str(motion_path)
    if motion_path.is_dir():
        candidates = sorted(motion_path.rglob(file_glob))
        if candidates:
            selected = candidates[0]
            log(f"[SMOKE] --motion is a directory; minimal run uses first .npz: {selected}")
            return str(motion_path), selected.name, str(selected)
    parser.error(
        f"--motion must be a .npz file or a directory containing .npz files: {path}"
    )


EPSILON_VALUES = _parse_float_list(args_cli.epsilons)
PUSH_VELOCITIES = _parse_float_list(args_cli.push_velocities)
args_cli.motion_dir, args_cli.motion_file_glob, args_cli.motion_original = _resolve_motion_input(
    args_cli.motion, args_cli.file_glob
)

if args_cli.video:
    args_cli.enable_cameras = True

if (
    sys.platform.startswith("linux")
    and not args_cli.headless
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    log("[SMOKE] No display detected; forcing --headless for Isaac Sim startup.")
    args_cli.headless = True

# This script does not use Hydra.  Leaving unknown CLI fragments in sys.argv can
# make Kit consume arguments that were intended for other launch paths.
if hydra_args:
    log(f"[SMOKE] Ignoring non-AppLauncher arguments: {hydra_args}")
sys.argv = [sys.argv[0]]

log(f"[SMOKE] Launching Isaac Sim via AppLauncher: device={args_cli.device}, headless={args_cli.headless}")
app_launcher = AppLauncher(args_cli)
log("[SMOKE] AppLauncher constructed; retrieving simulation_app...")
simulation_app = app_launcher.app
log("[SMOKE] simulation_app is ready.")

# ── After Isaac Sim is running ──────────────────────────────────────────
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from metrics import compute_zmp_margin, find_body_index, is_fallen
from ou_injector import configure_ou, reset_ou_states
from plot_results import load_and_plot
from push_controller import PushController
from results_io import ResultsStore, TrialResult

log("[SMOKE] Isaac Sim running, importing modules...")

# ── Minimal env build ───────────────────────────────────────────────────
task_entry = gym.envs.registry[args_cli.task]
env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]

@configclass
class _NoOpCfg:
    pass


# Monkey-patch: replace None managers with empty configs.
_orig = getattr(env_cfg_cls, '__post_init__', None)

def _safe_post_init(self_):
    if _orig is not None:
        _orig(self_)
    if getattr(self_, 'events', None) is None:
        self_.events = _NoOpCfg()
    if getattr(self_, 'curriculum', None) is None:
        self_.curriculum = _NoOpCfg()

env_cfg_cls.__post_init__ = _safe_post_init

env_cfg = env_cfg_cls()
if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
    env_cfg.sim.device = args_cli.device
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.commands.motion.motion = args_cli.motion_dir
if hasattr(env_cfg.commands.motion, "file_glob"):
    env_cfg.commands.motion.file_glob = args_cli.motion_file_glob
    if hasattr(env_cfg.commands.motion, "motion_preload_device"):
        env_cfg.commands.motion.motion_preload_device = args_cli.device
    if hasattr(env_cfg.commands.motion, "motion_dataset_shard_across_gpus"):
        env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
elif hasattr(env_cfg.commands.motion, "motion_file"):
    env_cfg.commands.motion.motion = args_cli.motion_original
    env_cfg.commands.motion.motion_file = args_cli.motion_original
if hasattr(env_cfg.commands.motion, "start_from_beginning"):
    env_cfg.commands.motion.start_from_beginning = True
if hasattr(env_cfg.commands.motion, "start_frame"):
    env_cfg.commands.motion.start_frame = 0

# Disable timeout
if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None

# Disable event/curriculum terms without setting the manager configs to None.
# Isaac Lab registers manager callbacks before sim.reset(), and callbacks assume
# cfg has a __dict__ even when there are no terms.
if not args_cli.keep_events:
    if hasattr(env_cfg, "events"):
        env_cfg.events = _NoOpCfg()
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = _NoOpCfg()

# Headless smoke tests should not register debug-visualization callbacks.  The
# contact sensor callback can block inside PhysX tensor reads during sim.reset().
if hasattr(env_cfg.commands.motion, "debug_vis"):
    env_cfg.commands.motion.debug_vis = False
if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
    env_cfg.scene.contact_forces.debug_vis = False

# Zero init randomisation
motion_cfg = getattr(env_cfg.commands, "motion", None)
if motion_cfg is not None:
    zero = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.),
            "roll": (0., 0.), "pitch": (0., 0.), "yaw": (0., 0.)}
    for attr in ("pose_range", "velocity_range"):
        if hasattr(motion_cfg, attr):
            setattr(motion_cfg, attr, zero)

log("[SMOKE] Creating env...")
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
log("[SMOKE] gym.make returned.")
if args_cli.video:
    video_kwargs = {
        "video_folder": os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "run_minimal"),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
env = RslRlVecEnvWrapper(env)
log(f"[SMOKE] Env created: {env.num_envs} envs")

# ── Load GMT ────────────────────────────────────────────────────────────
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

agent_cfg = G1FlatPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
runner._move_normalizer_to_device(args_cli.device)
runner.load(args_cli.checkpoint, load_optimizer=False, load_critic=False)
log("[SMOKE] GMT loaded")

env_unwrapped = env.unwrapped
robot = env_unwrapped.scene["robot"]
left_foot_idx = find_body_index(env_unwrapped, "left_ankle_roll_link")
right_foot_idx = find_body_index(env_unwrapped, "right_ankle_roll_link")
policy = runner.get_inference_policy(device=args_cli.device)
push_ctrl = PushController(
    num_envs=env.num_envs,
    device=args_cli.device,
    push_offset_range=(args_cli.push_offset_min, args_cli.push_offset_max),
)


def _run_tiny_condition(epsilon: float, delta_v: float) -> list[TrialResult]:
    configure_ou(env_unwrapped, epsilon, tau=args_cli.ou_tau)
    obs, _ = env.reset()
    env_ids_all = torch.arange(env.num_envs, device=args_cli.device)
    reset_ou_states(env_unwrapped, env_ids_all)

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=args_cli.device)
    fallen_during_settle = torch.zeros(env.num_envs, dtype=torch.bool, device=args_cli.device)
    settle_zmp = [[] for _ in range(env.num_envs)]
    post_zmp = [[] for _ in range(env.num_envs)]

    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        just_fallen = is_fallen(env_unwrapped) | dones.bool()
        newly_fallen = just_fallen & alive
        fallen_during_settle |= newly_fallen
        alive &= ~newly_fallen

        zmp = compute_zmp_margin(env_unwrapped, left_foot_idx, right_foot_idx)
        for i in range(env.num_envs):
            if alive[i] or newly_fallen[i]:
                settle_zmp[i].append(float(zmp[i]))

    push_ctrl.randomize(torch.where(alive)[0])

    for step in range(args_cli.observe_steps):
        push_ctrl.maybe_push(robot, observe_step=step, delta_v=delta_v, alive=alive)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        just_fallen = is_fallen(env_unwrapped) | dones.bool()
        newly_fallen = just_fallen & alive
        alive &= ~newly_fallen

        zmp = compute_zmp_margin(env_unwrapped, left_foot_idx, right_foot_idx)
        for i in range(env.num_envs):
            if not fallen_during_settle[i] and (alive[i] or newly_fallen[i]):
                post_zmp[i].append(float(zmp[i]))

    success = alive & ~fallen_during_settle
    return [
        TrialResult(
            success=bool(success[i]),
            fallen_before_push=bool(fallen_during_settle[i]),
            T_push_step=int(push_ctrl.push_at_step[i]),
            zmp_margins_settle=settle_zmp[i],
            zmp_margins_post=post_zmp[i],
            push_dir=push_ctrl.push_dir[i].detach().cpu().tolist(),
        )
        for i in range(env.num_envs)
    ]


meta = {
    "task": args_cli.task,
    "motion": args_cli.motion_original,
    "motion_dir": args_cli.motion_dir,
    "file_glob": args_cli.motion_file_glob,
    "checkpoint": args_cli.checkpoint,
    "epsilon_values": EPSILON_VALUES,
    "push_velocities": PUSH_VELOCITIES,
    "n_trials": args_cli.num_trials,
    "n_envs": args_cli.num_envs,
    "settle_steps": args_cli.settle_steps,
    "observe_steps": args_cli.observe_steps,
    "ou_tau": args_cli.ou_tau,
}
store = ResultsStore(meta)

log(
    f"[SMOKE] Tiny validation: eps={EPSILON_VALUES}, pushes={PUSH_VELOCITIES}, "
    f"trials={args_cli.num_trials}, settle={args_cli.settle_steps}, observe={args_cli.observe_steps}"
)
num_batches = (args_cli.num_trials + env.num_envs - 1) // env.num_envs
for ei, epsilon in enumerate(EPSILON_VALUES):
    for pi, delta_v in enumerate(PUSH_VELOCITIES):
        all_results = []
        log(f"[SMOKE] Condition epsilon={epsilon:.3f}, push={delta_v:.2f}")
        for _ in range(num_batches):
            all_results.extend(_run_tiny_condition(epsilon, delta_v))
            if len(all_results) >= args_cli.num_trials:
                break
        for ti, result in enumerate(all_results[: args_cli.num_trials]):
            store.add(ei, pi, ti, result)
        valid = [r for r in all_results[: args_cli.num_trials] if not r.fallen_before_push]
        rate = sum(1 for r in valid if r.success) / max(len(valid), 1) * 100.0
        log(f"   recovery={rate:.1f}% ({len(valid)} valid/{args_cli.num_trials} trials)")

output_dir = os.path.join(
    args_cli.output_dir,
    f"minimal_{Path(args_cli.motion_original).stem}",
)
store.save(output_dir)
load_and_plot(output_dir)

log(f"[SMOKE] Tiny validation pipeline OK. Results: {output_dir}")
env.close()
simulation_app.close()
faulthandler.cancel_dump_traceback_later()
log("[SMOKE] Done.")
