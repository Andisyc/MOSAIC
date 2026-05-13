"""
Robustness Budget Validation Experiment
========================================
One-stop script: configure → launch Isaac Sim → run experiment → save + plot.

Usage (from MOSAIC root):
    python scripts/robustness_validation/run_validation.py \
        --motion   /path/to/clean_amass_npz_dir   \
        --checkpoint /path/to/model_27000.pt \
        --headless

Configuration: edit the EXPERIMENT CONFIG section below.
Results are saved to OUTPUT_DIR; figures are auto-generated at the end.
"""

# ════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIG  — edit here before running
# ════════════════════════════════════════════════════════════════════════════

# OU noise: steady-state RMS values to sweep (metres, root-frame)
EPSILON_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# Lateral velocity perturbation magnitude (m/s) applied as instantaneous push
PUSH_VELOCITIES = [0.5, 1.0, 2.0, 3.0]

# Statistical power: total trials per condition
N_TRIALS  = 100
# GPU parallelism: number of simultaneous envs (keep low to avoid OOM)
N_PARALLEL = 1

# Phase durations (physics steps; 1 step = 0.02 s at 50 Hz)
SETTLE_STEPS = 100   # 2.0 s  — OU builds up; GMT reaches quasi-steady state
OBSERVE_STEPS = 200  # 4.0 s  — window to observe push recovery

# Random push timing: uniformly drawn from [PUSH_OFFSET_MIN, PUSH_OFFSET_MAX] steps
# into the observe phase (avoids always pushing at the same gait phase)
PUSH_OFFSET_MIN = 0
PUSH_OFFSET_MAX = 40

# OU temporal smoothing: camera jitter ≈ 0.5–2 s
OU_TAU = 0.5   # seconds

# Task: GMT-only, no FrontRES-specific observations, multi-motion command.
# The validation needs MotionPerturber, which is provided by MultiMotionCommand.
TASK = "General-Tracking-Flat-G1-Wo-State-Estimation-v0"

# Output directory (timestamped sub-folder created automatically)
OUTPUT_DIR = "verify/robustness_validation"

# ════════════════════════════════════════════════════════════════════════════
#  ISAAC SIM BOOTSTRAP  — must come before any other isaaclab imports
# ════════════════════════════════════════════════════════════════════════════

import argparse
import faulthandler
import os
import sys
from pathlib import Path


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


def log(message: str) -> None:
    print(message, flush=True)


_sanitize_python_path_for_isaac()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Robustness budget validation.")
parser.add_argument("--motion",     type=str, required=True,
                    help="Path to a clean AMASS .npz directory, or a single .npz file.")
parser.add_argument("--checkpoint", "--resume_path", dest="checkpoint", type=str, required=True,
                    help="Path to GMT checkpoint (.pt).")
parser.add_argument("--task",       type=str, default=TASK,
                    help="Gym task. Default uses the multi-motion GMT-only task.")
parser.add_argument("--file_glob",  type=str, default="*.npz",
                    help="Glob used when --motion is a directory.")
parser.add_argument("--num_envs",   type=int, default=N_PARALLEL,
                    help="Number of parallel envs.")
parser.add_argument("--num_trials", type=int, default=N_TRIALS,
                    help="Total trials per condition.")
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--startup_timeout", type=int, default=30,
                    help="Dump Python stack traces every N seconds while diagnosing hangs. Use 0 to disable.")
# --device / --headless are added by AppLauncher.add_app_launcher_args below
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _resolve_motion_input(path: str, file_glob: str) -> tuple[str, str, str]:
    """Return (motion_dir, file_glob, original_path) for the validation env."""

    original = str(Path(path).expanduser())
    motion_path = Path(original)
    if motion_path.is_file():
        # MultiMotionCommand requires a directory.  Restrict the loader to this
        # exact file so single-motion smoke validation still works.
        return str(motion_path.parent), motion_path.name, original
    if motion_path.is_dir():
        candidates = sorted(p for p in motion_path.rglob(file_glob) if p.is_file())
        if not candidates:
            parser.error(f"--motion directory contains no files matching {file_glob!r}: {path}")
        log(f"[Validation] Found {len(candidates)} clean motion files under {motion_path}")
        return str(motion_path), file_glob, original
    parser.error(f"--motion must be a .npz file or a directory containing .npz files: {path}")


args_cli.motion_dir, args_cli.motion_file_glob, args_cli.motion_original = _resolve_motion_input(
    args_cli.motion, args_cli.file_glob
)

if args_cli.startup_timeout > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(args_cli.startup_timeout, repeat=True)

if (
    sys.platform.startswith("linux")
    and not args_cli.headless
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    log("[Validation] No display detected; forcing --headless for Isaac Sim startup.")
    args_cli.headless = True

if hydra_args:
    log(f"[Validation] Ignoring non-AppLauncher arguments: {hydra_args}")
sys.argv = [sys.argv[0]]

log(f"[Validation] Launching Isaac Sim: device={args_cli.device}, headless={args_cli.headless}")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
log("[Validation] simulation_app is ready.")

# ════════════════════════════════════════════════════════════════════════════
#  MAIN IMPORTS  (after sim is running)
# ════════════════════════════════════════════════════════════════════════════

import datetime
import os
import sys
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # noqa: F401 — registers gym tasks

# local modules (same directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ou_injector     import configure_ou, reset_ou_states
from push_controller import PushController
from metrics         import (compute_zmp_margin, is_fallen, find_body_index)
from results_io      import ResultsStore, TrialResult
from plot_results    import load_and_plot


@configclass
class _NoOpCfg:
    pass


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _make_output_dir(base: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _build_env(
    motion_dir: str,
    motion_original: str,
    file_glob: str,
    num_envs: int,
    task: str,
    device: str,
):
    """
    Create IsaacLab environment:
      - GMT-only task (no FrontRES obs)
      - Disable timeout, events, obs noise, motion perturbations (we control them)
      - All envs track the same motion from the beginning
    """
    from whole_body_tracking.tasks.tracking.mdp.motion_perturbations import MotionPerturbationCfg

    # Load registered env cfg
    task_entry  = gym.envs.registry[task]
    env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_cls, str):
        module, cls = env_cfg_cls.rsplit(":", 1)
        import importlib
        env_cfg_cls = getattr(importlib.import_module(module), cls)

    # Monkey-patch __post_init__ so it never sets events/curriculum to None.
    # Isaac Lab's ManagerBase._resolve_terms_callback iterates over
    # self.cfg.__dict__ and crashes on NoneType.
    _orig_post_init = getattr(env_cfg_cls, '__post_init__', None)
    def _safe_post_init(self_):
        if _orig_post_init is not None:
            _orig_post_init(self_)
        if getattr(self_, 'events', None) is None:
            self_.events = _NoOpCfg()
        if getattr(self_, 'curriculum', None) is None:
            self_.curriculum = _NoOpCfg()
    env_cfg_cls.__post_init__ = _safe_post_init

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_cls()

    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs

    # Point to clean AMASS motions.  MultiMotionCommand expects a directory and
    # file_glob, while the legacy single-motion command expects one file.
    motion_cfg = env_cfg.commands.motion
    if hasattr(motion_cfg, "file_glob"):
        motion_cfg.motion = motion_dir
        motion_cfg.file_glob = file_glob
        if hasattr(motion_cfg, "motion_preload_device"):
            motion_cfg.motion_preload_device = device
        if hasattr(motion_cfg, "motion_dataset_shard_across_gpus"):
            motion_cfg.motion_dataset_shard_across_gpus = False
        if hasattr(motion_cfg, "motion_dataset_load_cap"):
            motion_cfg.motion_dataset_load_cap = None
    else:
        if not Path(motion_original).is_file():
            raise ValueError(
                f"Task {task!r} uses a single-motion command, so --motion must be one .npz file."
            )
        motion_cfg.motion = motion_original
        if hasattr(motion_cfg, "motion_file"):
            motion_cfg.motion_file = motion_original

    # Force all envs to start from the same frame
    if hasattr(motion_cfg, "start_from_beginning"):
        motion_cfg.start_from_beginning = True
    if hasattr(motion_cfg, "start_frame"):
        motion_cfg.start_frame = 0

    # Prevent motion resampling within our experiment horizon
    if hasattr(motion_cfg, "resampling_time_range"):
        motion_cfg.resampling_time_range = (1e9, 1e9)
    if hasattr(motion_cfg, "resample_motions_every_s"):
        motion_cfg.resample_motions_every_s = 1e9
    if hasattr(motion_cfg, "debug_vis"):
        motion_cfg.debug_vis = False

    # Zero initial pose/velocity randomisation for reproducibility
    _zero_init_randomisation(env_cfg)

    # Disable env timeout (we control episode length)
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Disable event manager and other managers (training domain randomisation)
    # Use an empty config rather than None: Isaac Lab's ManagerBase iterates over
    # self.cfg.__dict__ and crashes on NoneType.  Scan ALL top-level config fields
    # because gym.make may take the registered config class, not our modified obj.
    _manager_fields = ["events", "curriculum"]
    for _f in _manager_fields:
        if hasattr(env_cfg, _f):
            setattr(env_cfg, _f, _NoOpCfg())

    # Disable observation noise
    for group_name in ("policy", "teacher", "critic"):
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, group_name):
            grp = getattr(env_cfg.observations, group_name)
            if hasattr(grp, "enable_corruption"):
                grp.enable_corruption = False

    # Disable motion perturbations — we configure them ourselves via ou_injector
    if hasattr(env_cfg, "motion_perturbations"):
        env_cfg.motion_perturbations = MotionPerturbationCfg()  # all probs=0

    # Headless validation should not register debug-vis callbacks.  We still use
    # contact forces for ZMP; only the visualization is disabled.
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    # Debug: verify config before gym.make
    log(f"[_build_env] task={task}")
    log(f"[_build_env] motion={motion_cfg.motion}")
    if hasattr(motion_cfg, "file_glob"):
        log(f"[_build_env] file_glob={motion_cfg.file_glob}")
    log(f"[_build_env] device={getattr(env_cfg.sim, 'device', 'N/A')}")
    log(f"[_build_env] events={type(env_cfg.events).__name__ if hasattr(env_cfg, 'events') else 'N/A'}")
    _cur = getattr(env_cfg, 'curriculum', None)
    log(f"[_build_env] curriculum={type(_cur).__name__}")

    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    return env


def _zero_init_randomisation(env_cfg) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        return
    zero = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.),
            "roll": (0., 0.), "pitch": (0., 0.), "yaw": (0., 0.)}
    for attr in ("pose_range", "velocity_range"):
        if hasattr(motion_cfg, attr):
            setattr(motion_cfg, attr, zero)
    if hasattr(motion_cfg, "joint_position_range"):
        motion_cfg.joint_position_range = (0.0, 0.0)


def _load_gmt(env, checkpoint_path: str, device: str):
    """
    Load GMT checkpoint via OnPolicyRunner (same as play.py).
    Returns the runner (which holds policy + obs_normalizer).
    """
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
        G1FlatPPORunnerCfg,
    )
    agent_cfg = G1FlatPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner._move_normalizer_to_device(device)
    runner.load(checkpoint_path, load_optimizer=False, load_critic=False)
    return runner


def _get_robot(env_unwrapped):
    return env_unwrapped.scene["robot"]


# ════════════════════════════════════════════════════════════════════════════
#  SINGLE CONDITION: run N_TRIALS envs for one (epsilon, push_velocity) pair
# ════════════════════════════════════════════════════════════════════════════

def run_condition(
    env,
    runner: OnPolicyRunner,
    push_ctrl: PushController,
    left_foot_idx: int,
    right_foot_idx: int,
    epsilon: float,
    delta_v: float,
    device: str,
) -> list[TrialResult]:
    """
    Returns a list of N_TRIALS TrialResult objects.

    Phase 1 — Settle (SETTLE_STEPS):
      All envs track the motion with OU noise injected.
      ZMP margin recorded every step.

    Phase 2 — Observe (OBSERVE_STEPS):
      Push delivered at a random step per env.
      ZMP margin + fallen status recorded every step.
    """
    env_unwrapped = env.unwrapped
    robot = _get_robot(env_unwrapped)
    num_envs = env.num_envs

    # Configure OU for this epsilon
    configure_ou(env_unwrapped, epsilon, tau=OU_TAU)

    # Reset env: all envs start at motion frame 0 with OU state = 0
    obs, _ = env.get_observations()
    env_ids_all = torch.arange(num_envs, device=device)
    reset_ou_states(env_unwrapped, env_ids_all)

    # Track per-env state
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    fallen_during_settle = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Storage: ZMP history
    settle_zmp  = [[] for _ in range(num_envs)]
    post_zmp    = [[] for _ in range(num_envs)]
    success     = torch.zeros(num_envs, dtype=torch.bool, device=device)

    policy = runner.get_inference_policy(device=device)

    # ── Phase 1: Settle ─────────────────────────────────────────────────────
    for step in range(SETTLE_STEPS):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        # Detect falls (env auto-resets, but we track who fell)
        just_fallen = is_fallen(env_unwrapped) | dones.bool()
        newly_fallen = just_fallen & alive
        fallen_during_settle |= newly_fallen
        alive &= ~newly_fallen

        zmp = compute_zmp_margin(env_unwrapped, left_foot_idx, right_foot_idx)
        for i in range(num_envs):
            if alive[i] or newly_fallen[i]:  # record last known ZMP too
                settle_zmp[i].append(float(zmp[i]))

    # ── Phase 2: Observe ────────────────────────────────────────────────────
    # Randomize push timing for envs still alive
    alive_ids = torch.where(alive)[0]
    push_ctrl.randomize(alive_ids)

    for step in range(OBSERVE_STEPS):
        # Apply push for envs whose push_at_step == step
        push_ctrl.maybe_push(robot, observe_step=step, delta_v=delta_v, alive=alive)

        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        just_fallen = is_fallen(env_unwrapped) | dones.bool()
        newly_fallen = just_fallen & alive
        alive &= ~newly_fallen

        zmp = compute_zmp_margin(env_unwrapped, left_foot_idx, right_foot_idx)
        for i in range(num_envs):
            if not fallen_during_settle[i] and (alive[i] or newly_fallen[i]):
                post_zmp[i].append(float(zmp[i]))

    # Success = alive at end of observe phase AND did not fall during settle
    success = alive & ~fallen_during_settle

    # Build TrialResult objects
    results = []
    for i in range(num_envs):
        results.append(TrialResult(
            success=bool(success[i]),
            fallen_before_push=bool(fallen_during_settle[i]),
            T_push_step=int(push_ctrl.push_at_step[i]),
            zmp_margins_settle=settle_zmp[i],
            zmp_margins_post=post_zmp[i],
            push_dir=push_ctrl.push_dir[i].cpu().tolist(),
        ))

    return results


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = args_cli.device if torch.cuda.is_available() else "cpu"
    num_envs = args_cli.num_envs
    output_dir = _make_output_dir(args_cli.output_dir)

    log(f"\n{'='*60}")
    log(f"  Robustness Budget Validation")
    log(f"  Task:       {args_cli.task}")
    log(f"  Motion:     {args_cli.motion_original}")
    log(f"  Motion dir: {args_cli.motion_dir}")
    log(f"  File glob:  {args_cli.motion_file_glob}")
    log(f"  Checkpoint: {args_cli.checkpoint}")
    log(f"  Epsilons:   {EPSILON_VALUES}")
    log(f"  Push Δv:    {PUSH_VELOCITIES} m/s")
    log(f"  Trials:     {args_cli.num_trials} per condition")
    log(f"  Parallel:   {num_envs} envs")
    log(f"  Device:     {device}")
    log(f"  Output:     {output_dir}")
    log(f"{'='*60}\n")

    # Build env + GMT
    num_trials = args_cli.num_trials
    num_envs   = args_cli.num_envs
    num_batches = (num_trials + num_envs - 1) // num_envs

    env = _build_env(
        motion_dir=args_cli.motion_dir,
        motion_original=args_cli.motion_original,
        file_glob=args_cli.motion_file_glob,
        num_envs=num_envs,
        task=args_cli.task,
        device=device,
    )
    runner = _load_gmt(env, args_cli.checkpoint, device)

    env_unwrapped = env.unwrapped
    left_foot_idx  = find_body_index(env_unwrapped, "left_ankle_roll_link")
    right_foot_idx = find_body_index(env_unwrapped, "right_ankle_roll_link")
    log(f"[Validation] left_ankle_idx={left_foot_idx}, right_ankle_idx={right_foot_idx}")

    push_ctrl = PushController(
        num_envs=num_envs,
        device=device,
        push_offset_range=(PUSH_OFFSET_MIN, PUSH_OFFSET_MAX),
    )

    # Results store
    meta = {
        "task":           args_cli.task,
        "motion":         args_cli.motion_original,
        "motion_dir":     args_cli.motion_dir,
        "file_glob":      args_cli.motion_file_glob,
        "checkpoint":     args_cli.checkpoint,
        "epsilon_values": EPSILON_VALUES,
        "push_velocities": PUSH_VELOCITIES,
        "n_trials":       num_trials,
        "n_envs":         num_envs,
        "settle_steps":   SETTLE_STEPS,
        "observe_steps":  OBSERVE_STEPS,
        "ou_tau":         OU_TAU,
    }
    store = ResultsStore(meta)

    # ── Outer loop: conditions ───────────────────────────────────────────────
    total_conditions = len(EPSILON_VALUES) * len(PUSH_VELOCITIES)
    done_conditions  = 0

    for ei, epsilon in enumerate(EPSILON_VALUES):
        for pi, delta_v in enumerate(PUSH_VELOCITIES):
            done_conditions += 1
            log(f"\n[{done_conditions}/{total_conditions}] "
                f"ε={epsilon:.3f} m  |  Δv={delta_v:.1f} m/s")

            # Run batches to accumulate num_trials per condition
            all_results = []
            for batch in range(num_batches):
                env.reset()
                trial_results = run_condition(
                    env, runner, push_ctrl,
                    left_foot_idx, right_foot_idx,
                    epsilon=epsilon, delta_v=delta_v,
                    device=device,
                )
                all_results.extend(trial_results)
                # Stop early if we have enough
                if len(all_results) >= num_trials:
                    break

            # Store first num_trials results
            for ti, r in enumerate(all_results[:num_trials]):
                store.add(ei, pi, ti, r)

            # Quick progress print
            valid     = [r for r in all_results if not r.fallen_before_push]
            rec_rate  = sum(1 for r in valid if r.success) / max(len(valid), 1) * 100
            n_pre_fall = sum(1 for r in all_results if r.fallen_before_push)
            log(f"   recovery rate = {rec_rate:.1f}%  "
                f"({len(valid)} valid, {n_pre_fall} pre-fallen)")

    # ── Save + Plot ──────────────────────────────────────────────────────────
    store.save(output_dir)
    load_and_plot(output_dir)

    log(f"\n[Validation] Complete. Results in: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        simulation_app.close()
        # Isaac Sim background GPU threads don't exit on simulation_app.close().
        # os._exit() bypasses Python cleanup and terminates the process immediately,
        # allowing batch scripts to launch the next motion without getting stuck.
        import os as _os
        _os._exit(0)
