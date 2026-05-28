"""
RobotBridge / MuJoCo robustness validation smoke runner.

This script runs the same robustness-budget data layout as the IsaacLab
validation, but uses RobotBridge's MosaicEnv and MuJoCo backend.  It is meant
as the first migration step: single motion, small sweeps, compatible output.

Example:
    python scripts/robustness_validation/run_validation_mujoco.py \
        --robotbridge_root /home/chengyuxuan/RobotBridge \
        --motion "/home/chengyuxuan/RobotBridge/deploy/data/motion/Walking/amass_xxx.npz" \
        --checkpoint /home/chengyuxuan/RobotBridge/deploy/data/model/model_27000.onnx \
        --output_dir verify/robustness_validation_mujoco/smoke \
        --record_video
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import gc
import json
import logging
import math
import os

# Headless MuJoCo rendering must be configured before RobotBridge creates mujoco.Renderer.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from pathlib import Path
import random
import sys
from types import MethodType

import numpy as np

from results_io import ResultsStore, TrialResult


EPSILON_VALUES = [0.0, 0.02]
PUSH_VELOCITIES = [0.0, 1.0]
PERTURBATION_MODES = ["composite"]
PERTURBATION_MODE_CHOICES = ["composite", "mixed", "xy", "yaw", "z", "rp"]
N_TRIALS = 2
SETTLE_STEPS = 100
OBSERVE_STEPS = 200
PUSH_OFFSET_MIN = 0
PUSH_OFFSET_MAX = 40
RECOVERY_WINDOW_STEPS = 50
PUSH_SAFETY_MARGIN_STEPS = 20
OU_TAU = 0.5
IID_RATIO = 0.25


def _safe_token(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value).strip("_") or "x"


def _quat_from_euler_xyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3)]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., :3] *= -1.0
    denom = np.sum(q * q, axis=-1, keepdims=True)
    return out / np.maximum(denom, 1e-12)


def _quat_to_rotvec_xyzw(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), eps)
    if q[3] < 0.0:
        q = -q
    xyz = q[:3]
    xyz_norm = np.linalg.norm(xyz)
    angle = 2.0 * math.atan2(xyz_norm, q[3])
    if xyz_norm <= eps:
        return (2.0 * xyz).astype(np.float64)
    return (xyz * (angle / xyz_norm)).astype(np.float64)


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _apply_task_delta_pos(pos: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return (np.asarray(pos, dtype=np.float64) + np.asarray(delta, dtype=np.float64).reshape(6)[:3]).astype(np.float64)


def _apply_task_delta_quat(quat_xyzw: np.ndarray, delta: np.ndarray) -> np.ndarray:
    q_delta = _quat_from_euler_xyz(np.asarray(delta, dtype=np.float64).reshape(6)[3:6])
    # Match MOSAIC: FrontRES predicts a local-frame correction that is
    # right-multiplied onto the perturbed anchor quaternion.
    return _quat_mul_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(4), q_delta).astype(np.float64)


def _apply_task_delta_body_pos(body_pos: np.ndarray, anchor_pos: np.ndarray, delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    rot = _quat_to_matrix_xyzw(_quat_from_euler_xyz(delta[3:6]))
    body = np.asarray(body_pos, dtype=np.float64).reshape(-1, 3)
    anchor = np.asarray(anchor_pos, dtype=np.float64).reshape(1, 3)
    return (anchor + delta[:3].reshape(1, 3) + (body - anchor) @ rot.T).astype(np.float64)


def _apply_task_delta_body_quat(body_quat_xyzw: np.ndarray, delta: np.ndarray) -> np.ndarray:
    q_delta = _quat_from_euler_xyz(np.asarray(delta, dtype=np.float64).reshape(6)[3:6])
    body_q = np.asarray(body_quat_xyzw, dtype=np.float64).reshape(-1, 4)
    q_delta_batch = np.broadcast_to(q_delta.reshape(1, 4), body_q.shape)
    return _quat_mul_xyzw(body_q, q_delta_batch).astype(np.float64)


def _write_status(output_dir: Path, status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2))


class ReferenceFramePerturber:
    """OU + IID SE(3) perturbation applied to RobotBridge reference frames."""

    FAMILY_AXES = {
        "xy": (0, 1),
        "z": (2,),
        "rp": (3, 4),
        "yaw": (5,),
    }

    def __init__(
        self,
        epsilon: float,
        mode: str,
        dt: float,
        tau: float,
        iid_ratio: float,
        seed: int,
    ) -> None:
        self.epsilon = float(epsilon)
        self.mode = mode
        self.dt = float(dt)
        self.tau = max(float(tau), 1e-6)
        self.iid_ratio = float(iid_ratio)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(6, dtype=np.float64)
        self.last_timestep: int | None = None
        self.current = np.zeros(6, dtype=np.float64)
        self.channels = self._sample_channels(mode)

        self.alpha = math.exp(-self.dt / self.tau)
        self.ou_sigma = self.epsilon * math.sqrt(max(1.0 - self.alpha * self.alpha, 0.0))

    def _sample_channels(self, mode: str) -> tuple[str, ...]:
        if mode == "composite":
            return ("xy", "z", "rp", "yaw")
        if mode == "mixed":
            families = np.array(["xy", "z", "rp", "yaw"], dtype=object)
            # Match the FrontRES curriculum spirit: mostly single/two-family
            # perturbations, not a full four-family composite stress test.
            count = int(self.rng.choice([1, 2], p=[0.5, 0.5]))
            picked = self.rng.choice(families, size=count, replace=False)
            return tuple(str(v) for v in picked)
        if mode in self.FAMILY_AXES:
            return (mode,)
        raise ValueError(f"Unknown perturbation mode: {mode}")

    def reset(self) -> None:
        self.state[:] = 0.0
        self.current[:] = 0.0
        self.last_timestep = None

    def _mask(self) -> np.ndarray:
        mask = np.zeros(6, dtype=np.float64)
        for channel in self.channels:
            mask[list(self.FAMILY_AXES[channel])] = 1.0
        return mask

    def value(self, timestep: int) -> np.ndarray:
        if self.epsilon <= 0.0:
            return np.zeros(6, dtype=np.float64)
        if self.last_timestep != int(timestep):
            mask = self._mask()
            self.state = self.alpha * self.state + self.ou_sigma * self.rng.normal(size=6)
            iid = self.iid_ratio * self.epsilon * self.rng.normal(size=6)
            self.current = (self.state + iid) * mask
            self.last_timestep = int(timestep)
        return self.current.copy()

    def apply_pos(self, pos: np.ndarray, clean_anchor_pos: np.ndarray | None = None) -> np.ndarray:
        delta = self.value(0 if self.last_timestep is None else self.last_timestep)
        return (np.asarray(pos, dtype=np.float64) + delta[:3]).astype(np.float64)

    def apply_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        delta = self.value(0 if self.last_timestep is None else self.last_timestep)
        q_delta = _quat_from_euler_xyz(delta[3:6])
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        return _quat_mul_xyzw(q_delta, q).astype(np.float64)

    def transform_body_pos(self, body_pos: np.ndarray, clean_anchor_pos: np.ndarray) -> np.ndarray:
        delta = self.value(0 if self.last_timestep is None else self.last_timestep)
        rot = _quat_to_matrix_xyzw(_quat_from_euler_xyz(delta[3:6]))
        body = np.asarray(body_pos, dtype=np.float64)
        anchor = np.asarray(clean_anchor_pos, dtype=np.float64).reshape(1, 3)
        return (anchor + delta[:3].reshape(1, 3) + (body - anchor) @ rot.T).astype(np.float64)

    def transform_body_quat(self, body_quat_xyzw: np.ndarray) -> np.ndarray:
        delta = self.value(0 if self.last_timestep is None else self.last_timestep)
        q_delta = _quat_from_euler_xyz(delta[3:6])
        body_q = np.asarray(body_quat_xyzw, dtype=np.float64).reshape(-1, 4)
        q_delta_batch = np.broadcast_to(q_delta.reshape(1, 4), body_q.shape)
        return _quat_mul_xyzw(q_delta_batch, body_q).astype(np.float64)


def install_robotbridge_perturbation_patch(robotbridge_deploy: Path) -> None:
    """Monkey-patch MotionDataset properties for this validation process only."""

    sys.path.insert(0, str(robotbridge_deploy))
    from utils.dataset import MotionDataset

    if getattr(MotionDataset, "_mosaic_validation_patch", False):
        return

    orig_anchor_pos = MotionDataset.anchor_pos_w.fget
    orig_anchor_quat = MotionDataset.anchor_quat_w.fget
    orig_body_pos = MotionDataset.body_pos_w_aligned.fget
    orig_body_quat = MotionDataset.body_quat_w_aligned.fget

    def _pert(self):
        return getattr(self, "_validation_perturber", None)

    def _frontres_delta(self):
        return np.asarray(getattr(self, "_frontres_delta", np.zeros(6)), dtype=np.float64).reshape(6)

    def anchor_pos_w(self):
        clean = orig_anchor_pos(self)
        pert = _pert(self)
        out = np.asarray(clean, dtype=np.float64)
        if pert is not None:
            pert.value(int(self.timestep))
            out = pert.apply_pos(out)
        return _apply_task_delta_pos(out, _frontres_delta(self)).astype(np.float64)

    def anchor_quat_w(self):
        clean = orig_anchor_quat(self)
        pert = _pert(self)
        out = np.asarray(clean, dtype=np.float64)
        if pert is not None:
            pert.value(int(self.timestep))
            out = pert.apply_quat(out)
        return _apply_task_delta_quat(out, _frontres_delta(self)).astype(np.float64)

    def body_pos_w_aligned(self):
        # Match MOSAIC: FrontRES/DR changes the anchor command, not the full
        # clean motion library body poses. RobotBridge later computes relative
        # body targets from clean body poses and the perturbed/corrected anchor.
        return np.asarray(orig_body_pos(self), dtype=np.float64)

    def body_quat_w_aligned(self):
        return np.asarray(orig_body_quat(self), dtype=np.float64)

    MotionDataset.anchor_pos_w = property(anchor_pos_w)
    MotionDataset.anchor_quat_w = property(anchor_quat_w)
    MotionDataset.body_pos_w_aligned = property(body_pos_w_aligned)
    MotionDataset.body_quat_w_aligned = property(body_quat_w_aligned)
    MotionDataset._mosaic_validation_patch = True


def _compose_robotbridge_cfg(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str((Path(args.robotbridge_root).expanduser().resolve() / "deploy" / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=["sim=mujoco", "robot=g1_29dof"])

    OmegaConf.set_struct(cfg, False)
    cfg.mimic.motion.motion_path = str(Path(args.motion).expanduser().resolve())
    cfg.mimic.motion.loop = False
    cfg.mimic.motion.playback_speed = 1.0
    cfg.mimic.policy.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    cfg.mimic.policy.eval_mode = True
    cfg.mimic.policy.max_timestep = -1
    cfg.robot.control.viewer = False
    cfg.robot.control.real_time = False
    cfg.env.config.record_video.enabled = bool(args.record_video)
    video_dir = Path(args.video_output_dir)
    if bool(getattr(args, "use_variant_subdir", False)):
        video_dir = video_dir / args.policy_variant
    if args.video_tag:
        video_dir = video_dir / _safe_token(args.video_tag)
    cfg.env.config.record_video.output_dir = str(video_dir.resolve())
    cfg.env.config.record_video.fps = int(args.video_fps)
    cfg.env.config.record_video.width = int(args.video_width)
    cfg.env.config.record_video.height = int(args.video_height)
    cfg.env.config.record_video.include_incomplete = True
    cfg.env.config.record_video.exit_on_complete = False
    return cfg


def _instantiate_robotbridge_agent(args: argparse.Namespace):
    from hydra.utils import instantiate

    robotbridge_root = Path(args.robotbridge_root).expanduser().resolve()
    deploy_dir = robotbridge_root / "deploy"
    install_robotbridge_perturbation_patch(deploy_dir)
    os.chdir(deploy_dir)
    cfg = _compose_robotbridge_cfg(args)
    return instantiate(cfg.agent)


def _close_robotbridge_agent(agent) -> None:
    """Release MuJoCo video/rendering resources before Python teardown."""
    if agent is None:
        return
    env = getattr(agent, "env", None)
    recorder = getattr(env, "video_recorder", None)
    if recorder is not None and hasattr(recorder, "close"):
        try:
            recorder.close()
        except Exception as exc:
            logging.warning("Ignoring RobotBridge video recorder close error: %r", exc)
    simulator = getattr(env, "simulator", None)
    close_sim = getattr(simulator, "close", None)
    if callable(close_sim):
        try:
            close_sim()
        except Exception as exc:
            logging.warning("Ignoring RobotBridge simulator close error: %r", exc)
    close_env = getattr(env, "close", None)
    if callable(close_env):
        try:
            close_env()
        except Exception as exc:
            logging.warning("Ignoring RobotBridge env close error: %r", exc)
    gc.collect()


def _patch_no_auto_reset(env) -> None:
    def _check_termination_no_reset(self):
        self.validation_terminated = bool(self.simulator.check_termination())

    env.validation_terminated = False
    env._check_termination = MethodType(_check_termination_no_reset, env)


def _apply_root_velocity_push(env, push_dir: np.ndarray, push_velocity: float) -> None:
    if push_velocity <= 0.0:
        return
    data = env.simulator.mujoco_data
    data.qvel[0] += float(push_velocity) * float(push_dir[0])
    data.qvel[1] += float(push_velocity) * float(push_dir[1])


def _upright_margin(env) -> float:
    rpy = np.asarray(env.simulator.root_rpy, dtype=np.float64).reshape(-1)
    roll_pitch = max(abs(float(rpy[0])), abs(float(rpy[1]))) if rpy.size >= 2 else 0.0
    height = float(np.asarray(env.simulator.root_trans, dtype=np.float64).reshape(-1)[2])
    return min(1.2 - roll_pitch, height - 0.35)


def _set_frontres_delta(env, delta: np.ndarray | None) -> None:
    env.motion_loader._frontres_delta = (
        np.zeros(6, dtype=np.float64)
        if delta is None else np.asarray(delta, dtype=np.float64).reshape(6)
    )


def _set_frontres_anchor_error_from_perturber(env, perturber: ReferenceFramePerturber | None) -> None:
    if perturber is None:
        return
    loader = env.motion_loader
    timestep = int(getattr(loader, "timestep", 0))
    delta = perturber.value(timestep)

    # Match IsaacLab training: FrontRES observes the correction that undoes the
    # injected reference-frame artifact, not the robot tracking residual.
    anchor_idx = int(loader.motion_anchor_body_index)
    clean_raw = loader.motion.body_quat_w[timestep, anchor_idx].copy()[[1, 2, 3, 0]]
    clean_quat = loader.motion_init_align.align_quat(clean_raw)
    pert_quat = perturber.apply_quat(clean_quat)
    corr_quat = _quat_mul_xyzw(_quat_inv_xyzw(pert_quat), clean_quat)

    target = np.zeros(6, dtype=np.float64)
    target[:3] = -delta[:3]
    target[3:6] = _quat_to_rotvec_xyzw(corr_quat)
    env.frontres_anchor_error = target.astype(np.float32)


def _refresh_obs_with_frontres(agent, obs_buf_dict, frontres_runtime, perturber=None):
    if frontres_runtime is None:
        return obs_buf_dict
    _set_frontres_anchor_error_from_perturber(agent.env, perturber)
    delta = frontres_runtime.compute(agent.env, obs_buf_dict)
    if getattr(agent.env, "_frontres_debug_delta", False):
        step = int(getattr(agent.env.motion_loader, "timestep", 0))
        if step % 30 == 0:
            target = getattr(agent.env, "frontres_anchor_error", np.zeros(6, dtype=np.float32))
            bias = getattr(frontres_runtime, "last_bias_delta", np.zeros(6, dtype=np.float32))
            print(
                "[FrontRESDebug] "
                f"t={step} target={np.asarray(target).round(4).tolist()} "
                f"delta={np.asarray(delta).round(4).tolist()} "
                f"bias={np.asarray(bias).round(4).tolist()}",
                flush=True,
            )
    _set_frontres_delta(agent.env, delta)
    agent.env.compute_observation()
    return agent.env.obs_buf_dict


def _run_policy_step(agent, obs_buf_dict, frontres_runtime=None, perturber=None):
    obs_buf_dict = _refresh_obs_with_frontres(agent, obs_buf_dict, frontres_runtime, perturber)
    inputs = {key: obs_buf_dict[key].astype(np.float32) for key in obs_buf_dict}
    action = agent.policy.run(None, inputs)[0]
    return agent.env.step(action)


def run_trial(agent, perturber: ReferenceFramePerturber, push_velocity: float, seed: int, args, frontres_runtime=None) -> TrialResult:
    env = agent.env
    env.motion_loader._validation_perturber = perturber
    _set_frontres_delta(env, None)
    perturber.reset()
    if frontres_runtime is not None:
        frontres_runtime.reset()
    env.validation_terminated = False

    if env.video_recorder.enabled:
        env.video_recorder.start_sequence(str(args.motion))

    obs = env.reset()
    env.motion_loader.cur_motion_end = False
    rng = random.Random(seed)
    if args.push_direction_angle is None:
        angle = rng.uniform(0.0, 2.0 * math.pi)
    else:
        angle = math.radians(float(args.push_direction_angle))
    push_dir = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)
    max_offset_by_window = args.observe_steps - args.recovery_window_steps - args.push_safety_margin_steps
    push_offset_max = min(args.push_offset_max, max(args.push_offset_min, max_offset_by_window))
    if args.fixed_push_offset is None:
        push_offset = rng.randint(args.push_offset_min, push_offset_max)
    else:
        push_offset = int(np.clip(args.fixed_push_offset, 0, max_offset_by_window))
    push_step_abs = args.settle_steps + push_offset

    settle_margins: list[float] = []
    post_margins: list[float] = []
    fallen_before_push = False
    success = True
    last_margin: float | None = None
    pre_push_margin: float | None = None
    post_window_end = push_step_abs + args.recovery_window_steps

    total_steps = args.settle_steps + args.observe_steps
    for step in range(total_steps):
        if step == push_step_abs:
            pre_push_margin = last_margin if last_margin is not None else _upright_margin(env)
            _apply_root_velocity_push(env, push_dir, push_velocity)

        obs = _run_policy_step(agent, obs, frontres_runtime, perturber)
        margin = _upright_margin(env)
        if step < push_step_abs:
            settle_margins.append(margin)
        elif step < post_window_end:
            post_margins.append(margin)
        last_margin = margin

        if getattr(env, "validation_terminated", False):
            success = False
            fallen_before_push = step < push_step_abs
            break
        if env.motion_loader.cur_motion_end:
            break

    if env.video_recorder.enabled:
        env.video_recorder.save(env.motion_loader, complete=success, reason="validation_trial")
    _set_frontres_delta(env, None)

    return TrialResult(
        success=bool(success and not fallen_before_push),
        fallen_before_push=bool(fallen_before_push),
        T_push_step=int(push_offset),
        zmp_margins_settle=settle_margins,
        zmp_margins_post=post_margins,
        push_dir=push_dir.tolist(),
        push_step_abs=int(push_step_abs),
        push_phase=float(push_step_abs / max(total_steps, 1)),
        pre_push_margin=(
            float(pre_push_margin)
            if pre_push_margin is not None else (float(settle_margins[-1]) if settle_margins else None)
        ),
        min_zmp_after_push=float(np.min(post_margins)) if post_margins else None,
        mean_zmp_after_push=float(np.mean(post_margins)) if post_margins else None,
        margin_drop=(
            float((pre_push_margin if pre_push_margin is not None else settle_margins[-1]) - np.min(post_margins))
            if post_margins and (pre_push_margin is not None or settle_margins) else None
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RobotBridge/MuJoCo robustness validation.")
    parser.add_argument("--robotbridge_root", type=str, default="../RobotBridge")
    parser.add_argument("--config_name", type=str, default="mosaic")
    parser.add_argument("--motion", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="RobotBridge policy checkpoint, usually model_27000.onnx.")
    parser.add_argument("--frontres_checkpoint", type=str, default=None,
                        help="Optional MOSAIC FrontRES checkpoint. GMT ONNX is still supplied by --checkpoint.")
    parser.add_argument("--frontres_device", type=str, default="cpu")
    parser.add_argument("--frontres_history_length", type=int, default=5)
    parser.add_argument("--frontres_max_delta_pos", type=float, default=0.3)
    parser.add_argument("--frontres_max_delta_rpy", type=float, default=0.4)
    parser.add_argument("--frontres_active_task_dims", type=int, nargs="+", default=[2, 3, 4, 6, 7],
                        help="Task-space output dims enabled for FEMR. Default matches rp_z specialist.")
    parser.add_argument("--frontres_allow_upward_dz", action="store_true")
    parser.add_argument("--frontres_ignore_conf", action="store_true")
    parser.add_argument("--frontres_subtract_zero_error_bias", action="store_true",
                        help="Enable deployment-only zero-error FEMR output subtraction.")
    parser.add_argument("--frontres_debug_delta", action="store_true",
                        help="Print FEMR target/delta/bias values every 30 motion steps.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Shared run folder name. Use the same value for baseline and FrontRES.")
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Write directly under --output_dir instead of creating a timestamped run folder.")
    parser.add_argument("--video_tag", type=str, default=None,
                        help="Optional label added to video directory and file prefix, e.g. demo_frontres.")
    parser.add_argument("--motion_group", type=str, default="Ungrouped")
    parser.add_argument("--motion_name", type=str, default=None)
    parser.add_argument("--epsilon_values", type=float, nargs="+", default=EPSILON_VALUES)
    parser.add_argument("--push_velocities", type=float, nargs="+", default=PUSH_VELOCITIES)
    parser.add_argument("--perturbation_modes", type=str, nargs="+", default=PERTURBATION_MODES,
                        choices=PERTURBATION_MODE_CHOICES)
    parser.add_argument("--num_trials", type=int, default=N_TRIALS)
    parser.add_argument("--settle_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--observe_steps", type=int, default=OBSERVE_STEPS)
    parser.add_argument("--push_offset_min", type=int, default=PUSH_OFFSET_MIN)
    parser.add_argument("--push_offset_max", type=int, default=PUSH_OFFSET_MAX)
    parser.add_argument("--fixed_push_offset", type=int, default=None,
                        help="Fixed push step within the observe phase. If omitted, timing is randomized.")
    parser.add_argument("--push_direction_angle", type=float, default=None,
                        help="Fixed horizontal push direction in degrees. If omitted, direction is randomized per trial.")
    parser.add_argument("--recovery_window_steps", type=int, default=RECOVERY_WINDOW_STEPS,
                        help="Only this many steps after the push contribute to post-push min-margin metrics.")
    parser.add_argument("--push_safety_margin_steps", type=int, default=PUSH_SAFETY_MARGIN_STEPS,
                        help="Keep pushes at least this many steps away from the observation horizon.")
    parser.add_argument("--ou_tau", type=float, default=OU_TAU)
    parser.add_argument("--iid_ratio", type=float, default=IID_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    args = parser.parse_args()
    args.policy_variant = "frontres" if args.frontres_checkpoint else "baseline"

    output_root = Path(args.output_dir).expanduser().resolve()
    is_frontres_run = bool(args.frontres_checkpoint)
    if args.no_timestamp:
        run_root = output_root
        run_name = output_root.name
    else:
        run_name = _safe_token(args.run_name) if args.run_name else (
            f"run_{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_root = output_root / run_name
    if is_frontres_run:
        # A FrontRES smoke/demo run already contains its paired baseline sequence
        # internally, so keep it as one standalone run instead of nesting a
        # redundant frontres/baseline variant folder.
        output_dir = run_root
        args.video_output_dir = str(run_root / "videos")
        args.use_variant_subdir = False
    else:
        # Plain GMT validation remains a baseline artifact and should stay under
        # the caller's robustness_validation_mujoco output root.
        output_dir = run_root / args.policy_variant
        args.video_output_dir = str(run_root / "videos")
        args.use_variant_subdir = True
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.video_output_dir).mkdir(parents=True, exist_ok=True)
    if args.motion_name is None:
        args.motion_name = Path(args.motion).stem

    meta = {
        "backend": "RobotBridge/MuJoCo",
        "run_name": run_name,
        "run_root": str(run_root),
        "variant_output_dir": str(output_dir),
        "video_output_dir": args.video_output_dir,
        "motion": str(Path(args.motion).expanduser()),
        "motion_group": args.motion_group,
        "motion_name": args.motion_name,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "policy_variant": args.policy_variant,
        "video_tag": args.video_tag,
        "frontres_checkpoint": str(Path(args.frontres_checkpoint).expanduser()) if args.frontres_checkpoint else None,
        "frontres_device": args.frontres_device,
        "frontres_history_length": args.frontres_history_length,
        "frontres_max_delta_pos": args.frontres_max_delta_pos,
        "frontres_max_delta_rpy": args.frontres_max_delta_rpy,
        "frontres_active_task_dims": args.frontres_active_task_dims,
        "frontres_allow_upward_dz": args.frontres_allow_upward_dz,
        "frontres_ignore_conf": args.frontres_ignore_conf,
        "frontres_subtract_zero_error_bias": args.frontres_subtract_zero_error_bias,
        "frontres_debug_delta": args.frontres_debug_delta,
        "epsilon_values": args.epsilon_values,
        "push_velocities": args.push_velocities,
        "fixed_push_offset": args.fixed_push_offset,
        "push_direction_angle": args.push_direction_angle,
        "perturbation_modes": args.perturbation_modes,
        "n_trials": args.num_trials,
        "settle_steps": args.settle_steps,
        "observe_steps": args.observe_steps,
        "ou_tau": args.ou_tau,
        "iid_ratio": args.iid_ratio,
        "recovery_window_steps": args.recovery_window_steps,
        "push_safety_margin_steps": args.push_safety_margin_steps,
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
    }
    store = ResultsStore(meta)
    _write_status(output_dir, "running", **meta)

    agent = None
    try:
        agent = _instantiate_robotbridge_agent(args)
        _patch_no_auto_reset(agent.env)
        frontres_runtime = None
        if args.frontres_checkpoint:
            from frontres_runtime import FrontRESRuntime

            frontres_runtime = FrontRESRuntime(
                checkpoint=args.frontres_checkpoint,
                device=args.frontres_device,
                history_length=args.frontres_history_length,
                max_delta_pos=args.frontres_max_delta_pos,
                max_delta_rpy=args.frontres_max_delta_rpy,
                allow_upward_dz=args.frontres_allow_upward_dz,
                ignore_conf=args.frontres_ignore_conf,
                active_task_dims=args.frontres_active_task_dims,
                subtract_zero_error_bias=args.frontres_subtract_zero_error_bias,
            )
            print(
                f"[MuJoCoValidation] FrontRES enabled: {frontres_runtime.checkpoint} "
                f"(device={args.frontres_device}, history={args.frontres_history_length})",
                flush=True,
            )
        agent.env._frontres_debug_delta = bool(args.frontres_debug_delta)

        dt = float(getattr(agent.env.simulator, "high_dt", 0.02))
        for mode_idx, mode in enumerate(args.perturbation_modes):
            for eps_idx, eps in enumerate(args.epsilon_values):
                for push_idx, push_velocity in enumerate(args.push_velocities):
                    for trial_idx in range(args.num_trials):
                        seed = args.seed + 100000 * mode_idx + 1000 * eps_idx + 100 * push_idx + trial_idx
                        perturber = ReferenceFramePerturber(
                            epsilon=eps,
                            mode=mode,
                            dt=dt,
                            tau=args.ou_tau,
                            iid_ratio=args.iid_ratio,
                            seed=seed,
                        )
                        token = (
                            f"{mode}_eps{eps:g}_push{push_velocity:g}_trial{trial_idx:02d}"
                        )
                        if mode == "mixed":
                            token = (
                                f"{mode}-{'-'.join(perturber.channels)}"
                                f"_eps{eps:g}_push{push_velocity:g}_trial{trial_idx:02d}"
                            )
                        if agent.env.video_recorder.enabled:
                            prefix_parts = ["mujoco", args.policy_variant]
                            if args.video_tag:
                                prefix_parts.append(_safe_token(args.video_tag))
                            prefix_parts.append(_safe_token(token))
                            agent.env.video_recorder.prefix = "_".join(prefix_parts)
                        result = run_trial(agent, perturber, push_velocity, seed, args, frontres_runtime)
                        store.add(mode_idx, eps_idx, push_idx, trial_idx, result)
                        print(
                            "[MuJoCoValidation] "
                            f"mode={mode} channels={','.join(perturber.channels)} "
                            f"eps={eps:g} push={push_velocity:g} "
                            f"trial={trial_idx + 1}/{args.num_trials} "
                            f"success={result.success} pre_fall={result.fallen_before_push}",
                            flush=True,
                        )

        store.save(str(output_dir))
        _write_status(output_dir, "completed", **meta)
        return 0
    except Exception as exc:
        _write_status(output_dir, "failed", error=repr(exc), **meta)
        raise
    finally:
        _close_robotbridge_agent(agent)


if __name__ == "__main__":
    raise SystemExit(main())
