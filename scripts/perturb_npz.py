"""
perturb_npz.py — Apply q_ref domain randomisation to a reference-motion .npz file.

Mirrors MotionPerturber used in Stage-2 FrontRES training.
Source: tracking_env_cfg.py G1FlatFrontRESFinetuneEnvCfg.__post_init__() lines 1122-1129.

These perturbations are applied to q_ref (the reference motion fed to GMT/FrontRES),
NOT to the simulated robot itself (robot-level DR is handled by Isaac Lab EventManager).

Training DR parameters (applied independently per frame):
    float_prob=0.3,  float_ratio=0.05   root floats up    ≤ 5 cm  (30 % chance/frame)
    sink_prob=0.3,   sink_ratio=0.05    root sinks  down  ≤ 5 cm  (30 % chance/frame)
    slip_prob=0.2,   slip_ratio=0.03    root slips  horiz ≤ 3 cm  (20 % chance/frame)
    drag_prob=0.2,   drag_ratio=0.02    root dragged opp. velocity (20 % chance/frame)

All perturbations translate the ENTIRE skeleton (all 30 bodies shift by the same delta),
keeping the skeleton kinematically intact while making the reference unreachable for GMT.

Demo mode (--float_height N):
    Lifts the entire skeleton by a fixed N metres every frame.
    E.g. --float_height 0.5  →  person dancing 50 cm above ground.
    GMT cannot levitate the robot → immediate fall.  FrontRES corrects q_ref → stable.

Foot-slip detection:
    G1 defaults: left_ankle_roll_link = body index 6, right_ankle_roll_link = index 12.
    Override with --left_foot_idx / --right_foot_idx if your .npz uses different ordering.
    Use --auto_feet to auto-detect the two lowest-z bodies instead.

Usage
-----
# Demo: constant 0.5 m float — body in mid-air, GMT will immediately fall
python perturb_npz.py --motion_file dance.npz --output dance_float.npz --float_height 0.5

# Training-identical stochastic perturbation (reproducible with fixed seed)
python perturb_npz.py --motion_file dance.npz --output dance_dr.npz --seed 42

# Stack constant float + stochastic for a dramatic combined effect
python perturb_npz.py --motion_file dance.npz --output dance_combo.npz \\
    --float_height 0.3 --float_prob 0.5 --float_ratio 0.1 --seed 0

# Training-identical but disable stochastic (pure constant float)
python perturb_npz.py --motion_file dance.npz --output dance_air.npz \\
    --float_height 0.5 --no_stochastic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Perturbation implementations (pure numpy, mirrors MotionPerturber logic)
# ---------------------------------------------------------------------------

def _apply_float(body_pos: np.ndarray, float_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Shift entire skeleton upward by float_ratio * U[0,1] metres."""
    delta_z = float_ratio * rng.random()
    body_pos[:, 2] += delta_z
    return body_pos


def _apply_sink(body_pos: np.ndarray, sink_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Shift entire skeleton downward by sink_ratio * U[0,1] metres."""
    delta_z = sink_ratio * rng.random()
    body_pos[:, 2] -= delta_z
    return body_pos


def _apply_drag(body_pos: np.ndarray, root_vel: np.ndarray, drag_ratio: float) -> np.ndarray:
    """Shift entire skeleton opposite to root velocity (simulates air/ground resistance)."""
    drag = -root_vel * drag_ratio          # (3,)
    body_pos += drag[None, :]              # broadcast over all bodies
    return body_pos


def _apply_slip(
    body_pos: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    slip_ratio: float,
    slip_height: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shift entire skeleton horizontally when exactly one foot is in swing phase.
    Mirrors training: slip direction depends on which foot is airborne.
    """
    left_up  = left_foot_pos[2]  > slip_height
    right_up = right_foot_pos[2] > slip_height
    one_up   = left_up ^ right_up           # XOR: exactly one foot airborne

    if not one_up:
        return body_pos

    direction = 1.0 if left_up else -1.0   # left up → slip +X; right up → -X
    magnitude = slip_ratio * rng.random()
    body_pos[:, 0] += direction * magnitude
    return body_pos


# ---------------------------------------------------------------------------
# Core per-frame perturbation
# ---------------------------------------------------------------------------

def perturb_sequence(
    body_pos_w: np.ndarray,           # (T, B, 3)
    body_lin_vel_w: np.ndarray,       # (T, B, 3)
    left_foot_idx: int,
    right_foot_idx: int,
    float_prob: float,   float_ratio: float,
    sink_prob: float,    sink_ratio: float,
    slip_prob: float,    slip_ratio: float,  slip_height: float,
    drag_prob: float,    drag_ratio: float,
    rng: np.random.Generator,
    const_float_height: float = 0.0,
) -> np.ndarray:
    """
    Apply perturbations frame-by-frame.
    const_float_height is added on top of stochastic perturbations every frame.
    Returns a perturbed copy of body_pos_w — input is not mutated.
    """
    out = body_pos_w.copy()
    T = out.shape[0]

    for t in range(T):
        frame = out[t]                     # (B, 3) — view into out

        # Constant demo float applied every frame
        if const_float_height != 0.0:
            frame[:, 2] += const_float_height

        # Stochastic float
        if float_prob > 0.0 and rng.random() < float_prob:
            frame = _apply_float(frame, float_ratio, rng)

        # Stochastic sink
        if sink_prob > 0.0 and rng.random() < sink_prob:
            frame = _apply_sink(frame, sink_ratio, rng)

        # Body drag — uses root velocity from ORIGINAL data (mirrors training behaviour)
        if drag_prob > 0.0 and rng.random() < drag_prob:
            root_vel = body_lin_vel_w[t, 0]
            frame = _apply_drag(frame, root_vel, drag_ratio)

        # Foot slip
        if slip_prob > 0.0 and rng.random() < slip_prob:
            frame = _apply_slip(
                frame,
                left_foot_pos=out[t, left_foot_idx],
                right_foot_pos=out[t, right_foot_idx],
                slip_ratio=slip_ratio,
                slip_height=slip_height,
                rng=rng,
            )

        out[t] = frame

    return out


# ---------------------------------------------------------------------------
# Auto foot detection
# ---------------------------------------------------------------------------

def auto_detect_feet(body_pos_w: np.ndarray, root_idx: int = 0) -> tuple[int, int]:
    """
    Find the two non-root bodies with the lowest mean z-coordinate (heuristic for feet).
    Returns (left_idx, right_idx) — left has larger mean X in typical URDF convention.
    """
    mean_z = body_pos_w[:, :, 2].mean(axis=0)
    mean_x = body_pos_w[:, :, 0].mean(axis=0)

    candidates = [(i, mean_z[i]) for i in range(body_pos_w.shape[1]) if i != root_idx]
    candidates.sort(key=lambda x: x[1])
    foot_a_idx, foot_b_idx = candidates[0][0], candidates[1][0]

    if mean_x[foot_a_idx] >= mean_x[foot_b_idx]:
        return foot_a_idx, foot_b_idx
    else:
        return foot_b_idx, foot_a_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply q_ref domain randomisation to a motion .npz file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Paths (consistent with replay_npz.py style) ──────────────────────
    p.add_argument(
        "--motion_file", 
        type=str,
        default="./data/motion/dance1_subject1.npz",
        help="Input .npz path.",
    )
    p.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output .npz path.",
    )

    # ── Demo mode ─────────────────────────────────────────────────────────
    p.add_argument(
        "--float_height", type=float, default=0.0,
        help="Constant vertical lift (metres) applied every frame. "
             "0 = disabled. E.g. 0.5 → skeleton 50 cm above ground (clearly unreachable for GMT).",
    )

    # ── Stochastic DR (training-identical defaults) ───────────────────────
    # Source: tracking_env_cfg.py G1FlatFrontRESFinetuneEnvCfg.__post_init__() lines 1122-1129
    p.add_argument("--float_prob",  type=float, default=0.3,
                   help="Probability per frame that float is applied (training default: 0.3)")
    p.add_argument("--float_ratio", type=float, default=0.05,
                   help="Max float displacement in metres (training default: 0.05)")
    p.add_argument("--sink_prob",   type=float, default=0.3,
                   help="Probability per frame that sink is applied (training default: 0.3)")
    p.add_argument("--sink_ratio",  type=float, default=0.05,
                   help="Max sink displacement in metres (training default: 0.05)")
    p.add_argument("--slip_prob",   type=float, default=0.2,
                   help="Probability per frame that foot-slip is applied (training default: 0.2)")
    p.add_argument("--slip_ratio",  type=float, default=0.03,
                   help="Max slip displacement in metres (training default: 0.03)")
    p.add_argument("--slip_height", type=float, default=0.05,
                   help="Foot z-threshold (m) above which a foot is considered airborne (training default: 0.05)")
    p.add_argument("--drag_prob",   type=float, default=0.2,
                   help="Probability per frame that body-drag is applied (training default: 0.2)")
    p.add_argument("--drag_ratio",  type=float, default=0.02,
                   help="Drag displacement ratio relative to root velocity (training default: 0.02)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--no_stochastic", action="store_true",
                   help="Disable all stochastic perturbations (only --float_height is applied)")

    # ── Foot body indices ─────────────────────────────────────────────────
    p.add_argument("--left_foot_idx",  type=int, default=6,
                   help="Raw .npz body index for left_ankle_roll_link (G1 default: 6)")
    p.add_argument("--right_foot_idx", type=int, default=12,
                   help="Raw .npz body index for right_ankle_roll_link (G1 default: 12)")
    p.add_argument("--auto_feet", action="store_true",
                   help="Auto-detect foot indices from the two lowest-z bodies")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    input_path = Path(args.motion_file)
    if not input_path.exists():
        sys.exit(f"[ERROR] motion_file not found: {input_path}")

    if args.output is None:
        output_path = input_path.parent / (input_path.stem + "_perturbed.npz")
    else:
        output_path = Path(args.output)

    # ── Load ──────────────────────────────────────────────────────────────
    data = np.load(input_path, allow_pickle=True)
    keys = list(data.files)
    print(f"[perturb_npz] Loaded: {input_path}")
    print(f"  Keys: {keys}")

    for key in ("body_pos_w", "body_lin_vel_w", "joint_pos", "joint_vel"):
        if key not in keys:
            sys.exit(f"[ERROR] Required key '{key}' not found in .npz")

    body_pos_w     = data["body_pos_w"].astype(np.float64)      # (T, B, 3)
    body_lin_vel_w = data["body_lin_vel_w"].astype(np.float64)  # (T, B, 3)
    T, B, _ = body_pos_w.shape
    print(f"  Frames={T}  Bodies={B}")

    # ── Foot indices ───────────────────────────────────────────────────────
    if args.auto_feet:
        left_foot_idx, right_foot_idx = auto_detect_feet(body_pos_w)
        print(f"[perturb_npz] Auto-detected feet: left={left_foot_idx}, right={right_foot_idx}")
    else:
        left_foot_idx  = args.left_foot_idx
        right_foot_idx = args.right_foot_idx
        print(f"[perturb_npz] Foot indices: left={left_foot_idx}, right={right_foot_idx}")

    if max(left_foot_idx, right_foot_idx) >= B:
        sys.exit(
            f"[ERROR] Foot index out of range (bodies={B}). "
            "Use --auto_feet or adjust --left_foot_idx / --right_foot_idx."
        )

    # ── Report plan ────────────────────────────────────────────────────────
    if args.float_height != 0.0:
        print(f"[perturb_npz] Demo float: +{args.float_height:.3f} m every frame")
    if not args.no_stochastic:
        print("[perturb_npz] Stochastic q_ref DR (training-identical defaults):")
        print(f"  float: prob={args.float_prob}, ratio={args.float_ratio} m")
        print(f"  sink:  prob={args.sink_prob},  ratio={args.sink_ratio} m")
        print(f"  slip:  prob={args.slip_prob},  ratio={args.slip_ratio} m")
        print(f"  drag:  prob={args.drag_prob},  ratio={args.drag_ratio}")
        print(f"  seed={args.seed}")
    else:
        print("[perturb_npz] Stochastic DR disabled (--no_stochastic)")

    # ── Apply ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    perturbed_pos = perturb_sequence(
        body_pos_w         = body_pos_w,
        body_lin_vel_w     = body_lin_vel_w,
        left_foot_idx      = left_foot_idx,
        right_foot_idx     = right_foot_idx,
        float_prob         = 0.0 if args.no_stochastic else args.float_prob,
        float_ratio        = args.float_ratio,
        sink_prob          = 0.0 if args.no_stochastic else args.sink_prob,
        sink_ratio         = args.sink_ratio,
        slip_prob          = 0.0 if args.no_stochastic else args.slip_prob,
        slip_ratio         = args.slip_ratio,
        slip_height        = args.slip_height,
        drag_prob          = 0.0 if args.no_stochastic else args.drag_prob,
        drag_ratio         = args.drag_ratio,
        rng                = rng,
        const_float_height = args.float_height,
    )

    # ── Stats ──────────────────────────────────────────────────────────────
    delta_z = perturbed_pos[:, 0, 2] - body_pos_w[:, 0, 2]
    print(f"[perturb_npz] Root z-delta — mean={delta_z.mean():.4f} m  "
          f"max={delta_z.max():.4f} m  min={delta_z.min():.4f} m")

    # ── Save ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k: data[k] for k in keys}
    save_dict["body_pos_w"] = perturbed_pos.astype(np.float32)
    # body_lin_vel_w kept unchanged: drag modifies reference position only,
    # not the stored velocity field — mirrors training MotionPerturber behaviour.

    np.savez(output_path, **save_dict)
    print(f"[perturb_npz] Saved → {output_path}")


if __name__ == "__main__":
    main()