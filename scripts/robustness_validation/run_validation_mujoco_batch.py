"""
Batch launcher for RobotBridge / MuJoCo robustness validation.

Expected motion layout:

    motion_root/
      Walking/*.npz
      Turning/*.npz
      Upper/*.npz
      Lateral/*.npz

Each motion is saved independently:

    output_dir/
      run_meta.json
      motions/<group>/<motion_stem>/{meta.json,results_raw.npz,summary.csv,status.json,videos/*.mp4}
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_GROUPS = ["Walking", "Turning", "Upper", "Lateral"]


def _is_completed(path: Path) -> bool:
    status_path = path / "status.json"
    if not status_path.is_file():
        return False
    try:
        return json.loads(status_path.read_text()).get("status") == "completed"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch RobotBridge/MuJoCo robustness validation.")
    parser.add_argument("--robotbridge_root", type=str, required=True)
    parser.add_argument("--motion_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="RobotBridge policy checkpoint, usually model_27000.onnx.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--groups", type=str, nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--file_glob", type=str, default="*.npz")
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--epsilon_values", type=float, nargs="+", default=[0.0, 0.02])
    parser.add_argument("--push_velocities", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--fixed_push_offset", type=int, default=None,
                        help="Fixed push step within the observe phase.")
    parser.add_argument("--push_direction_angle", type=float, default=None,
                        help="Fixed horizontal push direction in degrees.")
    parser.add_argument("--perturbation_modes", type=str, nargs="+", default=["composite"],
                        choices=["composite", "xy", "yaw", "z", "rp"])
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--skip_completed", dest="skip_completed", action="store_true", default=True)
    parser.add_argument("--no-skip_completed", "--no_skip_completed", dest="skip_completed", action="store_false")
    parser.add_argument("--stop_on_failure", dest="stop_on_failure", action="store_true", default=False)
    parser.add_argument("--no-stop_on_failure", "--no_stop_on_failure", dest="stop_on_failure", action="store_false")
    args, passthrough = parser.parse_known_args()

    motion_root = Path(args.motion_root).expanduser().resolve()
    if not motion_root.is_dir():
        raise FileNotFoundError(f"motion_root not found: {motion_root}")

    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("verify/robustness_validation_mujoco") / f"run_{stamp}"
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "backend": "RobotBridge/MuJoCo",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "robotbridge_root": str(Path(args.robotbridge_root).expanduser()),
        "motion_root": str(motion_root),
        "checkpoint": args.checkpoint,
        "groups": args.groups,
        "file_glob": args.file_glob,
        "n_trials": args.num_trials,
        "epsilon_values": args.epsilon_values,
        "push_velocities": args.push_velocities,
        "fixed_push_offset": args.fixed_push_offset,
        "push_direction_angle": args.push_direction_angle,
        "perturbation_modes": args.perturbation_modes,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    validation_script = Path(__file__).with_name("run_validation_mujoco.py")
    jobs: list[tuple[str, Path, Path]] = []
    for group in args.groups:
        group_dir = motion_root / group
        if not group_dir.is_dir():
            print(f"[mujoco-batch] Missing group directory, skipping: {group_dir}", flush=True)
            continue
        for motion_path in sorted(group_dir.glob(args.file_glob)):
            if motion_path.is_file():
                motion_out = output_dir / "motions" / group / motion_path.stem
                jobs.append((group, motion_path, motion_out))

    print(f"[mujoco-batch] Found {len(jobs)} motion jobs. Output: {output_dir}", flush=True)
    failures = 0
    for idx, (group, motion_path, motion_out) in enumerate(jobs, start=1):
        if args.skip_completed and _is_completed(motion_out):
            print(f"[mujoco-batch] [{idx}/{len(jobs)}] skip completed: {group}/{motion_path.name}", flush=True)
            continue

        motion_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(validation_script),
            "--robotbridge_root", args.robotbridge_root,
            "--motion", str(motion_path),
            "--checkpoint", args.checkpoint,
            "--output_dir", str(motion_out),
            "--no_timestamp",
            "--motion_group", group,
            "--motion_name", motion_path.stem,
            "--num_trials", str(args.num_trials),
            "--epsilon_values", *[str(v) for v in args.epsilon_values],
            "--push_velocities", *[str(v) for v in args.push_velocities],
            "--perturbation_modes", *args.perturbation_modes,
            *passthrough,
        ]
        if args.fixed_push_offset is not None:
            cmd += ["--fixed_push_offset", str(args.fixed_push_offset)]
        if args.push_direction_angle is not None:
            cmd += ["--push_direction_angle", str(args.push_direction_angle)]
        if args.record_video:
            cmd.append("--record_video")

        print(f"[mujoco-batch] [{idx}/{len(jobs)}] running: {group}/{motion_path.name}", flush=True)
        completed = subprocess.run(cmd, cwd=Path.cwd())
        if completed.returncode != 0:
            failures += 1
            status = {
                "status": "failed",
                "motion_group": group,
                "motion_name": motion_path.stem,
                "motion": str(motion_path),
                "returncode": completed.returncode,
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            (motion_out / "status.json").write_text(json.dumps(status, indent=2))
            print(
                f"[mujoco-batch] FAILED: {group}/{motion_path.name} "
                f"returncode={completed.returncode}",
                flush=True,
            )
            if args.stop_on_failure:
                return completed.returncode

    print(f"[mujoco-batch] Done. failures={failures}, output={output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
