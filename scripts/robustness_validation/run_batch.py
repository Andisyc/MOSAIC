"""
批量运行验证实验：逐条启动 run_validation.py，等待退出后再运行下一条。

用法：
    python scripts/robustness_validation/run_batch.py \
        --motion_list results/selected_motions.txt   \
        --checkpoint  /path/to/model_27000.pt        \
        [--headless] [--timeout 1800] [--skip_done]

motion_list 格式（filter_motions.py 输出的文件，# 开头行忽略）：
    # 行走 (Walking)
    /hdd0/.../walk_subject1.npz
    /hdd0/.../walk_subject2.npz
    # 转身 (Turning)
    /hdd0/.../turn_01.npz
    ...
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_motion_list(path: str) -> list[str]:
    """读取 motion_list 文件，返回有效路径列表（跳过注释和空行）。"""
    motions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            motions.append(line)
    return motions


def result_exists(motion_path: str, output_dir: str) -> bool:
    """检查该 motion 是否已有结果（skip_done 模式用）。"""
    stem = Path(motion_path).stem
    out = Path(output_dir)
    if not out.exists():
        return False
    # 结果目录名格式：run_<timestamp>，内含 results.json 且 meta 记录了 motion_file
    import json
    for run_dir in sorted(out.glob("run_*")):
        meta_file = run_dir / "results.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text())
                saved_stem = Path(data.get("meta", {}).get("motion_file", "")).stem
                if saved_stem == stem:
                    return True
            except Exception:
                pass
    return False


def run_one(motion: str, checkpoint: str, extra_args: list[str],
            timeout: int) -> int:
    """
    启动一次 run_validation.py，等待完成，返回退出码。
    超时则强制 SIGKILL。
    """
    script = Path(__file__).parent / "run_validation.py"
    cmd = [
        sys.executable, str(script),
        "--motion",     motion,
        "--checkpoint", checkpoint,
    ] + extra_args

    print(f"\n{'─'*60}")
    print(f"  启动: {Path(motion).name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'─'*60}")

    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"\n[Batch] 超时 ({timeout}s)，强制终止...")
        try:
            # 先 SIGTERM，给 Isaac Sim 1s 机会
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(1)
        except Exception:
            pass
        try:
            # 再 SIGKILL 确保死透
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        return -1


def main():
    parser = argparse.ArgumentParser(description="批量验证实验启动器")
    parser.add_argument("--motion_list", type=str, required=True,
                        help="motion 路径列表文件（filter_motions.py 输出）")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="GMT checkpoint 路径")
    parser.add_argument("--output_dir",  type=str,
                        default="results/robustness_validation",
                        help="结果输出目录（与 run_validation.py 一致）")
    parser.add_argument("--timeout",     type=int, default=1800,
                        help="每条序列最长等待时间（秒），默认 1800 = 30 min")
    parser.add_argument("--skip_done",   action="store_true",
                        help="跳过已有结果的序列（断点续跑）")
    parser.add_argument("--headless",    action="store_true",
                        help="传给 run_validation.py 的 --headless 参数")
    parser.add_argument("--num_envs",    type=int, default=None,
                        help="传给 run_validation.py 的 --num_envs 参数")
    args = parser.parse_args()

    motions = parse_motion_list(args.motion_list)
    if not motions:
        sys.exit(f"[Batch] motion_list 为空：{args.motion_list}")

    print(f"[Batch] 共 {len(motions)} 条序列待处理")
    print(f"[Batch] checkpoint: {args.checkpoint}")
    print(f"[Batch] 每条超时: {args.timeout}s")

    # 构建传给 run_validation.py 的附加参数
    extra: list[str] = ["--output_dir", args.output_dir]
    if args.headless:
        extra += ["--headless"]
    if args.num_envs is not None:
        extra += ["--num_envs", str(args.num_envs)]

    success_count = 0
    fail_count    = 0
    skip_count    = 0

    for idx, motion in enumerate(motions, 1):
        print(f"\n[Batch] ── 序列 {idx}/{len(motions)}: {Path(motion).name}")

        if not Path(motion).exists():
            print(f"[Batch] ⚠ 文件不存在，跳过: {motion}")
            fail_count += 1
            continue

        if args.skip_done and result_exists(motion, args.output_dir):
            print(f"[Batch] ✓ 已有结果，跳过")
            skip_count += 1
            continue

        ret = run_one(motion, args.checkpoint, extra, args.timeout)

        if ret == 0:
            print(f"[Batch] ✓ 完成 (exit 0)")
            success_count += 1
        elif ret == -1:
            print(f"[Batch] ✗ 超时被杀")
            fail_count += 1
        else:
            print(f"[Batch] ✗ 非零退出 (exit {ret})")
            fail_count += 1

        # 两次启动之间等待，确保 GPU 内存完全释放
        if idx < len(motions):
            print("[Batch] 等待 10s 释放 GPU 内存...")
            time.sleep(10)

    print(f"\n{'='*60}")
    print(f"[Batch] 全部完成")
    print(f"  成功: {success_count} | 失败/超时: {fail_count} | 跳过: {skip_count}")
    print(f"  结果目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
