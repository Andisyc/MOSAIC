"""
从 AMASS 风格的 npz 数据集中筛选出适合验证实验的动作序列。

数据集结构：
  DATA_ROOT/
    <子数据集>/<数字子文件夹>/<动作名称>.npz
    例：LAFAN1/01/walk_subject1.npz

用法：
  python scripts/filter_motions.py --data_root /path/to/data
  python scripts/filter_motions.py --data_root /path/to/data --min_sec 3.0 --save results/motions.txt
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  配置区：按需修改关键词
# ─────────────────────────────────────────────────────────────────────────────

# 每个类别对应的关键词列表（匹配文件名，大小写不敏感，支持正则）
# 关键词以单词边界匹配（避免 "sidewalk" 被匹配成 "walk"）
CATEGORIES = {
    "行走 (Walking)": [
        r"\bwalk\b", r"\bwalking\b", r"\bstroll\b", r"\bjog\b", r"\bjogging\b",
        r"\bmarch\b", r"\bmarching\b", r"\bpace\b",
    ],
    "侧移 (Lateral)": [
        r"\bside\b", r"\bsidestep\b", r"\bside_step\b", r"\blateral\b",
        r"\bstrafe\b", r"\bcrab\b", r"\bsideways\b",
    ],
    "转身 (Turning)": [
        r"\bturn\b", r"\bturning\b", r"\bspin\b", r"\bspinning\b",
        r"\brotate\b", r"\brotation\b", r"\bpivot\b", r"\bcircle\b",
        r"\bdirection\b",
    ],
    "上肢大幅运动 (Upper-body)": [
        r"\bdance\b", r"\bdancing\b", r"\bwave\b", r"\bwaving\b",
        r"\bgesture\b", r"\breach\b", r"\bswing\b", r"\bpunch\b",
        r"\bkick\b", r"\bthrow\b", r"\bcatch\b", r"\bclap\b",
    ],
}

# 每个类别期望筛选出的序列数（达到后停止搜索该类别）
TARGET_PER_CATEGORY = 4

# ─────────────────────────────────────────────────────────────────────────────

def _read_num_frames(path: Path) -> int | None:
    """读取 npz 帧数，失败返回 None。"""
    try:
        data = np.load(path, mmap_mode="r")
        # 优先用 joint_pos，其次找第一个二维数组
        if "joint_pos" in data:
            return int(data["joint_pos"].shape[0])
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 1:
                return int(arr.shape[0])
    except Exception:
        pass
    return None


def _match_category(stem: str, patterns: list[str]) -> bool:
    """文件 stem（无扩展名，保留下划线/连字符）是否匹配任意关键词。"""
    name = stem.lower().replace("-", "_")
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


def _category_of(stem: str) -> str | None:
    """返回文件名匹配的第一个类别名，未匹配返回 None。"""
    for cat, patterns in CATEGORIES.items():
        if _match_category(stem, patterns):
            return cat
    return None


def scan(data_root: str, min_sec: float = 2.0, fps: float = 30.0,
         target: int = TARGET_PER_CATEGORY) -> dict[str, list[Path]]:
    """
    遍历 data_root 下所有 .npz，按类别分组，每类最多返回 target 条。

    Args:
        data_root:  数据集根目录
        min_sec:    最短时长（秒），低于此值的序列被跳过
        fps:        动作序列帧率（用于从帧数换算时长）
        target:     每类最多保留几条

    Returns:
        {类别名: [Path, ...]}
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] 目录不存在: {root}")

    min_frames = int(min_sec * fps)
    results: dict[str, list[Path]] = {cat: [] for cat in CATEGORIES}
    skipped_short = 0
    total_scanned = 0

    # 按路径排序，保证结果可复现
    for path in sorted(root.rglob("*.npz")):
        total_scanned += 1
        stem = path.stem          # 文件名（无扩展名）
        cat = _category_of(stem)
        if cat is None:
            continue
        if len(results[cat]) >= target:
            continue              # 该类已满，继续但不加入

        # 检查时长
        n_frames = _read_num_frames(path)
        if n_frames is not None and n_frames < min_frames:
            skipped_short += 1
            continue

        results[cat].append(path)

    print(f"\n扫描完成：共 {total_scanned} 个 .npz，"
          f"跳过 {skipped_short} 个过短（< {min_sec:.1f}s @ {fps:.0f}fps）\n")
    return results


def print_results(results: dict[str, list[Path]], data_root: str) -> None:
    root = Path(data_root).expanduser().resolve()
    any_empty = False
    for cat, paths in results.items():
        print(f"── {cat}  ({len(paths)} 条) ──")
        if not paths:
            print("  （未找到）")
            any_empty = True
        for p in paths:
            # 打印相对路径，方便阅读
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            n = _read_num_frames(p)
            dur = f"{n/30:.1f}s" if n else "?s"
            print(f"  {rel}  [{dur}]")
        print()

    if any_empty:
        print("[提示] 部分类别未找到序列，可能原因：")
        print("  1. 数据目录路径有误（请用 --data_root 指定正确路径）")
        print("  2. 数据集命名风格与关键词不符（请编辑脚本 CATEGORIES 字典）")
        print("  3. 该类别在你的子集中不存在\n")


def save_results(results: dict[str, list[Path]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for cat, paths in results.items():
            f.write(f"# {cat}\n")
            for p in paths:
                f.write(f"{p}\n")
            f.write("\n")
    print(f"[已保存] {out}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="筛选验证实验用动作序列")
    parser.add_argument("--data_root", type=str, required=True,
                        help="AMASS/项目 npz 根目录")
    parser.add_argument("--min_sec",   type=float, default=3.0,
                        help="最短时长（秒），默认 3.0")
    parser.add_argument("--fps",       type=float, default=30.0,
                        help="动作帧率，默认 30")
    parser.add_argument("--target",    type=int,   default=TARGET_PER_CATEGORY,
                        help=f"每类最多保留几条，默认 {TARGET_PER_CATEGORY}")
    parser.add_argument("--save",      type=str,   default=None,
                        help="将结果路径列表保存到此文件（可选）")
    args = parser.parse_args()

    results = scan(args.data_root, min_sec=args.min_sec,
                   fps=args.fps, target=args.target)
    print_results(results, args.data_root)

    if args.save:
        save_results(results, args.save)

    # 打印可直接复制到 run_validation.py 的命令示例
    all_paths = [str(p) for paths in results.values() for p in paths]
    if all_paths:
        print("── 示例：逐条运行验证实验 ──")
        for p in all_paths:
            print(f"  python scripts/robustness_validation/run_validation.py "
                  f"--motion {p} --checkpoint /path/to/model.pt --headless")


if __name__ == "__main__":
    main()
