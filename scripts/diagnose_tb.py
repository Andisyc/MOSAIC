"""
FrontRES TensorBoard Diagnostic Reader
=======================================
Reads TensorBoard event files and prints diagnostics without opening a browser.

Usage (run inside Isaac Lab conda env where tensorboard is installed):
    python scripts/diagnose_tb.py                          # latest run, table mode
    python scripts/diagnose_tb.py --log_dir logs/2026-...
    python scripts/diagnose_tb.py --last 20                # show last 20 iters
    python scripts/diagnose_tb.py --block                  # per-iter block mode
    python scripts/diagnose_tb.py --step 3125              # nearest to iter 3125

Default output: compact table — one row per iter, columns = key metrics.
Paste the whole table to share trends without describing curves manually.
"""

import argparse
import sys
from pathlib import Path


# ── Tag map: (tb_tag, short_label, format, unit, section) ────────────────────
_TAGS = [
    ("Train/mean_episode_length",         "ep_len",    "{:6.1f}",  "",    "Epis"),
    ("GMT/mean_episode_length",           "ep_gmt",    "{:6.1f}",  "",    "Epis"),
    ("Curriculum/training_survival_rate", "surv",      "{:.3f}",   "",    "Epis"),
    ("Policy/mean_noise_std",             "noise_std", "{:.4f}",   "",    "Epis"),
    ("Train/mean_r_delta",                "r_δ/ep",    "{:+.4f}",  "",    "Reward"),
    ("Curriculum/r_delta_ema",            "r_δ_ema",   "{:+.4f}",  "",    "Reward"),
    ("FrontRES/r_delta_per_step",         "r_δ/s",     "{:+.5f}",  "",    "Reward"),
    ("FrontRES/baseline_per_step",        "r_step/s",  "{:+.5f}",  "",    "Reward"),
    ("FrontRES/supervised_cos_sim",       "cos_sim",   "{:+.4f}",  "",    "Sup"),
    ("Loss/supervised_loss",              "sup_loss",  "{:.4f}",   "",    "Sup"),
    ("Curriculum/lambda_supervised",      "λ_sup",     "{:.3f}",   "",    "Sup"),
    ("FrontRES/delta_pos_abs_mean",       "|Δpos|m",   "{:.4f}",   "",    "Sup"),
    ("FrontRES/delta_rpy_abs_mean",       "|Δrpy|r",   "{:.4f}",   "",    "Sup"),
    ("Loss/surrogate_loss",               "surr",      "{:.4f}",   "",    "Loss"),
    ("Loss/value_function_loss",          "value",     "{:.4f}",   "",    "Loss"),
    ("Loss/learning_rate",                "lr",        "{:.1e}",   "",    "Loss"),
    ("Curriculum/dr_scale",               "dr_scale",  "{:.3f}",   "",    "Curr"),
]

_WARN = {
    "surv":      lambda v: v < 0.90,
    "noise_std": lambda v: abs(v - 0.01) > 0.005,
    "r_δ/s":     lambda v: v < -0.10,
    "cos_sim":   lambda v: v < -0.05,
    "surr":      lambda v: abs(v) > 1.0,
}


def _find_latest_log(root: str) -> str:
    root_path = Path(root)
    candidates = sorted(
        (p for p in root_path.glob("**/events.out.tfevents.*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.exit(f"[diagnose_tb] No tfevents files found under: {root}")
    return str(candidates[0].parent)


def _load_scalars(log_dir: str) -> dict:
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except ImportError:
        sys.exit(
            "[diagnose_tb] tensorboard not installed.\n"
            "  conda activate isaaclab && python scripts/diagnose_tb.py ..."
        )
    acc = ea.EventAccumulator(log_dir, size_guidance={ea.SCALARS: 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    data = {}
    for tag, label, *_ in _TAGS:
        if tag in available:
            data[label] = {e.step: e.value for e in acc.Scalars(tag)}
    return data


def _all_steps(data: dict) -> list[int]:
    steps: set[int] = set()
    for s in data.values():
        steps.update(s.keys())
    return sorted(steps)


def _trend(series: list[float]) -> str:
    """↑ ↓ ~ based on linear slope of the series."""
    if len(series) < 2:
        return " "
    n = len(series)
    xs = list(range(n))
    xm = sum(xs) / n
    ym = sum(series) / n
    denom = sum((x - xm) ** 2 for x in xs) or 1e-12
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, series)) / denom
    threshold = abs(ym) * 0.02 + 1e-6   # 2% of mean or absolute floor
    if slope > threshold:
        return "↑"
    if slope < -threshold:
        return "↓"
    return "~"


# ── Table mode ────────────────────────────────────────────────────────────────

def _print_table(data: dict, steps: list[int]):
    # Determine which labels have any data in these steps
    labels_present = [label for _, label, *_ in _TAGS if label in data]
    # Add derived r_rescue/s
    has_rescue = "r_δ/s" in data and "r_step/s" in data
    if has_rescue:
        rescue_label = "r_rsc/s"
        labels_present.insert(labels_present.index("r_step/s") + 1, rescue_label)

    # Build format lookup
    fmt_map = {label: fmt for _, label, fmt, *_ in _TAGS}
    fmt_map[rescue_label if has_rescue else "__"] = "{:+.5f}"

    # Column widths: max(header, data width)
    col_w = {}
    for label in labels_present:
        sample = fmt_map.get(label, "{:.4f}").format(0.0)
        col_w[label] = max(len(label), len(sample))

    iter_w = 6

    # Header
    header = f"{'iter':>{iter_w}} | " + " | ".join(
        f"{lbl:^{col_w[lbl]}}" for lbl in labels_present
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    # Rows
    history: dict[str, list[float]] = {lbl: [] for lbl in labels_present}
    for step in steps:
        row_vals: dict[str, float | None] = {}
        for label in labels_present:
            if label == rescue_label if has_rescue else False:
                continue
            row_vals[label] = data.get(label, {}).get(step)

        if has_rescue:
            rd = row_vals.get("r_δ/s")
            rs = row_vals.get("r_step/s")
            row_vals[rescue_label] = (rd - rs) if (rd is not None and rs is not None) else None

        cells = []
        for label in labels_present:
            v = row_vals.get(label)
            if v is None:
                cells.append(f"{'—':^{col_w[label]}}")
            else:
                history[label].append(v)
                fmt = fmt_map.get(label, "{:.4f}")
                txt = fmt.format(v)
                warn_fn = _WARN.get(label)
                flag = "!" if (warn_fn and warn_fn(v)) else " "
                if label == "ep_len":
                    ep_gmt_v = row_vals.get("ep_gmt")
                    if ep_gmt_v is not None and v < ep_gmt_v - 30:
                        flag = "!"
                cells.append(f"{txt+flag:>{col_w[label]}}")

        print(f"{step:>{iter_w}} | " + " | ".join(cells))

    # Trend row
    print(sep)
    trend_cells = []
    for label in labels_present:
        h = history[label]
        t = _trend(h[-8:]) if h else " "
        trend_cells.append(f"{t:^{col_w[label]}}")
    print(f"{'trend':>{iter_w}} | " + " | ".join(trend_cells))
    print(sep)
    print("! = warning threshold exceeded")


# ── Block mode ────────────────────────────────────────────────────────────────

def _print_blocks(data: dict, steps: list[int]):
    for step in steps:
        lines = [f"=== iter {step:>5d} ==="]
        from itertools import groupby
        section_order = ["Epis", "Reward", "Sup", "Loss", "Curr"]
        rows_by_section: dict[str, list[str]] = {s: [] for s in section_order}

        for _, label, fmt, unit, section in _TAGS:
            v = data.get(label, {}).get(step)
            if v is None:
                continue
            warn_fn = _WARN.get(label)
            flag = " !" if (warn_fn and warn_fn(v)) else ""
            rows_by_section[section].append(f"{label}={fmt.format(v)}{unit}{flag}")

        # Derived rescue
        rd = data.get("r_δ/s", {}).get(step)
        rs = data.get("r_step/s", {}).get(step)
        if rd is not None and rs is not None:
            rescue = rd - rs
            flag = " !" if rescue < -0.05 else ""
            rows_by_section["Reward"].append(f"r_rescue/s={rescue:+.5f}{flag}")

        for section in section_order:
            parts = rows_by_section[section]
            if parts:
                lines.append(f"  [{section:<6}] " + "  ".join(parts))

        print("\n".join(lines))
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FrontRES TensorBoard diagnostic reader")
    parser.add_argument("--log_dir",    type=str, default=None)
    parser.add_argument("--logs_root",  type=str, default="logs")
    parser.add_argument("--last",       type=int, default=20,
                        help="Show last N logged iterations (default: 20)")
    parser.add_argument("--step",       type=int, default=None,
                        help="Show a specific iteration (finds nearest)")
    parser.add_argument("--block",      action="store_true",
                        help="Per-iter block format instead of table")
    args = parser.parse_args()

    log_dir = args.log_dir or _find_latest_log(args.logs_root)
    log_dir = str(Path(log_dir).expanduser())
    print(f"[diagnose_tb] {log_dir}\n")

    data = _load_scalars(log_dir)
    if not data:
        sys.exit("[diagnose_tb] No scalar data found.")

    steps = _all_steps(data)

    if args.step is not None:
        nearest = min(steps, key=lambda s: abs(s - args.step))
        selected = [nearest]
    else:
        selected = steps[-args.last:]

    print(f"Total iters: {len(steps)}  |  Showing last {len(selected)}\n")

    if args.block:
        _print_blocks(data, selected)
    else:
        _print_table(data, selected)


if __name__ == "__main__":
    main()
