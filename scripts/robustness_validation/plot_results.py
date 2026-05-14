"""
Standalone plotting script — supports single or multiple result directories.

Single run:
    python plot_results.py --results_dir results/run_20250507_120000

Multiple motions (aggregated):
    python plot_results.py \
        --results_dirs results/run_walk results/run_dance results/run_squat \
        --motion_names walk dance squat \
        --output_dir  results/combined_figures
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from results_io import ResultsStore


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# Colours for push-velocity curves
PUSH_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
# Colours for individual motion curves (light, used behind the mean)
MOTION_COLORS = ["#90CAF9", "#A5D6A7", "#FFCC80", "#EF9A9A",
                 "#CE93D8", "#80DEEA", "#BCAAA4", "#B0BEC5"]


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINTS
# ═════════════════════════════════════════════════════════════════════════════

def load_and_plot(results_dir: str, output_dir: str | None = None) -> None:
    """Single-run convenience wrapper — backward compatible."""
    load_and_plot_multi([results_dir], motion_names=None, output_dir=output_dir)


def _validate_results_dir(path: str) -> str:
    """Validate a results directory and return its normalized path."""

    p = Path(path).expanduser()
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a results directory, got: {path}")

    missing = [
        name
        for name in ("meta.json", "results_raw.npz")
        if not (p / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Result directory {p} is missing required file(s): {', '.join(missing)}"
        )
    return str(p)


def load_and_plot_multi(
    results_dirs: list[str],
    motion_names: list[str] | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Load one or more result directories and generate all figures.

    When len(results_dirs) == 1  → single-motion mode (no per-motion lines).
    When len(results_dirs)  > 1  → multi-motion mode:
        - Individual motion curves plotted as thin dashed lines.
        - Mean across motions as thick solid line with ±1 std shading.
        - Heatmap shows mean recovery rate.
    """
    result_dirs = [_validate_results_dir(path) for path in results_dirs]

    if output_dir is None:
        # Default: sibling "figures" folder next to the first results dir
        output_dir = os.path.join(os.path.dirname(result_dirs[0]), "figures_combined")
    os.makedirs(output_dir, exist_ok=True)

    # Assign default motion names if not provided
    if motion_names is None:
        motion_names = [os.path.basename(d) for d in result_dirs]

    # Load stores + summaries
    named_summaries: list[tuple[str, dict]] = []
    meta = None
    for name, d in zip(motion_names, result_dirs):
        store = ResultsStore.load(d)
        named_summaries.append((name, store.to_summary()))
        if meta is None:
            meta = store.meta  # use first store's meta for axis labels

    # Merge into aggregated summary
    merged = ResultsStore.merge_summaries(named_summaries)
    multi_mode = len(result_dirs) > 1

    plot_recovery_curve(merged, meta, output_dir, multi_mode=multi_mode)
    plot_zmp_mechanism(merged, meta, output_dir, multi_mode=multi_mode)
    plot_recovery_heatmap(merged, meta, output_dir)
    plot_zmp_timeseries_multi(named_summaries, meta, output_dir)

    print(f"[plot] All figures saved to {output_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Recovery rate vs ε
# ═════════════════════════════════════════════════════════════════════════════

def plot_recovery_curve(
    merged: dict,
    meta: dict,
    output_dir: str,
    multi_mode: bool = False,
) -> None:
    """
    Main figure: recovery rate (%) vs ε for each push velocity.

    Single-motion mode: solid lines with Bernoulli CI shading.
    Multi-motion  mode: thin dashed lines per motion + thick solid mean ± std shading.
    """
    eps_vals  = meta["epsilon_values"]
    pvel_vals = meta["push_velocities"]
    n_eps, n_pvel = len(eps_vals), len(pvel_vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pi, pv in enumerate(pvel_vals):
        color = PUSH_COLORS[pi % len(PUSH_COLORS)]
        mean_rates = np.array([merged[ei][pi]["mean_rate"] * 100 for ei in range(n_eps)])
        ci_rates   = np.array([merged[ei][pi]["ci_rate"]   * 100 for ei in range(n_eps)])

        if multi_mode:
            # Individual motion curves (thin dashed, same hue but lighter)
            n_motions = len(merged[0][pi]["motion_names"])
            for mi in range(n_motions):
                ind_rates = np.array([
                    merged[ei][pi]["rates_per_motion"][mi] * 100
                    for ei in range(n_eps)
                ])
                label = merged[0][pi]["motion_names"][mi] if pi == 0 else None
                ax.plot(eps_vals, ind_rates,
                        color=color, linewidth=0.9, linestyle="--",
                        alpha=0.45, label=label)

            # Std band across motions
            std_rates = np.array([merged[ei][pi]["std_rate"] * 100 for ei in range(n_eps)])
            ax.fill_between(eps_vals,
                            mean_rates - std_rates,
                            mean_rates + std_rates,
                            color=color, alpha=0.18)
            ax.plot(eps_vals, mean_rates,
                    color=color, linewidth=2.2, linestyle="-",
                    marker="o", markersize=6,
                    label=f"Mean — Δv={pv} m/s")
        else:
            # Single motion: Bernoulli CI
            ax.fill_between(eps_vals,
                            mean_rates - ci_rates,
                            mean_rates + ci_rates,
                            color=color, alpha=0.15)
            ax.plot(eps_vals, mean_rates,
                    color=color, linewidth=2, linestyle="-",
                    marker="o", markersize=6,
                    label=f"Δv = {pv} m/s")

    ax.axhline(70.0, color="gray", linestyle="--", linewidth=1.0, label="70% threshold")
    ax.set_xlabel("Reference frame noise RMS ε (m)")
    ax.set_ylabel("Push recovery rate (%)")
    title = "Robustness Budget Consumption vs Reference Frame Noise"
    if multi_mode:
        n = len(merged[0][0]["motion_names"])
        title += f"\n(mean ± std over {n} motion sequences)"
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)

    # Legend: motions in upper part, push velocities in lower part
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.75,
              ncol=2 if multi_mode else 1)

    _save(fig, output_dir, "fig1_recovery_curve")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — ZMP margin vs ε (mechanism)
# ═════════════════════════════════════════════════════════════════════════════

def plot_zmp_mechanism(
    merged: dict,
    meta: dict,
    output_dir: str,
    multi_mode: bool = False,
) -> None:
    """
    Mechanism figure: mean ZMP margin during settle phase vs ε.
    ZMP is identical regardless of push velocity → averaged over push velocities.
    """
    eps_vals  = meta["epsilon_values"]
    pvel_vals = meta["push_velocities"]
    n_eps, n_pvel = len(eps_vals), len(pvel_vals)
    color = "#1976D2"

    fig, ax = plt.subplots(figsize=(6, 4))

    # Average ZMP over push velocities (settle ZMP is push-independent)
    mean_zmps, std_zmps = [], []
    for ei in range(n_eps):
        zm = [merged[ei][pi]["mean_zmp"] for pi in range(n_pvel)
              if not np.isnan(merged[ei][pi]["mean_zmp"])]
        sz = [merged[ei][pi]["std_zmp"]  for pi in range(n_pvel)
              if not np.isnan(merged[ei][pi]["std_zmp"])]
        mean_zmps.append(np.mean(zm) if zm else float("nan"))
        std_zmps.append(np.mean(sz)  if sz else float("nan"))

    mean_zmps = np.array(mean_zmps)
    std_zmps  = np.array(std_zmps)

    if multi_mode:
        # Per-motion lines
        n_motions = len(merged[0][0]["motion_names"])
        for mi in range(n_motions):
            ind_zmps = []
            for ei in range(n_eps):
                zm_pi = [merged[ei][pi]["zmp_per_motion"][mi]
                         for pi in range(n_pvel)
                         if mi < len(merged[ei][pi]["zmp_per_motion"])]
                ind_zmps.append(np.mean(zm_pi) if zm_pi else float("nan"))
            mname = merged[0][0]["motion_names"][mi]
            ax.plot(eps_vals, ind_zmps, color=color, linewidth=0.9,
                    linestyle="--", alpha=0.4, label=mname)

        ax.fill_between(eps_vals,
                        mean_zmps - std_zmps,
                        mean_zmps + std_zmps,
                        color=color, alpha=0.18, label="±1 std (motions)")
        ax.plot(eps_vals, mean_zmps, color=color, linewidth=2.2, linestyle="-",
                marker="s", markersize=7, label="Mean ZMP margin")
    else:
        ax.fill_between(eps_vals,
                        mean_zmps - std_zmps,
                        mean_zmps + std_zmps,
                        color=color, alpha=0.18, label="±1 std (trials)")
        ax.plot(eps_vals, mean_zmps, color=color, linewidth=2, linestyle="-",
                marker="s", markersize=7, label="Mean ZMP margin")

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0,
               label="Stability boundary")
    ax.set_xlabel("Reference frame noise RMS ε (m)")
    ax.set_ylabel("Mean ZMP lateral margin (m)")
    ax.set_title("Mechanism: ZMP Margin vs Reference Frame Noise")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    _save(fig, output_dir, "fig2_zmp_mechanism")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Heatmap (supplementary)
# ═════════════════════════════════════════════════════════════════════════════

def plot_recovery_heatmap(merged: dict, meta: dict, output_dir: str) -> None:
    """2D heatmap: mean recovery rate as function of (ε, push_velocity)."""
    eps_vals  = meta["epsilon_values"]
    pvel_vals = meta["push_velocities"]
    n_eps, n_pvel = len(eps_vals), len(pvel_vals)

    matrix = np.array([
        [merged[ei][pi]["mean_rate"] * 100 for ei in range(n_eps)]
        for pi in range(n_pvel)
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100,
                   aspect="auto", origin="lower")

    for pi in range(n_pvel):
        for ei in range(n_eps):
            val = matrix[pi, ei]
            color = "black" if 20 < val < 80 else "white"
            ax.text(ei, pi, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n_eps))
    ax.set_xticklabels([f"{e:.2f}" for e in eps_vals], rotation=30)
    ax.set_yticks(range(n_pvel))
    ax.set_yticklabels([f"{v:.1f}" for v in pvel_vals])
    ax.set_xlabel("Reference frame noise ε (m)")
    ax.set_ylabel("Push velocity Δv (m/s)")

    n_motions = len(merged[0][0]["motion_names"])
    suffix = f" (mean over {n_motions} motions)" if n_motions > 1 else ""
    ax.set_title(f"Recovery Rate Heatmap (ε × Push Magnitude){suffix}")
    plt.colorbar(im, ax=ax, label="Recovery rate (%)")

    _save(fig, output_dir, "fig3_recovery_heatmap")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — ZMP time-series per motion (qualitative)
# ═════════════════════════════════════════════════════════════════════════════

def plot_zmp_timeseries_multi(
    named_summaries: list[tuple[str, dict]],
    meta: dict,
    output_dir: str,
) -> None:
    """
    One subplot per motion sequence, showing ZMP margin summary statistics
    (cannot show raw trajectories without the full store, but can show the
    settle-phase mean ZMP vs epsilon as a bar chart per motion).

    If called with full ResultsStore objects, use plot_zmp_timeseries_from_stores.
    """
    eps_vals = meta["epsilon_values"]
    n_eps    = len(eps_vals)
    n_pvel   = len(meta["push_velocities"])
    n_motion = len(named_summaries)

    fig, axes = plt.subplots(
        1, n_motion,
        figsize=(4 * n_motion, 4),
        sharey=True,
    )
    if n_motion == 1:
        axes = [axes]

    fig.suptitle("ZMP Margin vs ε — per Motion Sequence", y=1.02)

    for ax_idx, (name, summary) in enumerate(named_summaries):
        ax = axes[ax_idx]
        # Average settle ZMP over push velocities
        means = []
        stds  = []
        for ei in range(n_eps):
            zm = [summary[ei][pi]["mean_zmp_settle"]
                  for pi in range(n_pvel)
                  if not np.isnan(summary[ei][pi]["mean_zmp_settle"])]
            sz = [summary[ei][pi]["std_zmp_settle"]
                  for pi in range(n_pvel)
                  if not np.isnan(summary[ei][pi]["std_zmp_settle"])]
            means.append(np.mean(zm) if zm else float("nan"))
            stds.append(np.mean(sz)  if sz else float("nan"))

        means = np.array(means)
        stds  = np.array(stds)

        ax.bar(range(n_eps), means, yerr=stds, capsize=4,
               color="#1976D2", alpha=0.7, error_kw={"elinewidth": 1.5})
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(range(n_eps))
        ax.set_xticklabels([f"{e:.2f}" for e in eps_vals], rotation=40, fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("ε (m)")
        if ax_idx == 0:
            ax.set_ylabel("Mean ZMP margin (m)")

    plt.tight_layout()
    _save(fig, output_dir, "fig4_zmp_per_motion")


def plot_zmp_timeseries_from_stores(
    named_stores: list[tuple[str, ResultsStore]],
    meta: dict,
    output_dir: str,
) -> None:
    """
    Full time-series ZMP plots using raw trial data.
    Shows settle + post-push ZMP trajectory medians for 3 epsilon levels.
    One row per motion sequence.
    """
    eps_vals      = meta["epsilon_values"]
    showcase_idxs = [0, len(eps_vals) // 2, len(eps_vals) - 1]
    n_motion      = len(named_stores)
    n_cols        = len(showcase_idxs)

    fig, axes = plt.subplots(n_motion, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_motion),
                             sharey=True)
    if n_motion == 1:
        axes = [axes]

    for row, (name, store) in enumerate(named_stores):
        for col, ei in enumerate(showcase_idxs):
            ax = axes[row][col]
            all_settle, all_post = [], []

            for ti in range(meta["n_trials"]):
                key = (ei, 0, ti)
                if key not in store._data:
                    continue
                r = store._data[key]
                if not r.fallen_before_push:
                    all_settle.append(r.zmp_margins_settle)
                    if r.zmp_margins_post:
                        all_post.append(r.zmp_margins_post)

            if not all_settle:
                ax.set_title(f"{name}\nε={eps_vals[ei]:.2f}m\n(no data)")
                continue

            def pad_med(seqs):
                if not seqs:
                    return np.array([])
                L = max(len(s) for s in seqs)
                padded = [s + [s[-1]] * (L - len(s)) for s in seqs]
                return np.median(padded, axis=0)

            sm = pad_med(all_settle)
            pm = pad_med(all_post)

            t_s = np.arange(len(sm)) * 0.02
            ax.plot(t_s, sm, color="#1976D2", linewidth=1.5)
            ax.axvline(t_s[-1], color="black", linestyle="--", linewidth=1.0)
            if len(pm):
                t_p = t_s[-1] + np.arange(len(pm)) * 0.02
                ax.plot(t_p, pm, color="#F44336", linewidth=1.5)

            ax.axhline(0.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{name} | ε={eps_vals[ei]:.2f}m")
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("ZMP margin (m)")

    plt.tight_layout()
    _save(fig, output_dir, "fig4b_zmp_timeseries_full")


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, output_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {name}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot robustness validation results.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results_dir",  type=str,
                       help="Single results directory containing meta.json and results_raw.npz.")
    group.add_argument("--results_dirs", type=str, nargs="+",
                       help="Multiple results directories (one per motion sequence).")

    parser.add_argument("--motion_names", type=str, nargs="*", default=None,
                        help="Human-readable labels for each directory "
                             "(default: directory basenames).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save figures.")

    args = parser.parse_args()

    if args.results_dir:
        load_and_plot_multi([args.results_dir],
                            motion_names=None,
                            output_dir=args.output_dir)
    else:
        load_and_plot_multi(args.results_dirs,
                            motion_names=args.motion_names,
                            output_dir=args.output_dir)
