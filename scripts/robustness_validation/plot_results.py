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
import csv
import json
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


def _discover_motion_result_dirs(run_dir: str) -> list[str]:
    """Find per-motion result directories under a decoupled validation run."""
    root = Path(run_dir).expanduser()
    candidates = sorted(root.glob("motions/*/*"))
    result_dirs: list[str] = []
    for p in candidates:
        if not ((p / "meta.json").is_file() and (p / "results_raw.npz").is_file()):
            continue
        status_path = p / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text()).get("status")
            except Exception:
                status = None
            if status not in (None, "completed"):
                print(f"[plot] Skipping {p}: status={status}")
                continue
        result_dirs.append(str(p))
    return result_dirs


def load_and_plot_run_root(results_dir: str, output_dir: str | None = None) -> None:
    """
    Load a decoupled run root:

      run_xxx/
        motions/Walking/motion_a/{meta.json,results_raw.npz,...}
        motions/Turning/motion_b/{meta.json,results_raw.npz,...}

    and generate Fig1/Fig2 that place all motion groups on the same axes.
    """
    result_dirs = _discover_motion_result_dirs(results_dir)
    if not result_dirs:
        raise FileNotFoundError(f"No per-motion results found under {results_dir}/motions/*/*")

    stores = [(Path(d).name, ResultsStore.load(d)) for d in result_dirs]

    if output_dir is None:
        output_dir = os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    _write_summary_all_csv(stores, os.path.join(results_dir, "summary_all.csv"))
    plot_grouped_recovery_curve(stores, output_dir, perturbation_mode="composite")
    plot_grouped_zmp_mechanism(stores, output_dir, perturbation_mode="composite")

    # Keep the old detailed multi-motion plots as appendix material.
    appendix_dir = os.path.join(output_dir, "appendix")
    os.makedirs(appendix_dir, exist_ok=True)
    motion_names = [f"{s.meta.get('motion_group', 'Ungrouped')}:{name}" for name, s in stores]
    load_and_plot_multi(result_dirs, motion_names=motion_names, output_dir=appendix_dir)

    print(f"[plot] Run-root figures saved to {output_dir}")


def load_and_plot_grouped_dirs(
    results_dirs: list[str],
    output_dir: str | None = None,
    perturbation_mode: str = "composite",
) -> None:
    """Generate paper-style grouped figures from arbitrary per-motion result dirs."""
    result_dirs = [_validate_results_dir(path) for path in results_dirs]
    stores = [(Path(d).name, ResultsStore.load(d)) for d in result_dirs]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(result_dirs[0]), "figures_grouped")
    os.makedirs(output_dir, exist_ok=True)
    _write_summary_all_csv(stores, os.path.join(output_dir, "summary_all.csv"))
    plot_grouped_recovery_curve(stores, output_dir, perturbation_mode=perturbation_mode)
    plot_grouped_zmp_mechanism(stores, output_dir, perturbation_mode=perturbation_mode)
    print(f"[plot] Grouped figures saved to {output_dir}")


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
#  FIGURE 1 — End-to-end success rate vs ε
# ═════════════════════════════════════════════════════════════════════════════

def plot_recovery_curve(
    merged: dict,
    meta: dict,
    output_dir: str,
    multi_mode: bool = False,
) -> None:
    """
    Main figure: end-to-end success rate (%) vs ε for each push velocity.
    Pre-push falls count as failures instead of being dropped, so every
    completed trial contributes a visible data point.

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

            ax.plot(eps_vals, mean_rates,
                    color=color, linewidth=2.2, linestyle="-",
                    marker="o", markersize=6,
                    label=f"Mean — Δv={pv} m/s")
        else:
            ax.plot(eps_vals, mean_rates,
                    color=color, linewidth=2, linestyle="-",
                    marker="o", markersize=6,
                    label=f"Δv = {pv} m/s")

    ax.set_xlabel("Reference frame noise ε")
    ax.set_ylabel("End-to-end success rate (%)")
    title = "Push-Recovery Rate"
    if multi_mode:
        n = len(merged[0][0]["motion_names"])
        title += f"\n(mean over {n} motion sequences)"
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)

    # Legend: motions in upper part, push velocities in lower part
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.75,
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
    Mechanism figure: minimum short-window margin after the push vs ε.
    The post-push window prevents later low-margin dance/kick phases from
    being misattributed to push recovery.
    """
    eps_vals  = meta["epsilon_values"]
    pvel_vals = meta["push_velocities"]
    n_eps, n_pvel = len(eps_vals), len(pvel_vals)
    color = "#1976D2"

    fig, ax = plt.subplots(figsize=(6, 4))

    # Average post-push minimum margin over push velocities.
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

        ax.plot(eps_vals, mean_zmps, color=color, linewidth=2.2, linestyle="-",
                marker="s", markersize=7, label="Min post-push margin")
    else:
        ax.plot(eps_vals, mean_zmps, color=color, linewidth=2, linestyle="-",
                marker="s", markersize=7, label="Min post-push margin")

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Reference frame noise ε")
    ax.set_ylabel("Minimum post-push margin (m)")
    ax.set_title("Stability Margin")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(left=0)

    _save(fig, output_dir, "fig2_zmp_mechanism")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Heatmap (supplementary)
# ═════════════════════════════════════════════════════════════════════════════

def plot_recovery_heatmap(merged: dict, meta: dict, output_dir: str) -> None:
    """2D heatmap: mean end-to-end success rate as function of (ε, push_velocity)."""
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
    ax.set_title(f"End-to-End Success Heatmap (ε × Push Magnitude){suffix}")
    plt.colorbar(im, ax=ax, label="End-to-end success rate (%)")

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
#  RUN-ROOT FIGURES — grouped by motion category
# ═════════════════════════════════════════════════════════════════════════════

def _mode_summary(store: ResultsStore, perturbation_mode: str) -> dict:
    try:
        return store.to_summary(perturbation_mode=perturbation_mode)
    except ValueError:
        return store.to_summary()


def _group_names(stores: list[tuple[str, ResultsStore]]) -> list[str]:
    groups = sorted({store.meta.get("motion_group", "Ungrouped") for _, store in stores})
    return groups + ["Overall"]


def _pooled_rate_by_eps(
    stores: list[tuple[str, ResultsStore]],
    group: str,
    perturbation_mode: str,
) -> tuple[list[float], list[float], list[int]]:
    ref_meta = stores[0][1].meta
    n_eps = len(ref_meta["epsilon_values"])
    n_pvel = len(ref_meta["push_velocities"])
    means, cis, totals = [], [], []
    real_groups = sorted({store.meta.get("motion_group", "Ungrouped") for _, store in stores})

    for ei in range(n_eps):
        groups_to_average = real_groups if group == "Overall" else [group]
        group_rates = []
        total = 0
        for group_name in groups_to_average:
            motion_rates = []
            for _, store in stores:
                if store.meta.get("motion_group", "Ungrouped") != group_name:
                    continue
                summary = _mode_summary(store, perturbation_mode)
                for pi in range(n_pvel):
                    cell = summary[ei][pi]
                    n = int(cell["n_total"])
                    rate = cell["end_to_end_success_rate"]
                    if n <= 0 or np.isnan(rate):
                        continue
                    motion_rates.append(float(rate))
                    total += n
            if motion_rates:
                group_rates.append(float(np.mean(motion_rates)))
        p = float(np.mean(group_rates)) if group_rates else float("nan")
        ci = float(np.sqrt(p * (1.0 - p) / total)) if total > 0 and not np.isnan(p) else 0.0
        means.append(p)
        cis.append(ci)
        totals.append(total)
    return means, cis, totals


def _pooled_zmp_by_eps(
    stores: list[tuple[str, ResultsStore]],
    group: str,
    perturbation_mode: str,
) -> tuple[list[float], list[float]]:
    ref_meta = stores[0][1].meta
    n_eps = len(ref_meta["epsilon_values"])
    n_pvel = len(ref_meta["push_velocities"])
    means, stds = [], []
    real_groups = sorted({store.meta.get("motion_group", "Ungrouped") for _, store in stores})

    for ei in range(n_eps):
        groups_to_average = real_groups if group == "Overall" else [group]
        group_vals = []
        for group_name in groups_to_average:
            vals = []
            for _, store in stores:
                if store.meta.get("motion_group", "Ungrouped") != group_name:
                    continue
                summary = _mode_summary(store, perturbation_mode)
                for pi in range(n_pvel):
                    cell = summary[ei][pi]
                    zmp = cell["mean_min_zmp_after_push"]
                    n = int(cell["n_total"])
                    if n <= 0 or np.isnan(zmp):
                        continue
                    vals.append(float(zmp))
            if vals:
                group_vals.append(float(np.mean(np.array(vals, dtype=float))))
        if group_vals:
            arr = np.array(group_vals, dtype=float)
            mean = float(np.mean(arr))
            means.append(mean)
            stds.append(float(np.std(arr)) if len(arr) > 1 else 0.0)
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
    return means, stds


def plot_grouped_recovery_curve(
    stores: list[tuple[str, ResultsStore]],
    output_dir: str,
    perturbation_mode: str = "composite",
) -> None:
    """Fig1: all motion categories and overall recovery rate on one axis."""
    eps_vals = stores[0][1].meta["epsilon_values"]
    groups = _group_names(stores)
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#212121"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gi, group in enumerate(groups):
        means, _, _ = _pooled_rate_by_eps(stores, group, perturbation_mode)
        y = np.array(means) * 100.0
        color = colors[gi % len(colors)]
        lw = 2.6 if group == "Overall" else 1.8
        alpha = 1.0 if group == "Overall" else 0.85
        ax.plot(eps_vals, y, marker="o", linewidth=lw, color=color, alpha=alpha, label=group)

    ax.set_xlabel("Reference frame noise ε")
    ax.set_ylabel("End-to-end success rate (%)")
    ax.set_title("Push-Recovery Rate")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, framealpha=0.8, loc="upper right")
    _save(fig, output_dir, "fig1_recovery_rate")


def plot_grouped_zmp_mechanism(
    stores: list[tuple[str, ResultsStore]],
    output_dir: str,
    perturbation_mode: str = "composite",
) -> None:
    """Fig2: mechanism view, all motion categories and post-push minimum margin."""
    eps_vals = stores[0][1].meta["epsilon_values"]
    groups = _group_names(stores)
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#212121"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for gi, group in enumerate(groups):
        means, _ = _pooled_zmp_by_eps(stores, group, perturbation_mode)
        y = np.array(means)
        color = colors[gi % len(colors)]
        lw = 2.6 if group == "Overall" else 1.8
        alpha = 1.0 if group == "Overall" else 0.85
        ax.plot(eps_vals, y, marker="s", linewidth=lw, color=color, alpha=alpha, label=group)

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Reference frame noise ε")
    ax.set_ylabel("Minimum post-push margin (m)")
    ax.set_title("Stability Margin")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, framealpha=0.8, loc="upper right")
    _save(fig, output_dir, "fig2_zmp_margin")


def _write_summary_all_csv(stores: list[tuple[str, ResultsStore]], path: str) -> None:
    rows = []
    for _, store in stores:
        modes = store.meta.get("perturbation_modes") or [store.meta.get("perturbation_mode", "composite")]
        for mode_idx, mode in enumerate(modes):
            summary = store.to_summary(mode_idx=mode_idx if store._has_mode_axis() else None)
            for ei in sorted(summary):
                for pi in sorted(summary[ei]):
                    cell = summary[ei][pi]
                    rows.append({
                        "motion_group": store.meta.get("motion_group", "Ungrouped"),
                        "motion_name": store.meta.get("motion_name", "unknown"),
                        "perturbation_mode": mode,
                        "epsilon_idx": ei,
                        "push_velocity_idx": pi,
                        "epsilon": cell["epsilon"],
                        "push_velocity": cell["push_velocity"],
                        "n_total": cell["n_total"],
                        "n_valid": cell["n_valid"],
                        "n_fallen_before": cell["n_fallen_before"],
                        "end_to_end_success_rate": cell["end_to_end_success_rate"],
                        "conditional_recovery_rate": cell["conditional_recovery_rate"],
                        "pre_fall_rate": cell["pre_fall_rate"],
                        "mean_zmp_settle": cell["mean_zmp_settle"],
                        "std_zmp_settle": cell["std_zmp_settle"],
                        "mean_min_zmp_after_push": cell["mean_min_zmp_after_push"],
                        "std_min_zmp_after_push": cell["std_min_zmp_after_push"],
                        "mean_zmp_after_push": cell["mean_zmp_after_push"],
                        "std_zmp_after_push": cell["std_zmp_after_push"],
                        "mean_margin_drop": cell["mean_margin_drop"],
                        "std_margin_drop": cell["std_margin_drop"],
                        "mean_push_phase": cell["mean_push_phase"],
                        "std_push_phase": cell["std_push_phase"],
                    })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[plot] Wrote {path}")


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
    parser.add_argument("--grouped", action="store_true",
                        help="For --results_dirs, plot paper-style curves grouped by motion_group.")
    parser.add_argument("--perturbation_mode", type=str, default="composite",
                        help="Perturbation mode to plot for grouped figures.")

    args = parser.parse_args()

    if args.results_dir:
        p = Path(args.results_dir).expanduser()
        if (p / "meta.json").is_file() and (p / "results_raw.npz").is_file():
            load_and_plot_multi([args.results_dir],
                                motion_names=None,
                                output_dir=args.output_dir)
        else:
            load_and_plot_run_root(args.results_dir, output_dir=args.output_dir)
    else:
        if args.grouped:
            load_and_plot_grouped_dirs(
                args.results_dirs,
                output_dir=args.output_dir,
                perturbation_mode=args.perturbation_mode,
            )
        else:
            load_and_plot_multi(args.results_dirs,
                                motion_names=args.motion_names,
                                output_dir=args.output_dir)
