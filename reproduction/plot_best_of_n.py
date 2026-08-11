"""
Fig. 2 - Best-of-N Search-Budget Sensitivity for FSPO.

All data points are audited from results/by_task_runs.csv (run-level WandB
exports). The candidate count is defined as N = num_iterations * top_k_prompts
(i.e., the total number of distinct candidate prompts evaluated during the
FSPO search loop).

Left panel (a): All FSPO seeds on Qwen3-8B x LiveBench-Math at N in {2, 20,
100} are plotted as individual scatter markers, with a mean curve. Baseline
(unoptimized human prompt) and GEPA reference scores from Table 2 are drawn
as horizontal lines.

Right panel (b): Paired N=20 -> N=100 slope chart across the five (model,
task) cells where both budget points were logged. This shows that a 5x
increase in candidate budget yields substantial gains on every cell, with
effect size ranging from +4.0 pp (GPT-4.1-mini x AIME-2025) to +33.95 pp
(Qwen3-8B x IFBench).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Audited data (from results/by_task_runs.csv, FSPO rows)
# ---------------------------------------------------------------------------
# Qwen3-8B x LiveBench-Math FSPO individual seeds, grouped by (N, fewshot)
LB_MATH_QWEN3_SEEDS = {
    (2, "3-shot"):    [31.75],
    (20, "3-shot"):   [29.37, 33.33, 35.71, 38.89],
    (20, "0-shot"):   [30.95, 33.33],
    (100, "3-shot"):  [66.67],
}

# Reference horizontals for left panel (from Table 2: Qwen3-8B Phase-2)
LB_MATH_BASELINE = 65.01
LB_MATH_GEPA = 70.57

# Right-panel cells: (label, color, N20_seeds, N100_seeds, baseline)
PAIRED_CELLS = [
    ("Qwen3-8B x LiveBench-Math",
     "#1f77b4",
     [29.37, 33.33, 35.71, 38.89],
     [66.67],
     65.01),
    ("Qwen3-8B x AIME-2025",
     "#2ca02c",
     [47.33],
     [54.67, 55.33, 60.00],
     47.33),
    ("Qwen3-8B x IFBench",
     "#d62728",
     [38.95, 40.82, 43.88],
     [75.17],
     38.61),
    ("GPT-4.1-mini x AIME-2025",
     "#9467bd",
     [48.00],
     [52.00],
     40.00),
    ("GPT-4.1-mini x IFBench",
     "#ff7f0e",
     [48.81, 51.53],
     [76.19],
     48.13),
]


def _mean_std(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _draw_left_panel(ax):
    """Per-seed scatter + mean curve on Qwen3-8B x LiveBench-Math."""
    # Individual seeds
    style = {
        "3-shot": dict(marker="o", color="#1f77b4", label="FSPO seeds (3-shot)"),
        "0-shot": dict(marker="^", color="#7f7f7f", label="FSPO seeds (0-shot ablation)"),
    }
    plotted_label = set()
    for (n, kind), scores in LB_MATH_QWEN3_SEEDS.items():
        s = style[kind]
        for v in scores:
            kw = dict(marker=s["marker"], color=s["color"],
                      s=70, edgecolors="black", linewidths=0.6,
                      alpha=0.8, zorder=3)
            if s["label"] not in plotted_label:
                kw["label"] = s["label"]
                plotted_label.add(s["label"])
            ax.scatter(n, v, **kw)

    # Mean curve over 3-shot only (for log-scale clarity)
    xs, ys, errs = [], [], []
    for n in [2, 20, 100]:
        seeds = LB_MATH_QWEN3_SEEDS.get((n, "3-shot"), [])
        if seeds:
            m, s = _mean_std(seeds)
            xs.append(n)
            ys.append(m)
            errs.append(s)
    ax.errorbar(xs, ys, yerr=errs, fmt="-", color="#1f77b4",
                linewidth=2, alpha=0.85, capsize=4, zorder=2,
                label="3-shot mean (+/- 1 sd)")

    # Reference horizontals
    ax.axhline(LB_MATH_BASELINE, color="#7f8c8d", linestyle=":",
               linewidth=1.4, alpha=0.9, zorder=1,
               label=f"Baseline (unoptimized) = {LB_MATH_BASELINE:.2f}")
    ax.axhline(LB_MATH_GEPA, color="#d62728", linestyle="--",
               linewidth=1.4, alpha=0.9, zorder=1,
               label=f"GEPA (3936 rollouts) = {LB_MATH_GEPA:.2f}")

    ax.set_xscale("log")
    ax.set_xticks([2, 20, 100])
    ax.set_xticklabels(["2", "20", "100"])
    ax.set_xlim(1.4, 160)
    ax.set_ylim(20, 80)
    ax.set_xlabel(r"Candidate budget $N = n_{\mathrm{iter}}\times k_{\mathrm{top}}$"
                  " (log scale)", fontweight="bold")
    ax.set_ylabel("Test Score on LiveBench-Math (%)", fontweight="bold")
    ax.set_title("(a) FSPO seed-level scatter on Qwen3-8B $\\times$ LiveBench-Math",
                 pad=8)
    ax.grid(True, which="major", ls="-", alpha=0.35, zorder=0)
    ax.grid(True, which="minor", ls="--", alpha=0.18, zorder=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)


def _draw_right_panel(ax):
    """Paired N=20 vs N=100 slope chart over 5 (model, task) cells."""
    x_positions = [0, 1]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r"$N=20$", r"$N=100$"])
    ax.set_xlim(-0.3, 1.55)

    legend_handles = []
    label_y = {
        "Qwen3-8B x LiveBench-Math": 67.0,
        "Qwen3-8B x AIME-2025": 58.0,
        "Qwen3-8B x IFBench": 74.2,
        "GPT-4.1-mini x AIME-2025": 51.5,
        "GPT-4.1-mini x IFBench": 78.0,
    }

    for label, color, n20_seeds, n100_seeds, baseline in PAIRED_CELLS:
        m20, s20 = _mean_std(n20_seeds)
        m100, s100 = _mean_std(n100_seeds)
        delta = m100 - m20

        # Slope line
        ax.plot(x_positions, [m20, m100], color=color,
                linewidth=2.0, alpha=0.85, zorder=3, marker="o", markersize=8,
                markeredgecolor="black", markeredgewidth=0.6)

        # Error bars where multiple seeds exist
        if len(n20_seeds) > 1:
            ax.errorbar(0, m20, yerr=s20, color=color,
                        capsize=4, alpha=0.85, zorder=3)
        if len(n100_seeds) > 1:
            ax.errorbar(1, m100, yerr=s100, color=color,
                        capsize=4, alpha=0.85, zorder=3)

        # Baseline tick on the left
        ax.plot([-0.18, -0.05], [baseline, baseline], color=color,
                linestyle=":", linewidth=1.2, alpha=0.7)

        # Right-side label with delta
        ax.annotate(f"{label}\n$\\Delta=${delta:+.1f} pp",
                    xy=(1, m100),
                    xytext=(1.08, label_y[label]),
                    fontsize=9, color=color, fontweight="bold",
                    va="center")

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2, marker="o",
                   markersize=7, label=label))

    # Baseline tick legend
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2,
               label="Per-cell baseline (left tick)"))

    ax.set_ylabel("Test Score (%)", fontweight="bold")
    ax.set_title("(b) Paired $N=20 \\to N=100$ across five logged cells",
                 pad=8)
    ax.grid(True, axis="y", ls="-", alpha=0.35, zorder=0)
    ax.set_ylim(28, 84)


def main():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    _draw_left_panel(axes[0])
    _draw_right_panel(axes[1])

    fig.suptitle("Search-Budget Sensitivity of FSPO (audited Best-of-$N$ pairs)",
                 y=1.0, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0.0, 1, 0.97])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "fig2_best_of_n_scaling.pdf")
    out_png = os.path.join(out_dir, "fig2_best_of_n_scaling.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Best-of-N figure saved to:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
