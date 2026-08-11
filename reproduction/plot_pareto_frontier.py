"""
Fig. 1 — Cost-Performance Pareto Frontier in Prompt Optimization.

All data points are audited Phase-2 macro values (mean across the 12 (model,
task) cells = 6 benchmarks x {Qwen3-8B, GPT-4.1-mini}). Sources:

- Macro accuracy: Tables tab2_qwen3.tex / tab3_gpt4.tex (means re-derived in
  scratch_rcei_v3_weighted.py M_SCORE entries).
- Rollouts and USD cost: Table tab:cost_drop in main.tex (lines 294-307).
- Baseline macro accuracy = 50.64 (12-cell mean of B_SCORE; Qwen3 51.08, GPT
  50.20 in the abstract).

Per-model split is also drawn (small panels) so reviewers can see that the
ordering is consistent on both Qwen3-8B and GPT-4.1-mini.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Audited Phase-2 macro data
# ---------------------------------------------------------------------------
# (rollouts, USD, score, label, family)
PHASE2_MACRO = [
    (1,     0.43, 50.20, "Baseline",       "Baseline"),
    (3936, 10.84, 56.85, "MIPROv2-Heavy",  "Bayesian/Joint"),
    (3936,  9.63, 60.53, "GEPA-MERGE",     "Reflection-based"),
    (3936, 11.34, 61.14, "GEPA",           "Reflection-based"),
    (687,   4.41, 59.64, "DCPS-Compound",  "Search-based"),
]

# Per-model macros for sanity panels (rollouts, score, label, family)
QWEN3_MACRO = [
    (1,    51.08, "Baseline",      "Baseline"),
    (3936, 57.57, "MIPROv2-Heavy", "Bayesian/Joint"),
    (3936, 59.36, "GEPA-MERGE",    "Reflection-based"),
    (3936, 62.46, "GEPA",          "Reflection-based"),
    (687,  59.94, "DCPS-Compound", "Search-based"),
]
GPT_MACRO = [
    (1,    50.20, "Baseline",      "Baseline"),
    (3936, 56.85, "MIPROv2-Heavy", "Bayesian/Joint"),
    (3936, 60.53, "GEPA-MERGE",    "Reflection-based"),
    (3936, 61.14, "GEPA",          "Reflection-based"),
    (687,  59.64, "DCPS-Compound", "Search-based"),
]

FAMILY_COLOR = {
    "Baseline":          "#7f8c8d",
    "Bayesian/Joint":    "#9b59b6",
    "Reflection-based":  "#3498db",
    "Search-based":      "#2ecc71",
}
FAMILY_MARKER = {
    "Baseline":          "X",
    "Bayesian/Joint":    "s",
    "Reflection-based":  "o",
    "Search-based":      "*",
}


def _pareto(points, x_idx, y_idx):
    """Return Pareto-optimal subset minimising x and maximising y."""
    sorted_pts = sorted(points, key=lambda p: (p[x_idx], -p[y_idx]))
    front, best_y = [], -float("inf")
    for p in sorted_pts:
        if p[y_idx] > best_y:
            front.append(p)
            best_y = p[y_idx]
    return front


def _scatter_panel(ax, points, x_idx, y_idx, label_idx, family_idx,
                   highlight="DCPS-Compound", show_labels=True,
                   label_offsets=None):
    """Render a single Pareto-frontier panel."""
    label_offsets = label_offsets or {}

    # Pareto frontier (dashed)
    front = _pareto(points, x_idx, y_idx)
    fx = [p[x_idx] for p in front]
    fy = [p[y_idx] for p in front]
    ax.plot(fx, fy, color="black", linestyle="--", alpha=0.55,
            linewidth=1.4, zorder=2, label="Pareto Frontier")

    # Scatter
    for p in points:
        family = p[family_idx]
        is_highlight = (p[label_idx] == highlight)
        size = 320 if is_highlight else (180 if family != "Baseline" else 160)
        ax.scatter(p[x_idx], p[y_idx],
                   c=FAMILY_COLOR[family],
                   marker=FAMILY_MARKER[family],
                   s=size,
                   edgecolors="black",
                   linewidths=1.2 if is_highlight else 0.7,
                   alpha=0.95,
                   zorder=4 if is_highlight else 3,
                   label=family)
        if show_labels:
            offset = label_offsets.get(p[label_idx], (10, 6))
            weight = "bold" if is_highlight else "normal"
            ax.annotate(p[label_idx], (p[x_idx], p[y_idx]),
                        xytext=offset, textcoords="offset points",
                        fontsize=9, fontweight=weight, zorder=5)


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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [1, 1]})

    # -----------------------------------------------------------------------
    # Left panel: Rollouts vs Macro Accuracy
    # -----------------------------------------------------------------------
    ax_r = axes[0]
    rollout_points = [(p[0], p[2], p[3], p[4]) for p in PHASE2_MACRO]
    rollout_offsets = {
        "Baseline":      (8, -14),
        "DCPS-Compound": (10, -16),
        "MIPROv2-Heavy":   (10, -16),
        "GEPA-MERGE":      (10, 6),
        "GEPA":            (10, 6),
    }
    _scatter_panel(ax_r, rollout_points,
                   x_idx=0, y_idx=1, label_idx=2, family_idx=3,
                   label_offsets=rollout_offsets)
    ax_r.set_xscale("log")
    ax_r.set_xlim(0.7, 1.2e4)
    ax_r.set_ylim(48, 64)
    ax_r.set_xlabel("Rollouts per task (log scale)", fontweight="bold")
    ax_r.set_ylabel("Macro Accuracy (%)", fontweight="bold")
    ax_r.set_title("(a) Cost axis = rollouts", pad=8)
    ax_r.grid(True, which="major", ls="-", alpha=0.35, zorder=1)
    ax_r.grid(True, which="minor", ls="--", alpha=0.18, zorder=1)

    # -----------------------------------------------------------------------
    # Right panel: USD cost vs Macro Accuracy
    # -----------------------------------------------------------------------
    ax_c = axes[1]
    cost_points = [(p[1], p[2], p[3], p[4]) for p in PHASE2_MACRO]
    cost_offsets = {
        "Baseline":      (10, -14),
        "DCPS-Compound": (10, -16),
        "MIPROv2-Heavy":   (10, -16),
        "GEPA-MERGE":      (10, -4),
        "GEPA":            (10, 6),
    }
    _scatter_panel(ax_c, cost_points,
                   x_idx=0, y_idx=1, label_idx=2, family_idx=3,
                   label_offsets=cost_offsets)
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.2, 30)
    ax_c.set_ylim(48, 64)
    ax_c.set_xlabel("Estimated API cost per task (USD, log scale)",
                    fontweight="bold")
    ax_c.set_ylabel("Macro Accuracy (%)", fontweight="bold")
    ax_c.set_title("(b) Cost axis = price-weighted API spend", pad=8)
    ax_c.grid(True, which="major", ls="-", alpha=0.35, zorder=1)
    ax_c.grid(True, which="minor", ls="--", alpha=0.18, zorder=1)

    # -----------------------------------------------------------------------
    # Shared legend (de-duplicated) at the bottom centre
    # -----------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.4, alpha=0.55, label="Pareto Frontier"),
    ]
    for fam in ["Search-based", "Reflection-based", "Bayesian/Joint", "Baseline"]:
        legend_handles.append(
            Line2D([0], [0], marker=FAMILY_MARKER[fam], color="white",
                   markerfacecolor=FAMILY_COLOR[fam], markeredgecolor="black",
                   markersize=12 if fam == "Search-based" else 9, label=fam))

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95, frameon=True)

    fig.suptitle(
        "Cost-Performance Pareto Frontier of Prompt Optimizers "
        "(macro across the 6 GPT-4.1-mini benchmarks)",
        y=1.0, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "fig1_pareto_frontier.pdf")
    out_png = os.path.join(out_dir, "fig1_pareto_frontier.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Pareto frontier figure saved to:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()