"""Canonical naming layer for the DCPS paper artifacts.

This module is the single source of truth that maps the *historical* names found
in the frozen wandb export (``raw_wandb_data.csv``) to the **canonical paper
names** used in ``main_v3.tex``. Every other script should import from here
rather than re-implementing method classification.

Why this exists
---------------
The single-prompt (StablePrompt) audit accumulated four naming layers over the
project's life:

    paper (main_v3.tex)   old draft      CSV `method`   wandb projects
    -------------------   ------------   ------------   ------------------------
    StablePrompt-PPO      StablePrompt   StablePrompt-  GFB_II/QA/TC, GFB_BBH_*,
                          -PPO           PPO            *_tc_GFB
    StablePrompt-DCPS     RAE            AlgPrompt      algprompt_*, mmlu, II*
    DCPS-Compound         FSPO           FSPO           *-dynamic-fewshot

"GFB" = *GFlowPO Baseline*, a leftover from an earlier GFlowPO-comparison era.
Confusingly the ``GFB_*`` wandb projects actually hold StablePrompt-PPO runs, and
the ``mmlu``/``II`` projects contain BOTH real-PPO and PPO-disabled runs. So the
wandb project name does NOT determine the method: whether PPO actually updated
does. That decision lives in :func:`ppo_ran` / :func:`classify_raw_method`, ported
verbatim from ``analyze_paper_data.py`` so the two can never diverge.

The acronym "DCPS" itself was overloaded in old code (dcps_supplement called it
"Dynamic few-shot Context Prompt Sampling"; GFB called it "Generate & Filter").
The paper's canonical expansion is the only correct one:

    DCPS = Demonstration-Conditioned Prompt Search

Usage
-----
    from dcps_naming import canonical_method, to_canonical

    df["paper_method"] = df.apply(canonical_method, axis=1)   # row -> paper name
    to_canonical("FSPO")          # -> "DCPS-Compound"
    to_canonical("AlgPrompt")     # -> "StablePrompt-DCPS"
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Canonical paper name constants (use these everywhere reader-facing)
# --------------------------------------------------------------------------
DCPS_EXPANSION = "Demonstration-Conditioned Prompt Search"

STABLEPROMPT_PPO = "StablePrompt-PPO"
STABLEPROMPT_DCPS = "StablePrompt-DCPS"
DCPS_COMPOUND = "DCPS-Compound"
GEPA = "GEPA"
BASELINE = "Baseline"
UNKNOWN = "Unknown"

# --------------------------------------------------------------------------
# Static string-level crosswalk: any historical label -> canonical paper name.
# This handles the CSV `method` column and free-text references. Row-level
# classification (below) is authoritative when a full run row is available,
# because PPO-vs-control cannot be decided from a string alone.
# --------------------------------------------------------------------------
RAW_TO_PAPER: dict[str, str] = {
    # DCPS-Compound (paper) <- FSPO (old draft) <- dynamic-fewshot (wandb)
    "FSPO": DCPS_COMPOUND,
    "DynamicFewshot": DCPS_COMPOUND,
    "dynamic-fewshot": DCPS_COMPOUND,
    # StablePrompt-DCPS (paper) <- RAE (old draft) <- AlgPrompt (CSV) <- GFB-NoPPO
    "AlgPrompt": STABLEPROMPT_DCPS,
    "RAE": STABLEPROMPT_DCPS,
    "GFB-NoPPO": STABLEPROMPT_DCPS,
    # StablePrompt-PPO (paper). cs= controlled-sampling variants collapse to the
    # single method reported in the paper.
    "StablePrompt-PPO": STABLEPROMPT_PPO,
    "APPO": STABLEPROMPT_PPO,
    # passthroughs
    "GEPA": GEPA,
    "Baseline": BASELINE,
    "Unknown": UNKNOWN,
}


def to_canonical(raw_method: str) -> str:
    """Map a historical method *string* to its canonical paper name.

    ``StablePrompt-PPO(cs=0.05)``-style controlled-sampling subtypes collapse to
    the single paper method ``StablePrompt-PPO``.
    """
    if raw_method is None:
        return UNKNOWN
    s = str(raw_method).strip()
    if s.startswith("StablePrompt-PPO"):  # incl. (cs=...) subtypes
        return STABLEPROMPT_PPO
    return RAW_TO_PAPER.get(s, s)


# --------------------------------------------------------------------------
# Row-level classification (authoritative).
# Ported verbatim from analyze_paper_data.py:determine_method so the frozen
# snapshot is classified identically to the paper pipeline. Keep in sync; if
# analyze_paper_data.py changes its logic, change it here too.
# --------------------------------------------------------------------------
_PPO_SIGNAL_COLS = [
    "summary_global_step", "summary_step",
    "summary_ppo/time/ppo/total", "summary_PPO/time/ppo/total",
    "summary_time/ppo/total", "summary_ppo/time/ppo/optimize_step",
    "summary_PPO/time/ppo/optimize_step", "summary_time/ppo/optimize_step",
    "summary_final/total_model_updates",
]


def ppo_ran(row) -> bool:
    """True iff any PPO counter/timer field is present (=> PPO actually updated)."""
    for c in _PPO_SIGNAL_COLS:
        if c in row and pd.notna(row.get(c)):
            return True
    return False


def classify_raw_method(row) -> str:
    """Classify a raw wandb run row into the *old CSV* method label.

    Mirrors analyze_paper_data.py exactly. Use :func:`canonical_method` to get
    the paper name in one step.
    """
    project = str(row.get("project", "")).lower()
    run_name_l = str(row.get("run_name", "")).lower()

    if "dynamic-fewshot" in project:
        return "FSPO"
    if project == "gepa":
        opt = row.get("config_optimizer_name")
        if pd.notna(opt):
            return "Baseline" if "Baseline" in str(opt) else str(opt)
        return "GEPA"
    if "baseline" in project:
        return "Baseline"
    if any(b in run_name_l for b in ["baseline", "cot", "zeroshot"]):
        return "Baseline"

    ppe_present = pd.notna(row.get("config_prompt_per_example"))
    bs_present = pd.notna(row.get("config_batch_size"))
    if ppe_present and bs_present:
        if ppo_ran(row):
            cs = row.get("config_cs")
            if pd.notna(cs):
                try:
                    return f"StablePrompt-PPO(cs={float(cs):g})"
                except Exception:
                    return "StablePrompt-PPO"
            return "StablePrompt-PPO"
        return "AlgPrompt"

    return "Unknown"


def canonical_method(row) -> str:
    """Row -> canonical paper method name (the function to use downstream)."""
    return to_canonical(classify_raw_method(row))


if __name__ == "__main__":
    # Smoke test against the frozen snapshot if present.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "raw_wandb_data.csv")
    if not os.path.exists(path):
        print(f"[skip] no snapshot at {path}")
    else:
        df = pd.read_csv(path, low_memory=False)
        df["paper_method"] = df.apply(canonical_method, axis=1)
        print("Canonical paper-method counts (excluding Unknown):")
        counts = df["paper_method"].value_counts()
        for name, n in counts.items():
            print(f"  {name:<20} {n}")
