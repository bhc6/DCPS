from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WINDOWS = ROOT / "fspo_run_windows_with_overlap.csv"
TOKENS = ROOT / "fspo_wandb_history_tokens_per_run.csv"
OUT = ROOT / "fspo_representative_runs_for_token_recovery.csv"


def main():
    windows = pd.read_csv(WINDOWS)
    tokens = pd.read_csv(TOKENS)
    merged = windows.merge(
        tokens[["project", "run_id", "input_tokens", "output_tokens", "estimated_cost_usd"]],
        on=["project", "run_id"],
        how="left",
    )
    merged["has_wandb_tokens"] = merged["input_tokens"].notna() & merged["output_tokens"].notna()
    merged["time_window_clean"] = merged["overlap_count_same_model"].fillna(0).astype(int).eq(0)
    merged["model_short"] = merged["config_solver_model"].astype(str).map(
        lambda s: "GPT-4.1-mini" if "gpt-4.1" in s.lower() else "GPT-4o-mini" if "gpt-4o" in s.lower() else "Qwen3-8B" if "qwen3-8b" in s.lower() else s
    )
    merged["score_num"] = pd.to_numeric(merged["score_for_selection"], errors="coerce")
    reps = []
    for (model, task), group in merged.groupby(["model_short", "task"], dropna=False):
        ranked = group.sort_values(["score_num", "has_wandb_tokens", "time_window_clean"], ascending=[False, False, False])
        top = ranked.iloc[0].copy()
        top["n_runs_in_cell"] = len(group)
        top["n_clean_windows_in_cell"] = int(group["time_window_clean"].sum())
        top["n_wandb_token_runs_in_cell"] = int(group["has_wandb_tokens"].sum())
        reps.append(top)
    out = pd.DataFrame(reps)
    cols = [
        "model_short",
        "task",
        "project",
        "run_id",
        "run_name",
        "score_for_selection",
        "runtime_mins",
        "created_at",
        "time_window_clean",
        "overlap_count_same_model",
        "overlap_run_ids",
        "has_wandb_tokens",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "n_runs_in_cell",
        "n_clean_windows_in_cell",
        "n_wandb_token_runs_in_cell",
    ]
    out[cols].to_csv(OUT, index=False)
    print(out[cols].to_string(index=False, max_rows=100))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
