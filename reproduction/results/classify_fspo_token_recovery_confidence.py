from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
QUALITY = ROOT / "fspo_openrouter_match_quality_per_run.csv"
OVERLAP = ROOT / "fspo_run_windows_with_overlap.csv"
OUT = ROOT / "fspo_openrouter_recovery_confidence.csv"


def classify(row):
    if bool(row.get("has_wandb_tokens", False)):
        return "A_wandb_direct"
    if row["matched_requests"] <= 0:
        return "D_no_match"
    if row["overlap_count_same_model"] > 0:
        return "C_overlap_same_model"
    if row["n_api_keys"] <= 1:
        return "A_clean_time_model"
    if row["top_key_request_frac"] >= 0.9:
        return "B_minor_key_mix"
    return "C_key_mix"


def main():
    quality = pd.read_csv(QUALITY)
    overlap = pd.read_csv(OVERLAP)[["project", "run_id", "overlap_count_same_model", "overlap_run_ids"]]
    merged = quality.merge(overlap, on=["project", "run_id"], how="left", suffixes=("", "_overlap"))
    token_path = ROOT / "fspo_wandb_history_tokens_per_run.csv"
    if token_path.exists():
        tok = pd.read_csv(token_path)[["project", "run_id", "input_tokens", "output_tokens", "estimated_cost_usd"]]
        tok["has_wandb_tokens"] = tok["input_tokens"].notna() & tok["output_tokens"].notna()
        merged = merged.merge(tok[["project", "run_id", "has_wandb_tokens"]], on=["project", "run_id"], how="left")
    else:
        merged["has_wandb_tokens"] = False
    merged["has_wandb_tokens"] = merged["has_wandb_tokens"].fillna(False)
    merged["overlap_count_same_model"] = pd.to_numeric(merged["overlap_count_same_model"], errors="coerce").fillna(0).astype(int)
    merged["confidence"] = merged.apply(classify, axis=1)
    cols = [
        "confidence",
        "task",
        "model_short",
        "run_id",
        "run_name",
        "matched_requests",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "n_api_keys",
        "top_api_key",
        "top_key_request_frac",
        "expected_key_request_frac",
        "overlap_count_same_model",
        "overlap_run_ids",
        "has_wandb_tokens",
        "key_breakdown",
    ]
    merged[cols].sort_values(["confidence", "model_short", "task", "run_id"]).to_csv(OUT, index=False)
    print(merged.groupby(["confidence", "model_short", "task"]).size().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
