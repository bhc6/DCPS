from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
QUALITY = ROOT / "fspo_openrouter_match_quality_per_run.csv"
FSPO_OVERLAP = ROOT / "fspo_run_windows_with_overlap.csv"
ALL_OVERLAP = ROOT / "fspo_all_run_overlap_audit.csv"
TOKENS = ROOT / "fspo_wandb_history_tokens_per_run.csv"
OUT = ROOT / "fspo_openrouter_recovery_confidence_v3.csv"


def classify(row):
    if bool(row.get("has_wandb_tokens", False)):
        return "A_wandb_direct"
    if row["matched_requests"] <= 0:
        return "D_no_match"
    if row["n_fspo_overlaps"] > 0:
        return "C_overlap_same_model_fspo"
    if row["n_non_fspo_overlaps"] > 0:
        return "C_overlap_same_model_non_fspo"
    if row["n_api_keys"] <= 1:
        return "A_clean_time_model"
    if row["expected_key_request_frac"] >= 0.99:
        return "B_same_task_key_mix"
    if row["top_key_request_frac"] >= 0.9:
        return "B_minor_key_mix"
    return "C_cross_task_key_mix"


def main():
    quality = pd.read_csv(QUALITY)
    fspo_overlap = pd.read_csv(FSPO_OVERLAP)[["project", "run_id", "overlap_count_same_model", "overlap_run_ids"]]
    all_overlap = pd.read_csv(ALL_OVERLAP).rename(columns={"fspo_run_id": "run_id"})
    all_overlap = all_overlap[["run_id", "n_fspo_overlaps", "n_non_fspo_overlaps", "non_fspo_overlap_ids", "non_fspo_overlap_names", "non_fspo_overlap_projects"]]
    merged = quality.merge(fspo_overlap, on=["project", "run_id"], how="left")
    merged = merged.merge(all_overlap, on="run_id", how="left")

    if TOKENS.exists():
        tokens = pd.read_csv(TOKENS)[["run_id", "input_tokens", "output_tokens"]]
        tokens["has_wandb_tokens"] = tokens["input_tokens"].notna() & tokens["output_tokens"].notna()
        merged = merged.merge(tokens[["run_id", "has_wandb_tokens"]], on="run_id", how="left")
    else:
        merged["has_wandb_tokens"] = False

    for col in ["has_wandb_tokens"]:
        merged[col] = merged[col].fillna(False)
    for col in ["overlap_count_same_model", "n_fspo_overlaps", "n_non_fspo_overlaps"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    merged["confidence"] = merged.apply(classify, axis=1)
    cols = [
        "confidence",
        "project",
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
        "n_fspo_overlaps",
        "n_non_fspo_overlaps",
        "overlap_run_ids",
        "non_fspo_overlap_ids",
        "non_fspo_overlap_names",
        "non_fspo_overlap_projects",
        "has_wandb_tokens",
        "key_breakdown",
    ]
    merged[cols].sort_values(["confidence", "model_short", "task", "run_id"]).to_csv(OUT, index=False)
    print(merged.groupby(["confidence", "model_short", "task"]).size().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
