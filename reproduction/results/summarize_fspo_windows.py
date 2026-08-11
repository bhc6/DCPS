from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WINDOWS = ROOT / "fspo_run_time_windows.csv"
RAW = ROOT / "raw_wandb_data.csv"
OUT = ROOT / "fspo_run_windows_with_overlap.csv"

SCORE_COLS = [
    "summary_best_score",
    "summary_final_score",
    "summary_test_score",
    "summary_score",
    "summary_val_score",
    "summary_best_val_score",
]


def task_name(text):
    s = str(text).lower()
    if "hotpot" in s:
        return "HotpotQA"
    if "ifbench" in s:
        return "IFBench"
    if "hover" in s:
        return "Hover"
    if "pupa" in s or "papillon" in s:
        return "PUPA"
    if "aime" in s:
        return "AIME-2025"
    if "livebench" in s:
        return "LiveBench-Math"
    return ""


def main():
    windows = pd.read_csv(WINDOWS)
    raw = pd.read_csv(RAW, low_memory=False)
    existing_scores = [c for c in SCORE_COLS if c in raw.columns]
    cols = ["project", "run_id", "run_name"] + existing_scores
    merged = windows.merge(raw[cols], on=["project", "run_id", "run_name"], how="left")
    merged["start"] = pd.to_datetime(merged["created_at"], utc=True)
    merged["end"] = merged["start"] + pd.to_timedelta(pd.to_numeric(merged["runtime_mins"], errors="coerce"), unit="m")
    merged["task"] = (merged["project"].astype(str) + " " + merged["run_name"].astype(str)).map(task_name)
    if existing_scores:
        merged["score_for_selection"] = merged[existing_scores].bfill(axis=1).iloc[:, 0]
    else:
        merged["score_for_selection"] = pd.NA
    merged["overlap_run_ids"] = ""
    merged["overlap_count_same_model"] = 0
    for _, group in merged.groupby("config_solver_model"):
        for idx, row in group.iterrows():
            overlaps = group[(group.index != idx) & (group["start"] < row["end"]) & (group["end"] > row["start"])]
            merged.loc[idx, "overlap_count_same_model"] = len(overlaps)
            merged.loc[idx, "overlap_run_ids"] = ";".join(overlaps["run_id"].astype(str).tolist())
    out_cols = [
        "project",
        "run_id",
        "run_name",
        "task",
        "config_solver_model",
        "runtime_mins",
        "created_at",
        "start",
        "end",
        "score_for_selection",
        "overlap_count_same_model",
        "overlap_run_ids",
    ]
    merged[out_cols].to_csv(OUT, index=False)
    print(merged[out_cols].to_string(index=False, max_rows=100))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
