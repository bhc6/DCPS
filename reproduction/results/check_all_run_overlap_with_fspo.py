"""Check ALL WandB runs (not just FSPO) that overlap in time+model with FSPO candidates."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_wandb_data.csv"
WINDOWS = ROOT / "fspo_run_time_windows.csv"
OUT = ROOT / "fspo_all_run_overlap_audit.csv"

MODEL_NORM = {
    "openrouter/openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openrouter/openai/gpt-4o-mini": "gpt-4o-mini",
    "openrouter/qwen/qwen3-8b": "qwen3-8b",
}


def norm_model(m):
    s = str(m).lower()
    for k, v in MODEL_NORM.items():
        if v in s:
            return v
    return s


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    raw = raw[raw["config_solver_model"].astype(str).str.contains("openrouter", case=False, na=False)].copy()
    windows = pd.read_csv(WINDOWS)

    # Get created_at for ALL openrouter runs via WandB API cache or raw_wandb_data
    # We already have fspo windows; for non-FSPO we need to fetch created_at
    # But raw_wandb_data doesn't have created_at. Use the export we already made.
    # Alternative: use the fspo_run_time_windows for FSPO, and for non-FSPO runs,
    # check if they exist in activity CSV by matching run times.
    # Simplest: fetch created_at for all 48 openrouter runs.

    import wandb
    api = wandb.Api()

    fspo_ids = set(windows["run_id"].astype(str))
    all_runs = []
    for _, row in raw.iterrows():
        run_id = str(row["run_id"])
        project = str(row["project"])
        entity = "awesome-prompt"
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            created_at = getattr(run, "created_at", "")
        except Exception:
            created_at = ""
        all_runs.append({
            "project": project,
            "run_id": run_id,
            "run_name": str(row.get("run_name", "")),
            "config_solver_model": str(row.get("config_solver_model", "")),
            "model_norm": norm_model(row.get("config_solver_model", "")),
            "runtime_mins": row.get("runtime_mins", 0),
            "created_at": created_at,
            "is_fspo": run_id in fspo_ids,
        })

    all_df = pd.DataFrame(all_runs)
    all_df["start"] = pd.to_datetime(all_df["created_at"], utc=True, errors="coerce")
    all_df["end"] = all_df["start"] + pd.to_timedelta(pd.to_numeric(all_df["runtime_mins"], errors="coerce"), unit="m")

    # For each FSPO candidate, find ALL overlapping runs with same model
    fspo_df = all_df[all_df["is_fspo"]].copy()
    results = []
    for _, fspo_row in fspo_df.iterrows():
        if pd.isna(fspo_row["start"]) or pd.isna(fspo_row["end"]):
            continue
        same_model = all_df[
            (all_df["model_norm"] == fspo_row["model_norm"]) &
            (all_df["run_id"] != fspo_row["run_id"]) &
            (all_df["start"] < fspo_row["end"]) &
            (all_df["end"] > fspo_row["start"])
        ]
        non_fspo_overlaps = same_model[~same_model["is_fspo"]]
        fspo_overlaps = same_model[same_model["is_fspo"]]
        results.append({
            "fspo_run_id": fspo_row["run_id"],
            "fspo_run_name": fspo_row["run_name"],
            "fspo_project": fspo_row["project"],
            "model_norm": fspo_row["model_norm"],
            "start": fspo_row["start"],
            "end": fspo_row["end"],
            "n_fspo_overlaps": len(fspo_overlaps),
            "n_non_fspo_overlaps": len(non_fspo_overlaps),
            "non_fspo_overlap_ids": ";".join(non_fspo_overlaps["run_id"].tolist()),
            "non_fspo_overlap_names": ";".join(non_fspo_overlaps["run_name"].tolist()),
            "non_fspo_overlap_projects": ";".join(non_fspo_overlaps["project"].tolist()),
        })

    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False)
    print(out[["fspo_run_id", "fspo_run_name", "model_norm", "n_fspo_overlaps", "n_non_fspo_overlaps", "non_fspo_overlap_names"]].to_string(index=False, max_rows=60))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
