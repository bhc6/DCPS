from pathlib import Path

import pandas as pd
import wandb

ENTITY = "awesome-prompt"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_wandb_data.csv"
OUT = ROOT / "fspo_run_time_windows.csv"


def main():
    df = pd.read_csv(RAW, low_memory=False)
    text = df.get("project", pd.Series("", index=df.index)).astype(str) + " " + df.get("run_name", pd.Series("", index=df.index)).astype(str)
    mask = text.str.contains("dynamic|fewshot|fspo", case=False, regex=True, na=False)
    rows = df.loc[mask].copy()
    api = wandb.Api()
    out = []
    for _, row in rows.iterrows():
        project = row["project"]
        run_id = row["run_id"]
        item = {
            "project": project,
            "run_id": run_id,
            "run_name": row.get("run_name", ""),
            "runtime_mins": row.get("runtime_mins", ""),
            "config_solver_model": row.get("config_solver_model", ""),
            "config_benchmark_name": row.get("config_benchmark_name", ""),
        }
        try:
            run = api.run(f"{ENTITY}/{project}/{run_id}")
            item["created_at"] = getattr(run, "created_at", "")
            item["updated_at"] = getattr(run, "updated_at", "")
            item["state"] = getattr(run, "state", "")
            item["error"] = ""
        except Exception as exc:
            item["created_at"] = ""
            item["updated_at"] = ""
            item["state"] = ""
            item["error"] = repr(exc)
        out.append(item)
        print(item)
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
