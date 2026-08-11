import json
import re
import time
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "awesome-prompt"
PROJECTS = [
    "GFB_BBH_TG",
    "GFB_BBH_TC",
    "II_re_correct",
    "II_re",
    "GFB_II",
    "GFB_QA",
    "GFB_TC",
    "snli_tc_GFB",
    "qnli_tc_GFB",
    "mnli_tc_GFB",
    "rte_tc_GFB",
    "mrpc_tc_GFB",
    "sst2_tc_GFB",
    "II",
    "tta_TC",
    "bbh_tg_algprompt",
    "BBH",
    "mmlu",
    "algprompt_classification_snli",
    "algprompt_classification_qnli",
    "algprompt_classification_mnli",
    "algprompt_classification_rte",
    "algprompt_classification_mrpc",
    "algprompt_classification_sst2",
]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLEAN = RESULTS / "clean_paper_data.csv"
OUT_RAW = RESULTS / "phase1_gpu_runtime_audit.csv"
OUT_SUMMARY = RESULTS / "phase1_gpu_runtime_summary.csv"

GPU_PATTERNS = [
    (re.compile(r"a100", re.I), "A100"),
    (re.compile(r"a40", re.I), "A40"),
]

PHASE1_AGENT = "google/gemma-1.1-7b-it"
PHASE1_METHODS = {"StablePrompt-PPO", "AlgPrompt"}


def normalize_gpu(value):
    if value is None:
        return ""
    text = str(value)
    for pattern, label in GPU_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def infer_method(row):
    method = str(row.get("method", ""))
    if method in PHASE1_METHODS:
        return method
    ppo = str(row.get("ppo_used", "")).lower()
    project = str(row.get("project", ""))
    run_name = str(row.get("run_name", ""))
    if ppo == "true" and ("GFB" in project or "gemma" in run_name):
        return "StablePrompt-PPO"
    if method == "AlgPrompt" or (ppo == "false" and "meta_change" in run_name):
        return "AlgPrompt"
    return method


def candidate_rows_from_clean():
    df = pd.read_csv(CLEAN, low_memory=False)
    df = df[df.get("agent_model", "").eq(PHASE1_AGENT)].copy()
    df["phase1_method"] = df.apply(infer_method, axis=1)
    df = df[df["phase1_method"].isin(PHASE1_METHODS)].copy()
    keep = [
        "project", "run_id", "run_name", "dataset", "seed", "runtime_mins",
        "agent_model", "method", "phase1_method", "ppo_used", "epochs", "rollouts",
        "batch_size", "prompt_per_example", "train_size", "test_size",
    ]
    return df[[c for c in keep if c in df.columns]].drop_duplicates("run_id")


def safe_get_run(api, project, run_id, retries=3):
    path = f"{ENTITY}/{project}/{run_id}"
    for attempt in range(1, retries + 1):
        try:
            return api.run(path)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[retry {attempt}/{retries}] {path}: {exc}; sleeping {wait}s")
            time.sleep(wait)
    return None


def gpu_from_run(run):
    sources = []
    try:
        sources.append(("tags", ",".join(run.tags or [])))
    except Exception:
        pass
    try:
        sources.append(("config", json.dumps(dict(run.config), default=str)))
    except Exception:
        pass
    try:
        sources.append(("summary", json.dumps(dict(run.summary), default=str)))
    except Exception:
        pass
    try:
        metadata = getattr(run, "metadata", None)
        sources.append(("metadata", json.dumps(metadata, default=str)))
    except Exception:
        pass
    for source, text in sources:
        gpu = normalize_gpu(text)
        if gpu:
            return gpu, source
    return "", ""


def main():
    base = candidate_rows_from_clean()
    api = wandb.Api()
    rows = []
    for _, row in base.iterrows():
        project = str(row["project"])
        run_id = str(row["run_id"])
        if project not in PROJECTS:
            pass
        run = safe_get_run(api, project, run_id)
        gpu, gpu_source = gpu_from_run(run) if run is not None else ("", "")
        out = row.to_dict()
        out["gpu"] = gpu
        out["gpu_source"] = gpu_source
        if run is not None:
            try:
                out["created_at"] = str(run.created_at)
                out["state"] = run.state
                out["runtime_mins_wandb"] = round(float(run.summary.get("_runtime", 0)) / 60.0, 2)
            except Exception:
                pass
        rows.append(out)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_RAW, index=False)

    usable = raw[raw["gpu"].isin(["A40", "A100"])].copy()
    usable["runtime_mins"] = pd.to_numeric(usable["runtime_mins"], errors="coerce")
    summary = (
        usable.dropna(subset=["runtime_mins"])
        .groupby(["gpu", "phase1_method"], as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            n_tasks=("dataset", "nunique"),
            mean_runtime_mins=("runtime_mins", "mean"),
            median_runtime_mins=("runtime_mins", "median"),
            total_runtime_mins=("runtime_mins", "sum"),
        )
    )
    for col in ["mean_runtime_mins", "median_runtime_mins", "total_runtime_mins"]:
        summary[col] = summary[col].round(2)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_RAW}")
    print(f"wrote {OUT_SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
