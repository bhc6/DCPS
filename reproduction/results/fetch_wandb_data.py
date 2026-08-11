"""Regenerate the frozen wandb snapshot (results/raw_wandb_data.csv).

This is PROVENANCE, not a runtime dependency. The public artifact ships the
already-frozen raw_wandb_data.csv; you only need this script to reproduce that
export from the original private wandb project. Requires `pip install wandb` and
`wandb login` (or WANDB_API_KEY in the environment).

Config via environment (with sensible defaults):
    WANDB_ENTITY   wandb entity that owns the runs   (default: awesome-prompt)
    DCPS_RESULTS_DIR   output directory              (default: this file's dir)
"""

import os
import time
import pandas as pd
from tqdm import tqdm

try:
    import wandb
except ImportError as e:  # optional dependency
    raise SystemExit(
        "wandb is not installed. It is only needed to REGENERATE the snapshot; "
        "the frozen results/raw_wandb_data.csv is shipped with the artifact.\n"
        "Install with: pip install wandb"
    ) from e

# ---------------------------------------------------------
# 1. 配置 WandB 参数 (entity/key from environment)
# ---------------------------------------------------------
ENTITY = os.environ.get("WANDB_ENTITY", "awesome-prompt")
PROJECTS = [
    "hotpotqa-dynamic-fewshot",
    "pupa-dynamic-fewshot-origin",
    "pupa-baseline-origin",
    "hover-dynamic-fewshot",
    "hover-baseline-test",
    "ifbench-baseline-test",
    "ifbench-dynamic-fewshot",
    "aime-math-baseline-litellm",
    "aime-math-baseline",
    "aime-math-litellm-agnostic-nb",
    "dynamic-fewshot-agnostic-nb",
    "dynamic-fewshot-agnostic",
    "GEPA",
    "livebench-math-dynamic-fewshot",
    "gepa-hotpotqa",
    "aime-math-dynamic-fewshot",
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
    "quickstart_playground",
    "tta_TC",
    "bbh_tg_algprompt",
    "BBH",
    "mmlu",
    "algprompt_classification_snli",
    "algprompt_classification_qnli",
    "algprompt_classification_mnli",
    "algprompt_classification_rte",
    "algprompt_classification_mrpc",
    "algprompt_classification_sst2"
]

# 如果你给论文最终用的实验打了特定的 tag (例如 'paper_final')，填在这里。留空则拉取所有 finished 的 runs
TAG_FILTER = ""

# Output directory: DCPS_RESULTS_DIR env override, else this file's own directory.
RESULTS_DIR = os.environ.get(
    "DCPS_RESULTS_DIR", os.path.dirname(os.path.abspath(__file__))
)
CSV_FILENAME = "raw_wandb_data.csv"
OUTPUT_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

os.makedirs(RESULTS_DIR, exist_ok=True)

def _flush(df_existing, new_rows, label):
    if not new_rows:
        return df_existing
    df_new = pd.DataFrame(new_rows)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"  [+] Saved {len(new_rows)} new runs after '{label}'. Total now: {len(df_final)}")
    return df_final


def _serialize_run(run, project_name):
    run_dict = {
        "project": project_name,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "runtime_mins": round(run.summary.get("_runtime", 0) / 60, 2),
        "tags": ",".join(run.tags) if run.tags else "",
    }
    for k, v in run.config.items():
        run_dict[f"config_{k}"] = v
    for k, v in run.summary.items():
        if not k.startswith("_"):
            run_dict[f"summary_{k}"] = v
    return run_dict


def _fetch_project_with_retry(api, path, max_retries=4):
    for attempt in range(1, max_retries + 1):
        try:
            runs = api.runs(path)
            return list(runs)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] api.runs({path}) failed: {e}; sleeping {wait}s")
            time.sleep(wait)
    return None


def _serialize_run_with_retry(run, project_name, max_retries=4):
    for attempt in range(1, max_retries + 1):
        try:
            return _serialize_run(run, project_name)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] run {run.id}: {e}; sleeping {wait}s")
            time.sleep(wait)
    return None


def fetch_runs_from_wandb():
    api = wandb.Api()

    # 断点续传：读取已保存的 runs
    if os.path.exists(OUTPUT_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
            existing_ids = set(df_existing['run_id'].tolist())
            print(f"Resuming: loaded {len(existing_ids)} existing runs from {OUTPUT_PATH}")
        except Exception:
            existing_ids = set()
            df_existing = pd.DataFrame()
    else:
        existing_ids = set()
        df_existing = pd.DataFrame()

    for project_name in PROJECTS:
        print(f"Fetching runs from project: {ENTITY}/{project_name} ...")
        path = f"{ENTITY}/{project_name}"
        run_list = _fetch_project_with_retry(api, path)
        if run_list is None:
            print(f"  [skip] {path}: max retries exhausted")
            continue

        new_rows = []
        for run in tqdm(run_list, desc=f"Processing {project_name}"):
            if run.state != "finished":
                continue
            if TAG_FILTER and TAG_FILTER not in run.tags:
                continue
            if run.id in existing_ids:
                continue
            row = _serialize_run_with_retry(run, project_name)
            if row is None:
                continue
            new_rows.append(row)
            existing_ids.add(run.id)

        # 每跑完一个项目就落盘一次，避免中途丢失全部进度
        df_existing = _flush(df_existing, new_rows, project_name)

    print(f"Done. Total runs in {OUTPUT_PATH}: {len(df_existing)}")

if __name__ == "__main__":
    fetch_runs_from_wandb()