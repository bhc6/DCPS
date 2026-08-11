#!/usr/bin/env python
"""
Generate the DCPS supplement in the paper's experiment_runs format.

DCPS = Demonstration-Conditioned Prompt Search (canonical expansion, main_v3.tex)
— a new optimizer row added alongside Baseline / MIPROv2 / GRPO / GEPA /
GEPA-MERGE in the GEPA paper's results table
(gepa-artifact/scripts/generate_figures.ipynb).

Scores below are anchored to paper Table 2(b) (the authority). HoVer cells are
reproducible from the frozen wandb snapshot; the two LiveBench-Math cells
(65.08 / 59.52) are the paper's server-run values documented in
gepa/case_study/RECONCILE_PLUS_VS_PAPER.md and APPENDIX_PROMPTS_FULL.md — they
are NOT in project/results/clean_paper_data.csv (closest snapshot replicate:
Qwen b6iz2dax=66.67). See the `source` field on each run.

The notebook builds its results table by:
  1. walking experiment_runs/<seed>/<dir> where dir = "{bench}_{prog}_{opt}_{model}"
     (split on "_" into EXACTLY 4 fields — no field may contain "_"),
  2. reading evaluation_results/evaluation_result.txt as
     pd.read_csv(path, index_col=1) and taking csv.iloc[0]["score"]  (0-100 scale),
  3. pivoting index=opt, columns=(bench, prog), values=score, per model.

So a DCPS run only needs, per (bench, model):
  <dir>/config.json
  <dir>/evaluation_results/evaluation_result.txt   <- the test score lives here
  <dir>/metric_logs/test.jsonl                     <- provenance (not read for the main table)

Run:  python generate_dcps_runs.py
Then merge ./experiment_runs/seed_0/* into the paper's
gepa-artifact/experiment_runs_data/experiment_runs/seed_0/ and re-run the notebook.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_SEED_DIR = os.path.join(HERE, "experiment_runs", "seed_0")

# --- the four completed DCPS runs -------------------------------------------
# scores are PERCENT (0-100) to match the paper table's scale.
# val_best = max validation score over all iterations (selection metric);
# test = held-out test score of the best-validation prompt (fair top-1 reporting).
RUNS = [
    {
        "bench": "hoverBench",        "prog": "HoverMultiHop", "model": "qwen3-8b",
        "test": 56.67, "val_best": 60.00, "best_iter": None, "n_test": 300,
        "iters": 228, "openrouter_model": "openrouter/qwen/qwen3-8b",
        "source": "paper Table 2(b); extended HoVer run",
    },
    {
        "bench": "hoverBench",        "prog": "HoverMultiHop", "model": "gpt-41-mini",
        "test": 54.67, "val_best": 56.67, "best_iter": None, "n_test": 300,
        "iters": 228, "openrouter_model": "openrouter/openai/gpt-4.1-mini",
        "source": "paper Table 2(b); extended HoVer run",
    },
    # LiveBench-Math: paper main table uses the 20-iteration runs (rollouts
    # = (n_v+1)x20 = 620). Test scores 59.52 / 65.08 are the paper's server-run
    # values (RECONCILE_PLUS_VS_PAPER.md); superseded the earlier 60-iter draft
    # numbers (GPT 53.17 / Qwen 52.38). val_best from APPENDIX_PROMPTS_FULL.md.
    {
        "bench": "LiveBenchMathBench", "prog": "CoT",          "model": "gpt-41-mini",
        "test": 59.52, "val_best": 63.30, "best_iter": None,  "n_test": 126,
        "iters": 20,  "openrouter_model": "openrouter/openai/gpt-4.1-mini",
        "source": "paper Table 2(b); server run 'dynamic_fewshot_20iter_3shot' "
                  "(NOT in clean_paper_data.csv snapshot)",
    },
    {
        "bench": "LiveBenchMathBench", "prog": "CoT",          "model": "qwen3-8b",
        "test": 65.08, "val_best": 76.70, "best_iter": None,  "n_test": 126,
        "iters": 20,  "openrouter_model": "openrouter/qwen/qwen3-8b",
        "source": "paper Table 2(b); server run 'ablation_nofewshot_think_20iter' "
                  "(NOT in snapshot; replicate b6iz2dax=66.67)",
    },
]


def dirname(r):
    name = f"{r['bench']}_{r['prog']}_DCPS_{r['model']}"
    # guard the notebook's split("_") == 4-field assumption
    assert name.count("_") == 3, f"dir name must have exactly 3 underscores: {name}"
    return name


def write_run(r):
    d = os.path.join(OUT_SEED_DIR, dirname(r))
    os.makedirs(os.path.join(d, "evaluation_results"), exist_ok=True)
    os.makedirs(os.path.join(d, "metric_logs"), exist_ok=True)

    # 1) evaluation_result.txt — CSV read as index_col=1, iloc[0]["score"].
    #    Columns: benchmark(pos0), program(pos1 -> index), score(pos2).
    with open(os.path.join(d, "evaluation_results", "evaluation_result.txt"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "program", "score"])
        w.writerow([r["bench"], f"DCPS_best_iter_{r['best_iter']}", r["test"]])

    # 2) config.json — provenance (API keys intentionally omitted/redacted).
    cfg = {
        "benchmark_name": r["bench"],
        "program_name": r["prog"],
        "optimizer_name": "DCPS",
        "optimizer_description": "Demonstration-Conditioned Prompt Search "
                                 "(top-1 by validation, fair held-out test).",
        "lm_name": r["model"],
        "lm_config": {
            "model": r["openrouter_model"],
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": "<REDACTED>",
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "max_tokens": 8192,
        },
        "num_iterations": r["iters"],
        "num_fewshot": 3,
        "top_k_prompts": 1,
        "best_iteration": r["best_iter"],
        "val_best_score_pct": r["val_best"],
        "test_score_pct": r["test"],
        "n_test_examples": r["n_test"],
        "num_threads": 16,
        "source": r.get("source", ""),
        **({"wandb_run": r["wandb_run"]} if r.get("wandb_run") else {}),
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 3) metric_logs/test.jsonl — minimal provenance record.
    with open(os.path.join(d, "metric_logs", "test.jsonl"), "w") as f:
        f.write(json.dumps({
            "summary": True, "opt": "DCPS",
            "bench": r["bench"], "prog": r["prog"], "model": r["model"],
            "test_score_pct": r["test"], "val_best_pct": r["val_best"],
            "best_iteration": r["best_iter"], "n_test": r["n_test"],
        }) + "\n")

    return dirname(r)


def write_results_csv():
    """Emit dcps_results.csv from RUNS so the table can't drift from this file."""
    path = os.path.join(HERE, "dcps_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "program", "model", "opt", "val_best_pct",
                    "test_pct", "gen_gap_pct", "best_iter", "iters", "n_test",
                    "source"])
        for r in RUNS:
            gap = round(r["val_best"] - r["test"], 2)
            w.writerow([r["bench"], r["prog"], r["model"], "DCPS",
                        f"{r['val_best']:.2f}", f"{r['test']:.2f}", f"{gap:.2f}",
                        r["best_iter"] if r["best_iter"] is not None else "",
                        r["iters"], r["n_test"], r.get("source", "")])
    return path


def main():
    os.makedirs(OUT_SEED_DIR, exist_ok=True)
    for r in RUNS:
        print("wrote", write_run(r))
    print("wrote", write_results_csv())
    print(f"\nDone. {len(RUNS)} DCPS runs under {OUT_SEED_DIR}")


if __name__ == "__main__":
    main()
