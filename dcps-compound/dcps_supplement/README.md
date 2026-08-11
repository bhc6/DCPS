# DCPS Supplement Package

Adds the **DCPS** (Demonstration-Conditioned Prompt Search) method as a new row
in the GEPA paper's results table, across 2 benchmarks × 2 models (4 runs).

## Contents

| File | What it is |
|------|------------|
| `dcps_results.md`   | The DCPS row formatted for the paper table + full detail (val / test / gap). |
| `dcps_results.csv`  | Same numbers, machine-readable. |
| `generate_dcps_runs.py` | Generates `experiment_runs/seed_0/*` in the paper's exact format. |
| `experiment_runs/seed_0/<dir>/` | 4 run dirs the paper's notebook can read directly (created by the script). |

## How it plugs into the paper

`gepa-artifact/scripts/generate_figures.ipynb` builds the results table by:
1. walking `experiment_runs/<seed>/<dir>`, where `<dir>` = `{bench}_{prog}_{opt}_{model}`
   (split on `_` into **exactly 4 fields**);
2. reading `evaluation_results/evaluation_result.txt` via
   `pd.read_csv(path, index_col=1)` → `csv.iloc[0]["score"]` (percent, 0–100);
3. pivoting `index=opt, columns=(bench, prog), values=score`, per model.

The 4 generated dirs satisfy all three. Verified: loading them with the notebook's
exact parse reproduces the DCPS row in the pivot for both models.

## To merge into the paper artifact

```bash
python generate_dcps_runs.py            # writes ./experiment_runs/seed_0/*
cp -r experiment_runs/seed_0/* \
   ../gepa-artifact/experiment_runs_data/experiment_runs/seed_0/
# then re-run generate_figures.ipynb — DCPS appears as a new row.
```

To show DCPS in the figures, add `'DCPS'` to the `opts_order` lists and a color
in `color_map` (e.g. `'DCPS': '#8e44ad'`) in the notebook.

## Alignment & differences vs the paper's runs

- **Same**: benchmarks (HoVer, LiveBench-Math), programs (HoverMultiHop, CoT),
  models (qwen3-8b, gpt-4.1-mini), held-out test sets (300 / 126), metric
  (AMPS_Hard symbolic-equivalence judging identical to upstream).
- **Different**: inference via **OpenRouter** (not the paper's local Arbor/vLLM).
  This affects speed and introduces occasional `SSL: UNEXPECTED_EOF` failures
  (scored 0). qwen3-8b runs in thinking mode (cannot be disabled on the
  OpenRouter endpoint) and at temperature 0.6 shows real run-to-run variance.
- **Fair reporting**: `top_k=1`; the reported test score is that of the
  best-**validation** prompt, never `max(test_score)` over a top-k set.

## Reproduce

From repo root (`c:/Users/123/Desktop/gepa`):
```bash
# hover (228 iters)            -> examples/hover/dynamic_fewshot*.py
uv run python -m examples.livebench_math.dynamic_fewshot          --num-iterations 60   # qwen3-8b
uv run python -m examples.livebench_math.dynamic_fewshot_gpt41mini --num-iterations 60  # gpt-4.1-mini
```
Runs checkpoint each iteration and resume on re-run. A timeout-deadlock in the
AMPS_Hard metric (sympy.simplify hanging on nested radicals) was fixed by moving
the timeout to a killable multiprocessing worker — see the metric file's
`run_with_timeout`.

> Security note: the paper artifact's existing `config.json` files contain a
> plaintext OpenRouter API key committed to git. Rotate that key. DCPS configs
> here use `<REDACTED>`.
