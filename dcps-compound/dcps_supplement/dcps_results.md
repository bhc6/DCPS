# DCPS Results — extends the GEPA paper results table

**DCPS** (Demonstration-Conditioned Prompt Search) is a new optimizer row, to be
placed alongside `Baseline / MIPROv2 / GRPO / GEPA / GEPA-MERGE` in the paper's
main results table (built by `gepa-artifact/scripts/generate_figures.ipynb`,
pivot: rows = optimizer, columns = (benchmark, program), values = test score).

Scores are **percent (0–100)**, matching the paper's scale. Test score is the
held-out test accuracy of the **best-validation** prompt (fair top-1: selection
done on validation only, never on test).

## DCPS row (drop into the paper table)

### Model: qwen3-8b
| opt | (hoverBench, HoverMultiHop) | (LiveBenchMathBench, CoT) |
|-----|----------------------------:|--------------------------:|
| **DCPS** | **56.67** | **65.08** |

### Model: gpt-41-mini
| opt | (hoverBench, HoverMultiHop) | (LiveBenchMathBench, CoT) |
|-----|----------------------------:|--------------------------:|
| **DCPS** | **54.67** | **59.52** |

## Full detail (val selection + generalization gap)

| Benchmark        | Program       | Model        | Val (best) | Test  | Gen. gap | Best iter | Iters | N(test) |
|------------------|---------------|--------------|-----------:|------:|---------:|----------:|------:|--------:|
| hoverBench       | HoverMultiHop | qwen3-8b     | 60.00      | 56.67 | 3.33     | —         | 228   | 300     |
| hoverBench       | HoverMultiHop | gpt-41-mini  | 56.67      | 54.67 | 2.00     | —         | 228   | 300     |
| LiveBenchMathBench | CoT         | gpt-41-mini  | 63.30      | 59.52 | 3.78     | —         | 20    | 126     |
| LiveBenchMathBench | CoT         | qwen3-8b     | 76.70      | 65.08 | 11.62    | —         | 20    | 126     |

- **Val (best)** = max validation score over all iterations (the selection metric;
  carries selection bias, so it sits above test — this is the paper's "generalization gap").
- **Gen. gap** = Val(best) − Test.
- **Alignment note (Aug 2026):** LiveBench-Math test scores are the **paper
  Table 2(b)** values (65.08 / 59.52), from the 20-iteration server runs
  documented in `../case_study/RECONCILE_PLUS_VS_PAPER.md`. These supersede the
  earlier 60-iter draft numbers (52.38 / 53.17). The two LiveBench cells are
  **not** present in `project/results/clean_paper_data.csv` (closest snapshot
  replicate: Qwen `b6iz2dax`=66.67); HoVer cells are reproducible from the
  snapshot. Regenerate this file with `generate_dcps_runs.py` — do not hand-edit.

## CSV

See `dcps_results.csv` (same numbers, machine-readable).
