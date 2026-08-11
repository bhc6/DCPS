# GFB / StablePrompt naming crosswalk (paper ↔ CSV ↔ wandb)

The single-prompt (StablePrompt) audit uses three different naming systems. This
file is the verified mapping so every Table 2(a) number is traceable to source
runs. Authority: `analyze_paper_data.py:118-156` (`determine_method`) and the
wandb configs on entity `awesome-prompt`.

## Method-name mapping

| paper (`main_v3.tex`) | old draft (`prefill.md`) | CSV `method` | wandb projects |
|---|---|---|---|
| **StablePrompt-PPO** | StablePrompt-PPO | `StablePrompt-PPO`, `StablePrompt-PPO(cs=0/0.05/0.1)` | `GFB_II`, `GFB_QA`, `GFB_TC`, `GFB_BBH_TC`, `GFB_BBH_TG` |
| **StablePrompt-DCPS** | RAE (RL-Ablated Equivalent) | `AlgPrompt` | `mmlu`, `II`, `II_re`, `II_re_correct`, `algprompt_classification_*`, `bbh_tg_algprompt`, `BBH` |
| **DCPS-Compound** | FSPO (Fewshot-Search PO) | `FSPO` | `*-dynamic-fewshot` (aime/livebench/ifbench/hotpotqa/hover/pupa) |

## How the CSV classifier decides (not by project name)

`mmlu` and `II` wandb runs **also carry** a `trl_ppo_trainer_config` block, so
the project/config alone cannot tell PPO from the frozen control. The classifier
keys on whether PPO **actually updated**:
- StablePrompt codebase = has both `prompt_per_example` and `batch_size`.
- PPO ran (nonzero completed PPO steps) → `StablePrompt-PPO` (`cs=` is the
  controlled-sampling sub-type).
- Same codebase, PPO disabled → `AlgPrompt` = StablePrompt-DCPS.
- `*-dynamic-fewshot` project → `FSPO` = DCPS-Compound.
- Config fields missing → `Unknown` (excluded from paper aggregates).

## Task-family mapping (paper ↔ CSV `task_family`)

| paper family | CSV `task_family` / dataset | source |
|---|---|---|
| GLUE/SuperGLUE | GLUE (SST2, MRPC, MNLI, SNLI, RTE, QNLI) | `*_tc_GFB`, `algprompt_classification_*` |
| BBII-TC | subset of BBH-II (classification subsets) | `GFB_BBH_TC`, `bbh_tg_algprompt` |
| BBII-Gen | subset of BBH-II (generation subsets) | `GFB_BBH_TG` |
| II (24 subsets) | BBH-II / II | `GFB_II`, `II`, `II_re_correct` |
| MMLU (57 subj) | MMLU | `GFB_QA` (PPO), `mmlu` (DCPS) |

## Verification (gemma-1.1-7b-it, from `by_task_summary.csv`)

Reconstructing macro means confirms the mapping; residual gaps are aggregation
protocol (subset set, best-of-Top-5, seed handling), NOT mislabeling:

| family | DCPS recon | paper DCPS | PPO recon | paper PPO |
|---|---:|---:|---:|---:|
| GLUE | 78.97 (n=4) | 77.1 | **76.71** (n=12) | **76.7** |
| MMLU | 53.68 | 54.1 | 52.71 | 55.9 |
| BBH-II (TC+Gen lumped) | 59.67 | 57.2 / 60.7 | 48.29 | 57.6 / 63.0 |

GLUE-PPO 76.71 ≈ paper 76.7 is an essentially exact match. To reproduce the
paper's exact cells, apply the paper protocol (3 seeds, best test among Top-5 by
training reward, paper's subset lists) rather than the flat `by_task_summary`
macro — `analyze_paper_data.py` is the script that does this.

## Caveats for rebuttal use

- `cs=0/0.05/0.1` are controlled-sampling variants (a GFLOWPO-era knob). The
  paper reports StablePrompt-PPO as one method; if a reviewer asks, `cs=0` is the
  no-controlled-sampling PPO run closest to vanilla StablePrompt.
- `Unknown` rows (105) are runs with missing config fields — never fold them into
  a method's mean.
- `AlgPrompt` runs on `II`/`mmlu` often have no per-run `seed` recorded; the
  script aligns PPO vs DCPS by `dataset` when seed is absent (line 582-591).
