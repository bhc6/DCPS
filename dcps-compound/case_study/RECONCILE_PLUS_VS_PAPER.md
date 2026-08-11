# Reconciliation: CASE_STUDYplus / CASE_STUDY_PROMPTS vs paper Table 2(b)

**Verdict of the run-id audit.** The `*plus` case studies draw from the **same
FSPO=DCPS experiment family** as the paper (all cited run-ids live in the
`*-dynamic-fewshot` / GEPA / PUPA wandb projects and appear in
`project/results/clean_paper_data.csv`). They are **not** a different server's
experiments. But they (a) pick **non-main-table run configurations** as if they
were the headline result, (b) contain at least one **mislabeled score** that
produced a false claim, and (c) use an **inconsistent rollout convention**.
Therefore: keep their verbatim prompts and mechanism/qualitative observations,
**discard their score table and rollout numbers**, and re-anchor everything to
the run-ids below.

Authority = `clean_paper_data.csv` (run-id → dataset/model/method/score) +
paper Table 2(b). Rollout convention = paper's `(n_v+1)×num_iterations`.

## The mislabel that must not reach the rebuttal

CASE_STUDYplus §1.1/§2 claims **"GEPA on Papillon = 77.37, below baseline."**
There is **no GEPA-PUPA run scoring 77.37**. The real runs are:

| PUPA (GPT-4.1-mini) | score | run-id | method (CSV) |
|---|---:|---|---|
| Baseline | 80.81 | `c7evheyb` | Baseline |
| Baseline (variant) | **77.37** | `kd87hy29` | **Baseline — NOT GEPA** |
| **GEPA** | **90.10** | `r52c9gta` | GEPA |
| GEPA-MERGE | 93.52 | `y1lojof8` | GEPA-MERGE |
| MIPROv2-Heavy | 83.59 | `xj0qjc4s` | MIPROv2-Heavy |
| **DCPS (FSPO)** | **94.09** | `p16uyuga` | FSPO |

`77.37` is a **baseline** run (`kd87hy29`) that the plus doc misread as GEPA. The
"GEPA overfits below baseline on Papillon" narrative is built on this error and
**must be deleted**. Correct story: DCPS 94.09 > GEPA-MERGE 93.52 > GEPA 90.10 >
MIPRO 83.59 > baseline 80.81 — DCPS still wins, but GEPA does *not* fall below
baseline.

## Main-table cell → authoritative run-id (GPT-4.1-mini)

| Task | metric | paper 2b | run-id | note / plus-doc error |
|---|---|---:|---|---|
| AIME-2025 | DCPS | 48.00 | `mwjwwnoh` | plus used 52.0 (`13xynbot`, val=45 extended) |
| | GEPA | 48.00* | `1ksi41t3`=50.0 | paper reports 48.00; `1ksi41t3` logs 50.0 (seed/eval diff — use paper) |
| HotpotQA | DCPS | 59.00 | `89wk61np` | plus used **63.67** (extended-budget run, not main table) |
| | baseline | 36.10 | `e63dl8v1`=35.33 | plus baseline 35.33 ≈ ok |
| IFBench | DCPS | 51.53 | `0tv7woty` | plus used "~49"; `w5dkndzl`=76.19 is val-contaminated, exclude |
| | GEPA | 49.83 | `wverd6or` | |
| | MIPRO | 51.19 | `wxz5mpbs` | |
| HoVer | DCPS | 54.67 | (extended, server) | plus said "needs extended budget / heavy wins" — WRONG, DCPS wins main table |
| | GEPA | 49.67–50.33 | `grkqqy50`/`lgwdl9nf` | small-budget DCPS `wqp8lojs`=45.67 is NOT the main cell |
| PUPA | DCPS | 94.09 | `p16uyuga` | ok |
| | GEPA | 90.10 | `r52c9gta` | plus's 77.37 is a mislabeled baseline |
| LiveBench | DCPS | 59.52 | (server, 20-iter) | plus used 31.75 (`prmdtdw1`, Qwen val=30 replicate) |
| | GEPA | 64.21 | `8hllkzv8` | |

\*AIME-GPT GEPA: paper prints 48.00; the CSV `1ksi41t3` logs 50.0. Use the paper
value and footnote the discrepancy; do not silently cite 50.0.

## Main-table cell → authoritative run-id (Qwen3-8B)

| Task | metric | paper 2b | run-id | note |
|---|---|---:|---|---|
| AIME | DCPS | 55.33 | (main-table run) | CSV max `8v2qjgrx`=60.0 is a different config |
| HotpotQA | DCPS | 61.33 | `trvbqp3c` | ✓ |
| HoVer | DCPS | 56.67 | (extended) | small `aa8u5puv`=44.0 is not the cell |
| PUPA | DCPS | 90.02 | `n7o1crf6` | ✓ |
| LiveBench | DCPS | 65.08 | (server run) | CSV `b6iz2dax`=66.67 is a replicate |
| | GEPA | 70.57 | `ag4xsrtm` | ✓ |

## Rollout convention (replace plus-doc numbers)

Plus docs use `val_size × iterations` and figures like AIME 450, HotpotQA 6300,
Papillon 900, GEPA 2426. **Paper convention** (verified, six-for-six):
`rollouts = (n_v+1) × num_iterations` → AIME 320, IFBench/LiveBench 620,
PUPA 920, HotpotQA 1020, HoVer 7068; macro 1761. GEPA-family = published defaults
(HotpotQA/MIPRO 6871, HoVer 7051, PUPA 2426, AIME/LiveBench 1839, IFBench 3593).
Use these; drop the plus-doc rollout column.

## What is SAFE to reuse from the plus docs

- ✅ **Verbatim optimized prompts** (`CASE_STUDY_PROMPTS.md` Part II): DCPS
  redaction/response prompts, HotpotQA 4-stage prompts with the Cranberries/Mike
  Hogan worked example, AIME CoT prompt, GEPA/MIPRO prompts — these are real
  artifacts. Re-label each with the correct run-id above.
- ✅ **DCPS meta-prompt templates** (`CASE_STUDY_PROMPTS.md` Part I): the AIME /
  Papillon / HotpotQA proposal signatures and selection rule — match the code in
  `examples/*/dynamic_fewshot*.py`; safe.
- ✅ **Mechanism observations**: DCPS discovers (role assignment, checklist
  decomposition, output-format pinning, one worked demo); GEPA hard-codes
  validation-set surface artifacts ("Android/Kotlin/ViewModel"). These are
  qualitative and hold — but state them WITHOUT the false 77.37 comparison.
- ✅ **Budget-sensitivity sweeps** (CASE_STUDYplus §5.4): LiveBench val=5/30/100
  and HotpotQA 20/40/120-iter runs are real CSV runs (`prmdtdw1`, `xji9z2aa`,
  `qy66amuq`, `gs527dox`, `4436w5h5`, `89wk61np`, `9y92co33`, `zzvvoi77`); they
  are legitimate sensitivity data (answers Sv3D-3, p9rm-Q1) — but relabel their
  rollout counts to the `(n_v+1)×iters` convention and keep them clearly as
  **Qwen val-size probes**, not GPT main-table cells.

## What must be DISCARDED / rewritten

- ❌ §1.1 and §2 **score/verdict tables** — wrong scores (63.67, 31.75, 77.37)
  and inverted win/loss (HoVer, IFBench). Rebuild from the run-id tables above.
- ❌ **All rollout numbers** in the plus docs (450/6300/900/2426).
- ❌ The **"GEPA below baseline on Papillon"** claim entirely.
- ❌ **Backbone mixing**: plus §2 blends Qwen PUPA (85.76 `tituojj7`) and GPT
  results in one column — separate by backbone.
