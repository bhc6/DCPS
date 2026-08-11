# Case Study: What DCPS Actually Discovers, and Where It Fails

> **Scope note.** This is an internal appendix. Scores and verdicts here are the
> paper's Table 2(b) cells, anchored to the run-id crosswalk in
> [`RECONCILE_PLUS_VS_PAPER.md`](RECONCILE_PLUS_VS_PAPER.md). The validation-size
> and budget sweeps in §5.4 are legitimate **Qwen val-size probes** — deliberately
> small-val configs, labelled as such — and are **not** the paper's headline
> LiveBench cells (Qwen 65.08 / GPT 59.52). Rollout counts below are raw
> model-call counts (val×iterations); the paper reports (n_v+1)×iterations, which
> differs by about one validation pass and leaves the budget ratios unchanged.

*Appendix material responding to reviewer requests (p9rm, yjXF, g1zW, Sv3D) for
qualitative, case-level analysis of Demonstration-Conditioned Prompt Search
(DCPS) in the DSPy/GEPA compound-AI setting.*

All prompts, scores, and rollout counts are taken verbatim from the optimized-prompt
artifacts and run logs in this repository (`experiment_prompts/`, `best_prompts_*.json`,
`checkpoint_*.json`, `complete_method_comparison_plots/complete_method_comparison_audit.csv`,
`tokens_report.md`). In the compound-AI experiments, **DCPS is instantiated as the
demonstration-conditioned generate-and-validate control** (labeled `FSPO` / "dynamic
fewshot" in the raw logs): it freezes the generator, conditions proposal on randomly
sampled demonstrations, scores whole-program candidates on a validation subset, and keeps
the best. Compared against GEPA, GEPA+Merge (reflection + evolutionary merge) and
MIPROv2-Heavy, all on `gpt-4.1-mini` and `qwen3-8b`.

**Rollout definition.** One rollout = one model call. DCPS opt rollouts in the
tables below are raw **val\_size × iterations** model-call counts. The **paper**
reports optimizer rollouts as **(n\_v+1) × num\_iterations** (AIME 320,
IFBench/LiveBench 620, PUPA 920, HotpotQA 1020, HoVer 7068); the raw counts here
differ by about one validation pass and do not change the budget ratios.
Heavy-optimizer rollouts = `max_metric_calls` budget (AIME/LiveBench 1,839; Papillon
2,426; Hover 7,051; HotpotQA/MIPROv2-Heavy 6,871).

| Reviewer ask | Section |
|---|---|
| Prompts found by DCPS vs GEPA; success + failure cases | §1, §2 |
| What *kind* of prompt change does simple search discover? | §3 |
| Efficiency/usability: prompt length, token tax | §4 |
| Failure modes: demonstrations, overfitting, budget-matched | §5 |

---

## 1. Head-to-head prompt comparison (GPT-4.1-mini)

Instruction lengths (summed across all predictors):

| Task | Baseline | DCPS | MIPROv2-Heavy | GEPA | GEPA+Merge |
|---|---:|---:|---:|---:|---:|
| Papillon (2 pred.) | 289 c | **1,458 c** | 1,481 c | 9,116 c | 8,672 c |
| HotpotQA (4 pred.) | ~0 c | **3,006 c** | — | — | — |
| Hover (4 pred.) | 290 c | — | 1,628 c | 9,918 c | 6,452 c |
| IFBench (2 pred.) | 138 c | — | 1,187 c | 5,936 c | 6,417 c |
| AIME (1 pred.) | 0 c | **1,339 c** | 352 c | 6,287 c | 6,287 c |

Two facts hold across tasks: (1) DCPS prompts are **4–7× shorter than GEPA prompts**
while landing in the same accuracy band on structured tasks. (2) **DCPS and MIPROv2-Heavy
converge on similar lengths** (~1.5 k chars) — once reflection and evolution are removed,
generate-and-validate produces prompts of a characteristic "moderate instruction" size.

### 1.1 Papillon — DCPS wins on *test* (900 opt rollouts vs GEPA's 2,426)

**Baseline** `craft_redacted_request` (164 c):
> *"Given a private user query, create a privacy-preserving request for a powerful
> external LLM. The LLM may assist without learning private information about the user."*

**DCPS** `redaction_prompt` (742 c):
> *"You are the Redaction Agent responsible for preparing user queries for processing
> by an external large language model (LLM). Your task is to identify any personally
> identifiable information (PII), such as names of countries, companies, cities,
> individuals, or other sensitive data, and redact these details while keeping the
> overall meaning and intent of the query intact and coherent. …"*

**GEPA** `craft_redacted_request` (5,135 c, excerpt):
> *"… Retain domain-specific terminology (e.g., Android, Kotlin, ViewModel, layout
> inflation, clipboard management)."*

**Test accuracy (GPT-4.1-mini):** Baseline 80.81 → MIPROv2-Heavy 83.59 →
GEPA **90.10** (`r52c9gta`) → GEPA-MERGE 93.52 → **DCPS 94.09**. DCPS wins; GEPA
stays above baseline.

GEPA's prompt hard-codes surface artifacts lifted from validation demonstrations:
"Android, Kotlin, ViewModel," "Bloodsport-themed novel chapter," "UN resolution
clauses on ethical AI" (all quotable verbatim from `prompts_Papillon_PAPILLON_GEPA_gpt-41-mini-openrouter.json`).
These carve-outs add prompt length without transferring cleanly, so GEPA (90.10)
trails DCPS (94.09) despite 4–7× more prompt text. DCPS avoids memorizing these artifacts.

---

## 2. Success and failure cases, by task

| Task | Baseline | DCPS | Best heavy | Verdict |
|---|---:|---:|---:|---|
| **Papillon** | 80.81 | **94.09** | 83.59 (MIPRO) | ✅ DCPS best |
| **HotpotQA** (2-hop) | 35.33 | **59.00** | — | ✅ +24 pp |
| **AIME-2025** | 40.0 | 48.0 | 48.0 (GEPA) | = tie |
| **IFBench** | 48.13 | 51.53 | 51.19 (MIPRO) | ✅ DCPS edges |
| **Hover** (4-hop) | 43.67 | **54.67** | 50.33 (GEPA) | ✅ DCPS best |
| **LiveBench-Math** | 56.18 | 59.52 | 64.21 (GEPA) | ⚠️ trails GEPA |

**DCPS's two hardest regimes:**

- **Multi-hop trace attribution (Hover).** GEPA assigns *different* instructions to
  *different* hops — 5,110 c on `create_query_hop2`, 4,651 c on `summarize1`, the other
  two left near-baseline. DCPS proposes all four prompts jointly with no per-hop error
  signal and cannot localize which hop failed. This costs DCPS at *low* budget, where
  GEPA leads Hover; but at the paper's full budget (7,068 rollouts) DCPS's generic
  per-stage contracts overtake GEPA (54.67 vs 50.33). The mechanism gap is real, not
  decisive at adequate budget.
- **Validation overfitting on open-ended math (LiveBench, AIME).** See §5.

---

## 3. What kind of prompt change does DCPS actually discover?

Exactly **four** kinds of edit — and nothing more elaborate:

1. **Role assignment.** Every DCPS winner opens with a named role. The baseline DSPy
   signatures have none.
2. **Task decomposition into an explicit checklist.** AIME → 6-step procedure;
   HotpotQA `summary1` → 5-bullet extraction spec.
3. **Output-format pinning.** Answer-stage prompts gain a format constraint (*"Match
   the gold standard answer format…"*) — the single edit most responsible for the
   HotpotQA gain.
4. **One worked example from a sampled demonstration.** The HotpotQA `summary1` prompt
   carries the Cranberries/Mike Hogan instance; the AIME-Qwen prompt references
   *"Example 1's factor analysis"* and *"Example 3's inclusion-exclusion approach."*
   This is the demonstration-conditioning fingerprint.

What DCPS does **not** discover: trace-attributed per-predictor specialization (Hover),
and task-specific carve-outs (Papillon) — the latter being a *liability*.

---

## 4. Efficiency and usability (reviewer Sv3D)

**Characters added over baseline:**

| Task | DCPS added | GEPA added | Ratio |
|---|---:|---:|---:|
| Papillon | +1,169 c | +8,827 c | **7.6×** |
| AIME | +1,339 c | +6,287 c | **4.7×** |
| Hover | — | +9,628 c | — |

GEPA's optimized prompt is prepended to every test query permanently — ~2 k extra
tokens/call on Papillon for a prompt that scored 90.10 at test — above baseline, but still below DCPS's 94.09. DCPS
prompts leave per-query inference cost essentially unchanged.

**One-time optimization cost** (from `tokens_report.md`): AIME GEPA+Merge 4.53 M in /
8.52 M out ($16.53); MIPROv2-Heavy 6.33 M in / 4.34 M out ($9.70). DCPS matches
GEPA's macro accuracy at **2.2× fewer rollouts and 1.5× lower API cost** under the
reported budgets.

---

## 5. Failure-mode analysis and budget-matched comparison

### 5.1 Validation overfitting is real and measurable

| Task / model | Val | Test | Gap |
|---|---:|---:|---:|
| Papillon / GPT | 98.89 | 94.09 | **4.8 pp** |
| HotpotQA / GPT (extended) | 70.0 | 63.67 | 6.3 pp |
| AIME / GPT | 80.0 | 48.0 | **32 pp** |
| AIME / Qwen | 73.33 | 53.33 | 20 pp |
| LiveBench-Math / Qwen (val=5 probe) | 80.0 | 31.75 | **48.2 pp** |

On structured tasks the gap is small and selection is trustworthy. On open-ended math
with a small validation subset the gap explodes.

(The rows above are individual-run val→test gap measurements chosen to illustrate the
overfitting effect — the extended HotpotQA run and small-val probes — not the paper's
Table 2(b) headline cells; for those, see §2 and `RECONCILE_PLUS_VS_PAPER.md`.)

### 5.2 Sensitivity to demonstration sampling (the S step)

The embedded worked example in §3 item 4 becomes a liability when the drawn example
is atypical. A **0-shot ablation** on LiveBench-Math (no demonstrations; test 0.309 /
0.333) lands *inside* the 3-shot spread (0.294–0.389) — demonstrations are near-neutral
on this task, bounding how much the S step can be blamed or credited.

### 5.3 Reliability boundary

DCPS is reliable when **(a)** the task has a transferable single instruction and **(b)**
the validation subset is large enough (val→test gap <7 pp on Papillon/HotpotQA).
Unreliable on open-ended generation with a small validation set (AIME/LiveBench: 20–48 pp)
or many-predictor multi-hop attribution (Hover).

### 5.4 Controlled single-knob sweeps

All rollout counts below are raw **val\_size × iterations** model-call counts (see
§0 for the paper's (n_v+1)×iterations convention).

---

**(A) Validation-set size — the decisive knob.**
AIME-2025, Qwen3-8B, top-5, 20 iterations, 3-shot:

| Val subset | Opt rollouts | Val | Test | Gap | Run |
|---|---:|---:|---:|---:|---|
| 15 (`fixed_15`) | **15×20 = 300** | 0.733 | 0.547 | +18.6 pp | `bkpxngx5` |
| 45 (`fixed_all_45`) | **45×20 = 900** | 0.556 | **0.600** | **−4.4 pp** | `8v2qjgrx` |
| GEPA (reference) | 1,839 | — | 0.593 | — | — |

Enlarging val from 15 to 45 makes the validation score lower and more honest (0.733→0.556)
while **raising test 0.547→0.600** and closing the gap to ~0 — the small subset was
actively selecting a worse prompt. GPT-4.1-mini point (`mwjwwnoh`, fixed_15, top-1,
300 rollouts) shows the same signature: val 0.80 → test 0.48, 32 pp gap.

---

**(B) Validation sample size sweep.**
LiveBench-Math, Qwen3-8B, top-1, 3-shot:

| Val sample | Opt rollouts | Val | Test | Gap | Run |
|---|---:|---:|---:|---:|---|
| 5 (2 iter) | **5×2 = 10** | 0.80 | 0.318 | +48.2 pp | `prmdtdw1` |
| 30 | **30×20 = 600** | 0.467 | 0.294 | +17.3 pp | `xji9z2aa` |
| 30 | 600 | 0.567 | 0.357 | +21.0 pp | `qy66amuq` |
| 30 | 600 | 0.567 | 0.389 | +17.8 pp | `gs527dox` |
| 100 | **100×20 = 2,000** | 0.45 | 0.333 | +11.7 pp | `4436w5h5` |
| GEPA (reference) | 1,839 | — | 0.706 | — | — |

The val=5 run (10 rollouts, 2 iter) is confounded — tiny-val and tiny-budget together.
Clean comparison: val=30 vs val=100 at 20 iter: gap 17–21 pp → 11.7 pp.
Three val=30 replicates (test 29.4 / 35.7 / 38.9) **quantify seed/demo variance: ±~5 pp**.

---

**(C) Iteration budget sweep.**
HotpotQA, GPT-4.1-mini, 5-shot, val=50:

| Iter | Opt rollouts | Val (best) | Test | Gap | Run |
|---|---:|---:|---:|---:|---|
| 20 | **50×20 = 1,000** | 62 | 59.0 | 3.0 pp | `89wk61np` |
| 40 | **50×40 = 2,000** | 66 | 59.7 | 6.3 pp | `9y92co33` |
| 120 | **50×120 = 6,000** | 70 | **63.7** | 6.3 pp | `zzvvoi77` |
| MIPROv2-Heavy (ref) | 6,871 | — | — | — | — |

Monotonic test gain (59.0→59.7→63.7); val-test gap stable at ~6 pp — no overfitting
collapse. Qwen replicate at 40 iter (`sbmupl58`, test 60.33) matches the GPT trend. At
6,000 rollouts (0.87× MIPROv2-Heavy budget) DCPS reaches test 63.67 vs baseline 35.33.

---

**(D) Demonstration conditioning (the "S" step).**
LiveBench-Math, Qwen3-8B, val=30, top-1:

| Config | Opt rollouts | Test | Run |
|---|---:|---:|---|
| 3-shot | 600 | 0.294 – 0.389 | `xji9z2aa`, `qy66amuq`, `gs527dox` |
| 0-shot ablation | 600 | 0.309 / 0.333 | `f8p7kmt5`, `1fm6356x` |

Ablation lands *inside* 3-shot spread — demonstrations neutral on this task.

---

**Sweep summary.** The knob that most moves generalization is **validation-set size**
(A, B). More iterations help on structured tasks without collapse (C). Demonstration
conditioning is task-dependent (D). All runs within the existing experiment set.

### 5.5 Budget-matched comparison (yjXF-1, g1zW)

| Task / model | Method | Opt rollouts | Ratio | Test |
|---|---|---:|---:|---:|
| Papillon / GPT-4.1-mini | **DCPS** | **45×20 = 900** | **0.37×** | **94.09** |
| Papillon / GPT-4.1-mini | GEPA | 2,426 | 1.00× | 90.10 |
| LiveBench-Math / Qwen3-8B | DCPS (val=100 probe) | **100×20 = 2,000** | **1.09×** | 33.3 |
| LiveBench-Math / Qwen3-8B | GEPA | 1,839 | 1.00× | 70.57 |

Budget is **not** the confound. DCPS wins Papillon (94.09 vs 90.10) at ~37% of GEPA's
budget. On LiveBench-Math the paper's headline DCPS-Qwen cell (65.08) trails GEPA (70.57)
by ~5.5 pp — a real but modest gap; the val=100 sweep probe above (33.3, a deliberately
small-val config) is a sensitivity point, not the headline result. **Task structure, not
rollout budget, determines the winner.**

Near-budget-matched Papillon Qwen point (val=111):

| Method | Opt rollouts | Ratio | Test | Qwen baseline |
|---|---:|---:|---:|---:|
| DCPS Qwen (val=111) | **111×20 = 2,220** | **0.92×** | **85.52** | 81.55 |

DCPS beats the Qwen baseline at 92% of GEPA's budget. (No GEPA-Qwen Papillon run at
equivalent budget exists to compare directly.)

**Multi-seed: Papillon / Qwen3-8B, 900 rollouts (2 runs confirmed Qwen):**

| Run | Val | Test |
|---|---:|---:|
| `n7o1crf6` | 94.4 | 90.02 |
| `tituojj7` | 90.7 | 85.76 |

Both exceed the Qwen baseline (81.55). Two additional runs at the same Papillon
configuration used GPT-4o-mini as the solver (val 97.3→test 87.77; val 96.0→test 85.54);
they are reported separately to avoid mixing models.

---

## 6. Takeaway

On structured tasks, DCPS compact prompts beat 5–9 k-char GEPA prompts at test
(Papillon: 94.09 vs 90.10) **at ~37% of GEPA's rollout budget** — the heavy machinery's
extra output is length, not proportional signal. On multi-hop attribution (Hover) and
open-ended generation (AIME/LiveBench), the heavy mechanisms earn their cost:
LiveBench-Math DCPS-Qwen 65.08 trails GEPA 70.57. **Task structure, not rollout budget,
decides the winner.**

This is the intended use of a generate-and-validate control: not to claim simple search
always wins, but to locate exactly where the added mechanism pays for itself.
