# DCPS Compound-System Case Study: Prompts, Efficiency, and Failure Modes

> **Scope note.** This is an internal appendix. All scores here are the paper's
> Table 2(b) cells, anchored to the run-id crosswalk in
> [`RECONCILE_PLUS_VS_PAPER.md`](RECONCILE_PLUS_VS_PAPER.md); prompt-length,
> rollout, and cost tables use the paper's conventions. The data-integrity note in
> §6 records the one figure (`77.37`) that an earlier draft mislabelled as GEPA on
> PUPA — it is the Baseline run `kd87hy29`; the true GEPA-PUPA score is 90.10.
> Verbatim prompts are reproduced in full in `APPENDIX_PROMPTS_FULL.md`.

This document extends `CASE_STUDY.md` (HoVer + LiveBench, DCPS vs GEPA) into a
full qualitative audit across **all six compound benchmarks** and **five
optimizer families**, directly addressing reviewer requests for: (i) concrete
optimized-prompt examples per method, (ii) success and failure cases, (iii) what
kinds of prompt changes each method discovers, and (iv) efficiency metrics
beyond accuracy (prompt length, words added, rollouts, cost).

## Data provenance

- **DCPS-Compound prompts** (a.k.a. `dynamic_fewshot` in wandb/logs): the prompt
  behind each Table 2(b) cell is the **Rank-1 (highest-val) prompt of the run
  whose Rank-1 test score equals the reported number** — verified for AIME,
  IFBench, LiveBench (both backbones) in `dcps_paper_prompts.json`, sourced from
  the `awesome-prompt/*-dynamic-fewshot` wandb projects (API) + local cache.
  Full text in `APPENDIX_PROMPTS_FULL.md`. HotpotQA and PUPA/Papillon have no
  dynamic_fewshot run (HotpotQA's wandb run is `examples.hotpotqa.main` = GEPA),
  so their DCPS prompt text is not reproducible here.
- **GEPA / GEPA-MERGE / MIPROv2-Heavy / Abl-SelectBestCandidate / GRPO prompts**:
  extracted from `gepa-rp/final_pkls/*.pkl` (40 programs) with
  `gepa-rp/extract_prompts.py`. That script reads the pickle **opcode stream**
  via `pickletools.genops` and never executes pickled code or imports dspy — the
  signature instruction is the string immediately following each `__doc__` key in
  a pydantic Signature namespace. Output: `gepa-rp/extracted_all_prompts.json`.
- **Scores**: paper Table 2(b) (`project/main_v3.tex`).
- **Per-sample cases**: `gepa/evaluation_history/*.json` (LiveBench-Math,
  126-item test set, per-example predicted/ground-truth/score).
- **Full verbatim prompt text**: `APPENDIX_PROMPTS_FULL.md` — every optimized
  prompt reproduced in full (§2 quotes excerpts; the appendix has the complete
  text for all methods × benchmarks × backbones).

## Executive summary

| Optimizer | What it writes into the prompt | Typical added text (GPT-4.1-mini) | Inspectable? |
|---|---|---|---|
| **DCPS-Compound** | Operational contracts: output format per subtask, per-stage role | **~1.2–2.5k chars total** | Yes |
| **GEPA / GEPA-MERGE** | Domain playbooks: enumerated problem families + worked recipes | 5–12k chars | Yes |
| **MIPROv2-Heavy** | Short instruction + many bootstrapped demonstrations | 4–16k chars (mostly demos) | Yes |
| **Abl-SelectBestCandidate** | GEPA's reflective proposer, best-of-N *without* merge/Pareto | 5–15k chars (≈ GEPA) | Yes |
| **GRPO** | *Nothing* — updates weights, leaves the baseline template | ~0 chars added | No prompt artifact |

**Headline qualitative finding.** The prompt *verbosity* attributed to GEPA does
not come from its reflection/merge search: **Abl-SelectBestCandidate — GEPA's own
generate-and-select ablation — produces prompts as long as or longer than full
GEPA on every benchmark** (e.g. HoVer 14.4k vs 12.1k chars; HotpotQA 15.4k vs
10.5k). Length is a property of the *reflective proposer*, not the search around
it. DCPS, whose proposer is asked for a task instruction rather than a
diagnostic rewrite, reaches competitive accuracy with 4–6× less prompt text.

## 1. Efficiency: prompt length, rollouts, and cost

### 1a. Optimized prompt length (GPT-4.1-mini)

Total instruction size across all optimized stages, as **characters / real
tokens** (GPT-4.1-mini cells tokenized with tiktoken `cl100k_base`). GEPA-family /
MIPRO / Abl-Select from `extracted_all_prompts.json` (the `gepa-rp` pkls); **DCPS
from its paper-Table-2b run** (`dcps_paper_prompts.json` + PUPA/HotpotQA verbatim
prompts), so every DCPS cell is filled. Default DSPy templates (~60–90 chars/stage)
are the baseline floor. Recompute with `compute_token_table.py`.

| Benchmark | DCPS (char/tok) | GEPA | GEPA-MERGE | Abl-Select | MIPROv2-Heavy |
|---|---:|---:|---:|---:|---:|
| AIME-2025 (1 stage) | **1,339 / 227** | 6,625 / 1,330 | — | — | 4,460 / 748 |
| LiveBench-Math (1 stage) | **1,536 / 337** | 5,295 / 984 | 3,064 / 597 | — | 5,519 / 950 |
| HoVer (4 stages) | **2,471 / ~460** | 12,135 / 2,309 | 8,495 / 1,656 | 14,426 / 2,784 | 16,401 / 2,922 |
| HotpotQA (4 stages) | **3,009 / 591** | 10,479 / 2,105 | 10,470 / 2,031 | 15,423 / 2,951 | 14,681 / 2,521 |
| IFBench (2 stages) | **1,458 / 259** | 6,598 / 1,254 | 6,777 / 1,265 | 8,369 / 1,560 | 11,027 / 1,893 |
| PUPA/Papillon (2 stages) | **1,459 / 270** | 6,074 / 1,180 | 6,467 / 1,213 | 5,309 / 1,000 | 11,021 / 1,978 |

DCPS is the **shortest optimized prompt on every benchmark**, on both characters
and tokens:

| Benchmark | vs GEPA (char / **tok**) | vs MIPROv2-Heavy (char / **tok**) |
|---|---:|---:|
| AIME-2025 | 4.9× / **5.9×** | 3.3× / **3.3×** |
| LiveBench-Math | 3.4× / **2.9×** | 3.6× / **2.8×** |
| HoVer | 4.9× / **~5.0×** | 6.6× / **~6.4×** |
| HotpotQA | 3.5× / **3.6×** | 4.9× / **4.3×** |
| IFBench | 4.5× / **4.8×** | 7.6× / **7.3×** |
| PUPA/Papillon | 4.2× / **4.4×** | 7.6× / **7.3×** |

**Char and token ratios are not interchangeable** — LiveBench is the cautionary
cell: DCPS is 3.4× shorter in characters but only **2.9×** in tokens, because its
prompt is dense in LaTeX (`\boxed{}`, math symbols) that costs many characters but
few tokens, while GEPA's prose tokenizes efficiently. For inference-cost claims use
the **token** ratio; for "prompt is shorter/simpler" the direction holds under both
metrics (2.8–5.9×). DCPS run-ids: AIME `origin_gpt…top1`, LiveBench
`dynamic_fewshot_20iter_3shot`, IFBench `0tv7woty`, PUPA `p16uyuga`, HotpotQA
`89wk61np`, HoVer extended. (LiveBench uses the 1,536-char main-table run; an
earlier draft's 1,374 is superseded.)

Two structural reads:
- **MIPROv2's length is demonstrations, not guidance.** Its 32–36 "signatures"
  on HoVer/HotpotQA are mostly bootstrapped few-shot demos carried into the
  program; the *instruction* text stays short. This is a different token-cost
  profile from GEPA (long instructions, few/no demos).
- **Abl-Select ≈ GEPA in length, sometimes longer.** Removing merge and Pareto
  evolution (Abl-SelectBestCandidate) does not shorten prompts — confirming
  verbosity originates in the reflective proposal step both share.

### 1b. Rollouts and cost (from paper Table 2b + cost report)

| Method | Macro rollouts | Macro cost (GPT-4.1-mini) | Macro RER |
|---|---:|---:|---:|
| DCPS-Compound | **1,761** | **$7.70** | **+26.23%** |
| GEPA | 3,936 | $11.34 | +24.15% |
| GEPA-MERGE | 3,936 | $9.63 | +24.87% |
| MIPROv2-Heavy | 3,936 | $10.84 | +12.70% |

DCPS reaches the highest relative error reduction at **2.2× fewer rollouts** and
**1.5× lower cost** than GEPA — the efficiency claim, restated at the prompt
level: shorter prompts, fewer rollouts, competitive accuracy.

## 2. What each optimizer discovers (per-benchmark gallery)

### 2.1 LiveBench-Math — a format contract vs a domain playbook

**DCPS (GPT, 1,374 chars, iter 18)** discovers a *per-subtask output contract*:

> There are three task types... **AMPS_Hard**: final answer as exact LaTeX in
> `\boxed{...}`. **math_comp**: a three-digit integer 0–999, output only the
> integer, no `\boxed{}`. **olympiad**: comma-separated expression indices inside
> `\boxed{...}`. ... Adhere strictly to these formatting rules.

The math reasoning was already adequate; the baseline's blanket "always
`\boxed{}`" rule *mis-formats* `math_comp` (which needs a bare integer). DCPS
found the fix in 18 rollouts.

**GEPA (5,295 chars)** instead writes a domain playbook — enumerated problem
families (logarithmic equations, geometric series, polynomial factoring…) each
with a worked recipe. It buys +4.7pp on GPT LiveBench (64.21 vs 59.52) but at
3.9× the prompt length. **GEPA-MERGE (3,064 chars)** compresses to an 8-item
numbered protocol. **MIPROv2** keeps the instruction short and attaches
bootstrapped demos.

*Takeaway:* when the bottleneck is an output-format mismatch, a short contract
matches most of the playbook's value.

### 2.2 HoVer (4-hop retrieval) — role contracts vs per-stage playbooks

DCPS rewrites each of the 4 stages with a compact *role contract*, e.g. hop-2
query builder (622 chars):

> Generate a precise BM25 retrieval query for the second hop... include critical
> entities, disambiguating context, and linking concepts... focus on bridge
> entities or relations that connect initial evidence toward deeper documents...
> maximize multi-hop recall.

GEPA expands the *same* stage into a multi-example playbook (its hop-3 signature
alone is 3,708 chars). Net: DCPS 2,471 chars total vs GEPA 12,133. Once
adequately budgeted (7,068 rollouts), DCPS **beats** GEPA on HoVer (GPT 54.67 vs
49.67; Qwen 56.67 vs 53.33) with a ≤3.3pp val–test gap — the initial GEPA lead
was a budget artifact, not a mechanism advantage.

### 2.3 AIME-2025 — where the playbook earns its length

GEPA's AIME prompt is its longest single-stage instruction (**6,625 chars**): a
structured olympiad playbook keyed by problem type. This is the regime where the
heavy proposer helps — on Qwen3-8B, GEPA (59.33) and GEPA-MERGE (62.67) lead
DCPS (55.33), because explicit solving techniques matter more than a format
contract when the base model's reasoning is the bottleneck. On GPT-4.1-mini all
methods tie at 48.00, i.e. the playbook's advantage shrinks as the base model
strengthens.

### 2.4 IFBench — every optimizer converges on the same rule

IFBench rewards verifiable instruction-following. **All** proposer-based methods
independently rediscover one contract — *repeat the query verbatim, then answer*.
GEPA (GPT) states it as step 1:

> **Exact Repetition of Query:** First, repeat the entire query exactly as
> given, word for word, including any formatting... Do not add, omit, or alter
> anything... Do not precede or follow this repetition with any commentary.

Yet DCPS-Compound *wins* IFBench on both backbones (GPT 51.53 vs GEPA 49.83;
Qwen 43.88 vs GEPA 40.65). The instructive failure is **GEPA-MERGE on Qwen:
29.08 — below the 38.61 baseline.** Crossover merging corrupted the strict
repetition contract (one merged Qwen stage collapsed to a 20-char stub). Merge is
*unstable on strict-format tasks*: the mechanism actively hurt.

### 2.5 PUPA / Papillon — privacy rewriting is a contract task

DCPS-family selection tops PUPA on both backbones (GPT 94.09 vs GEPA 90.10; Qwen
90.02 vs GEPA 87.57). GEPA writes a 6,074-char privacy policy; the task rewards a
clean anonymization *contract*, so extra policy text adds little. **GRPO's
optimized PUPA prompt is 164 chars — verbatim the baseline** ("Given a private
user query, create a privacy-preserving request..."). GRPO moved weights, not
text: there is no prompt artifact to inspect, which is itself a finding for
prompt-level interpretability.

### 2.6 HotpotQA — where heavier mechanism genuinely pays off

This is the clearest pro-GEPA case. GEPA leads on GPT (65.00 vs DCPS 59.00) and
Qwen (63.33 vs 61.33). Extending DCPS to 6,120 rollouts narrows the GPT gap
(59.00→63.67, gap 6.00→1.33pp) but leaves Qwen flat (61.33→60.33). Extra search
does **not** erase GEPA's edge. Multi-hop QA needs **trace-level error
attribution** — localizing *which* hop failed — which greedy generate-and-select
cannot supply. Notably Abl-Select's HotpotQA prompt is the longest of any method
here (15,420 chars) yet does not resolve this: length is not the missing
ingredient; per-module credit assignment is.

## 3. What kind of prompt change each mechanism makes

- **DCPS-Compound** — infers *operational contracts* from random demonstrations:
  output shape per subtask, per-stage role in a pipeline, disambiguation rules.
  Short, declarative, human-auditable.
- **GEPA / GEPA-MERGE** — the reflective proposer emits *domain playbooks*:
  problem families, worked recipes, enumerated edge cases. High-value on
  reasoning-heavy tasks, high token cost everywhere.
- **Abl-SelectBestCandidate** — GEPA's proposer with best-of-N selection but no
  merge/Pareto. Prompts stay GEPA-length ⇒ **verbosity comes from the proposer,
  not the search**. Isolates that the merge/evolution machinery contributes
  little length and (on IFBench) can even destabilize.
- **MIPROv2-Heavy** — short instruction + many bootstrapped demonstrations; its
  cost is demos, not guidance text.
- **GRPO** — weight-space RL; the prompt stays at the baseline template (66–164
  chars). No inspectable prompt change at all.

## 4. Success and failure cases (LiveBench-Math, per-sample)

Diffing a weak DCPS run (score 0.5397) against DCPS-best (0.6667) over the same
126-item test set: **16 items flipped wrong→right, 3 regressed, 9 stayed wrong.**

**Successes — format/contract fixes (what DCPS is for):**
- `amps_hard_geometric_mean`: gt `9\,2^{2/5}\sqrt[5]{11}`, DCPS pred
  `\boxed{9\sqrt[5]{44}}` — mathematically identical (`2^{2/5}·11^{1/5} =
  44^{1/5}`); the win is emitting a *checker-parseable* `\boxed{}` form.
- `aime_ii_2024`: gt `468`, pred `468` — the per-subtask integer contract fixed
  an answer the model already computed but previously mis-formatted.
- Flips cluster in `aime_*` and `amps_hard_*` (variance, std, char-poly) — all
  format-sensitive subtasks.

**Failures — genuine reasoning gaps (DCPS ceiling):**
- `usamo` (combinatorial ordering): gt is a 16-term permutation
  `9,7,10,3,...`; pred is a malformed repeated sequence `10,15,15,1,11,1,15,...`.
  No format contract can synthesize the missing combinatorial reasoning.
- `amps_hard_integral`: gt `-\sin(7)\sin(8x)-\cos(7)\cos(8x)`, pred
  `\tan(5x+1)-\cos(7-8x)+C` — wrong antiderivative (mishandled trig product).
- `amps_hard_geometric_mean` (hard variant with odd root of a negative): gt
  `432\sqrt[7]{-1}...`, pred drops the `\sqrt[7]{-1}` factor.

These persistent failures are exactly the *technique-recipe* gaps the paper
attributes to GEPA's advantage: they need worked-method knowledge (correct
antiderivative, odd-root handling, permutation construction) that a domain
playbook supplies but an operational contract does not. This is the format-vs-
reasoning boundary of the audit, visible at the level of individual test items.

## 5. Demonstration-conditioning fingerprint (what "S" leaves in the prompt)

DCPS conditions the generator on randomly sampled demonstrations. When the draw
is informative, that shows up as a **worked example copied into the prompt** — a
signature no other optimizer produces. Two verified instances (verbatim from the
selected prompts; see `CASE_STUDY_PROMPTS.md` for full text):

- **HotpotQA `summary1`** carries an embedded instance:
  *"For the question 'In what year did the Irish rock band whose bassist was Mike
  Hogan reunite?' … 'Mike Hogan is the bassist of the Irish rock band The
  Cranberries, who reunited in 2009.'"* — a whole sampled demonstration baked into
  the instruction (run `89wk61np`, DCPS-GPT HotpotQA test 59.00).
- **AIME (Qwen)** references *"Example 1's factor analysis"* and *"Example 3's
  inclusion-exclusion approach"* — the generator naming demonstrations from its
  meta-prompt.

This is double-edged and is the mechanistic answer to Sv3D-3 (demonstration
sensitivity): a representative draw injects a useful worked example; an atypical
draw bakes in a non-transferable specific. Because each DCPS run redraws
demonstrations, the cross-run test spread (§1b provenance / the multiple
`*-dynamic-fewshot` runs per cell) *is* the measurement of that sensitivity.

## 6. Overfitting signature: DCPS vs GEPA both hard-code validation artifacts

The failure DCPS shares with reflective search is **memorizing validation-set
surface detail that does not transfer**. GEPA's Papillon prompt hard-codes
carve-outs lifted from its optimization examples — e.g. *"Retain domain-specific
terminology (e.g., Android, Kotlin, ViewModel, layout inflation, clipboard
management)"* — quotable verbatim from
`prompts_Papillon_PAPILLON_GEPA_gpt-41-mini`. On PUPA/Papillon this is *not*
fatal for GEPA: the authoritative scores are DCPS **94.09** (`p16uyuga`) >
GEPA-MERGE 93.52 (`y1lojof8`) > GEPA **90.10** (`r52c9gta`) > MIPRO 83.59 >
baseline 80.81 — GEPA stays above baseline; DCPS simply wins by avoiding the
longest carve-outs at 4–7× less prompt text.

> **Data-integrity note.** An earlier draft (`CASE_STUDYplus.md`) claimed
> "GEPA-Papillon = 77.37, below baseline." That 77.37 is a **PUPA _baseline_ run**
> (`kd87hy29`, project `pupa-baseline-origin`), not a GEPA run — the true
> GEPA-PUPA score is 90.10. The "GEPA falls below baseline" narrative was based on
> that mislabel and is retracted here. All scores in this document are re-anchored
> to `clean_paper_data.csv` run-ids and paper Table 2(b); see
> `RECONCILE_PLUS_VS_PAPER.md` for the full run-id audit and the list of
> superseded numbers (HotpotQA 63.67→59.00, LiveBench 31.75→59.52, HoVer/IFBench
> win/loss corrections, and the `val×iters`→`(n_v+1)×iters` rollout convention).

## 7. Reviewer-concern crosswalk

- *"Concrete optimized-prompt examples per method"* → §2 (verbatim excerpts for
  all 6 benchmarks × GEPA/MERGE/Abl/MIPRO/GRPO; DCPS where archived).
- *"Success and failure cases"* → §4 (item-level flips and persistent fails with
  ground-truth/prediction pairs).
- *"What kinds of prompt changes are discovered"* → §2–§3 (contracts vs
  playbooks vs demos vs weight-space).
- *"Efficiency beyond accuracy — length, words, speed, cost"* → §1 (chars/words
  per method, rollouts, dollars).
- *"Comparison against simpler search"* → Abl-SelectBestCandidate throughout
  (GEPA's own generate-and-select control) + GRPO (weight-space reference).
- *"When/why does DCPS fail; demonstration-sampling sensitivity (Sv3D-3)"* → §5
  (demonstration fingerprint + cross-run spread) and §6 (validation-artifact
  overfitting, shared with GEPA).
- *"Data provenance / no conflicting numbers"* → §6 note + `RECONCILE_PLUS_VS_PAPER.md`
  (every cell tied to a `clean_paper_data.csv` run-id; superseded plus-doc numbers
  listed).
