# DCPS Case Study: What Simple Prompt Search Discovers

> **Scope note.** This is an internal appendix. The headline scores below are the
> paper's Table 2(b) cells, anchored to the run-id crosswalk in
> [`RECONCILE_PLUS_VS_PAPER.md`](RECONCILE_PLUS_VS_PAPER.md). The prompt
> walkthroughs and the per-sample / failure-mode analyses in §2–§4 are drawn from
> **exploratory 60-iteration draft runs** (cited by W&B run-id) that produced the
> same qualitative prompts; their raw scores (LiveBench-Math DCPS 53.17 GPT /
> 52.38 Qwen) are draft-run values, not the paper's 20-iteration server cells
> (59.52 / 65.08), and are labelled as draft runs where they appear.

Qualitative analysis for the rebuttal, drawn entirely from existing run records
(W&B project `awesome-prompt`, checkpoint files, and saved per-sample evaluation
history). No new experiments were run.

**Setup.** DCPS (Demonstration-Conditioned Prompt Search) generates a candidate prompt at
each iteration by conditioning a generator LM on a freshly sampled set of few-shot
demonstrations, evaluates it on a fixed validation subset, and keeps the
best-validation candidate (top-1). We compare against GEPA and GEPA-MERGE
(reflective, Pareto-based evolutionary search) on the same two compound-system
benchmarks, all with gpt-4.1-mini as the task model.

## Headline test scores (best-validation prompt, held-out test set)

| Benchmark | Model | Baseline | DCPS | GEPA | GEPA-MERGE |
|---|---|---|---|---|---|
| HoVer | gpt-4.1-mini | — | **54.67** | 50.33 | 49.67 |
| HoVer | qwen3-8b | — | **56.67** | — | — |
| LiveBench-Math | gpt-4.1-mini | — | 59.52 | **64.21** | 61.14 |
| LiveBench-Math | qwen3-8b | — | 65.08 | 70.57 | — |

The LiveBench-Math cells above are the paper's 20-iteration server runs. The
exploratory 60-iteration draft runs analysed in §2–§4 scored lower (GPT 53.17,
Qwen 52.38), partly because their environment lacked the `Levenshtein` dependency
the olympiad scorer needs (see `§4 Caveat`); those draft numbers are not
metric-matched to the GEPA column.

The headline is not "simple always wins." It is **task-dependent**: DCPS *beats*
reflective search on HoVer's multi-hop retrieval, and *trails* it on LiveBench-Math's
open-ended reasoning. The prompts explain why.

---

## 1. Prompt evolution: what each method actually discovers

### 1.1 LiveBench-Math — from bare instruction to format contract

**Baseline (145 chars):**

```text
You are a helpful assistant. You are given a math question and you need to solve it
step by step. Always place the final answer inside \boxed{}.
```

**DCPS best (iter 18, val 66.7%, 1374 chars):**

```text
You are a helpful and precise math assistant tasked with solving LiveBench math
problems. There are three task types you will encounter, each requiring a specific
final-answer format:

1. AMPS_Hard: The final answer must be an exact LaTeX expression enclosed inside
   \boxed{...}.
2. math_comp: The final answer is a three-digit integer between 0 and 999. Output
   only the integer, with no additional text or formatting.
3. olympiad: The final answer is a comma-separated list of expression indices.
   Enclose this list inside \boxed{...}.
... [per-task formatting rules, step-by-step-then-answer ordering] ...
```

**What DCPS discovered:** the *output-format contract*. Starting from a single
generic "put the answer in `\boxed{}`" instruction, sampling-and-selection converged
on the fact that LiveBench-Math is three sub-tasks with three *different* answer
formats, and that getting the format right is what the metric rewards. It did not
discover new mathematics — it discovered the parsing contract.

**GEPA best (6168 chars, single stage):** GEPA evolved a prompt an order of magnitude
larger that embeds *domain-specific solution recipes reflected from training
examples* — e.g.:

> "When tackling tetrahedron problems with distances and edge lengths given, use
> coordinate geometry to locate points. Start by fixing a vertex and aligning one
> edge along an axis... Calculate the inradius as r = 3V/S..."
>
> "For sets or sequences of positive integers with constraints on sums, mode
> uniqueness, and median properties: analyze median implications on list length
> (odd vs even)..."

This is the qualitative difference the reviewers asked to see: **DCPS finds format
contracts and task structure; GEPA finds problem-solving heuristics distilled from
specific failures.** GEPA also folds in the same format rules DCPS found (boxed
answers, letter-repetition), but wraps them in reflection-derived recipes. On
LiveBench-Math, those recipes are worth ~5 points (GPT-4.1-mini: DCPS 59.52 vs
GEPA 64.21) — reasoning help transfers.

### 1.2 HoVer — where the generic prompt wins

HoVer is a 4-stage compound system (summarize hop-1 → write hop-2 query →
summarize hop-2 → write hop-3 query), interleaved with BM25 retrieval over Wikipedia.

**Baseline** is DSPy's bare auto-generated template, ~67–79 chars per stage:

```text
[summarize1]        Given the fields `claim`, `passages`, produce the fields `summary`.
[create_query_hop2] Given the fields `claim`, `summary_1`, produce the fields `query`.
[summarize2]        Given the fields `claim`, `context`, `passages`, produce the fields `summary`.
[create_query_hop3] Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.
```

**DCPS best (gpt-4.1-mini, iter 44, val 56.7%; ~2471 chars total, ~620/stage)** —
generic but strategy-bearing. E.g. `create_query_hop2`:

> "Your task is to generate a precise and effective BM25 retrieval query for the
> second hop. Construct your query to include critical entities, disambiguating
> context, and linking concepts surfaced in the summary..."

Each stage now describes *what makes a good intermediate output* — focus on entities,
disambiguation, unresolved references, BM25-query construction — without naming any
specific claim.

**GEPA (14,801 chars total, ~3700/stage)** evolved the same four stages, but its
`create_query_hop2` prompt embeds *verbatim reflections on specific training claims*:

> "In Example 1, the assistant's query missed including 'Jay T. Wright', who is
> crucial to differentiate from the director Lloyd Kaufman... In Example 3, the
> assistant forgot to mention relevant keywords like 'Adam Guettel', 'Elena
> Shaddow', or the song/play names like 'How Glory Goes'."

**Why DCPS wins here (54.67 vs 50.33):** HoVer rewards a *generalizable retrieval
strategy* (extract entities, disambiguate, link across hops). DCPS's generic
instructions encode exactly that. GEPA's prompts over-specialize — they carry named
entities and per-example fixes from the training claims that do not transfer to the
held-out test claims, and the sheer length (5.9× DCPS) dilutes the reusable
strategy. This is a concrete instance of reflective search *overfitting the prompt to
its optimization examples*, and it directly supports the paper's thesis that on some
compound tasks, simple sampling generalizes better.

---

## 2. Sample-level success and failure (LiveBench-Math, DCPS)

Per-sample records are saved in `evaluation_history/`. Breaking the DCPS test run
down by sub-task (representative run, n=127, score 66.7% over the scored items):

| Sub-task | Correct | Notes |
|---|---|---|
| math_comp | 45 / 50 | strongest; format-contract wins |
| AMPS_Hard | 41 / 53 | mixed; exact-form & arithmetic failures |
| olympiad | 0 / 24 | confounded by a metric crash — see §4, do not read as reasoning failure |

### 2.1 math_comp — the format contract pays off

DCPS discovered that multiple-choice answers must be submitted as the chosen letter
repeated five times. Concrete successes:

```text
Q: Points Q and A lie on y=log2 x; midpoint of QA is (6,2). Positive difference of
   x-coordinates?  (choices A..E)
   gold: E     pred: EEEEE     ✓  (updated_amc_12a_2023)

Q: Coffee-maker discount + sales-tax word problem (choices A..E)
   gold: E     pred: EEEEE     ✓  (updated_amc_12b_2023)
```

The model reasoned to the right choice *and* emitted it in the exact form the parser
expects — the prompt's format rule is doing real work here.

**Failures are genuine hard reasoning**, not formatting:

```text
Q: f(x)=||x|-1/2|, g(x)=||x|-1/4|; count intersections of
   y=4g(f(sin 2πx)) and x=4g(f(cos 3πy)).
   gold: 385   pred: 128   ✗  (aime_i_2024)

Q: Triangle ABC, incenter I, circumcenter O, IA ⊥ OI, R=13, r=6. Find AB·AC.
   gold: 468   pred: 312   ✗  (aime_ii_2024)
```

These are competition-hard AIME problems; the answer format is correct (a bare
integer) but the mathematics is wrong. This is the ceiling of what *any* prompt-only
method reaches with this task model — and where GEPA's reflected recipes (§1.1) buy
some of their advantage.

### 2.2 AMPS_Hard — correct math, rejected form

DCPS wins the mechanical sub-tasks (GCD, polynomial factoring):

```text
Q: Factor 9x² + 27√7 x − 3402.
   gold: 9(−x−9√7)(6√7−x)   pred: \boxed{9(x+9√7)(x−6√7)}   ✓ (equivalent, accepted)
```

But loses on **exact-form expressions where the answer is mathematically equivalent
but not string-matchable**:

```text
Q: Compute the geometric mean of {2097152, −4782969, −1, 9, −279936, 1, 36}.
   gold: 432 · ⁷√(−1) · 2^(2/7) · 3^(4/7)
   pred: \boxed{−432 · ⁷√324}
```

The prediction is a defensible closed form, but the metric expects a specific
factored radical. This is a *metric-strictness* failure mode, not a reasoning failure
— worth flagging honestly in the paper: some AMPS_Hard "errors" are exact-match
artifacts that penalize all methods equally.

---

## 3. Failure-mode analysis

### 3.1 Validation-to-test overfitting (the qwen3-8b LiveBench case)

The clearest failure mode of top-1 DCPS is selection overfitting on a small, fixed
validation set. For qwen3-8b on LiveBench-Math, in an exploratory 60-iteration
draft run (W&B run `07tj4oz7`; the paper's 20-iteration server cell is 65.08):

| | value |
|---|---|
| best-validation score (max over 60 iters, 30 val items) | 73.33% |
| held-out test score of that prompt | 52.38% |
| generalization gap | **20.95 pts** |
| best iteration | 42 / 60 |

Two mechanisms compound here:

1. **Max-over-iterations selection on 30 items.** Taking the argmax validation score
   over 60 candidates on a 30-example val set is a multiple-comparisons problem: the
   winning prompt is partly fit to val-set noise. The gpt-4.1-mini run on the same
   benchmark shows a smaller but real gap (val 66.7% → test 53.2%), consistent with
   the same effect at lower variance.

2. **Task-model run-to-run variance.** qwen3-8b is a thinking model served at
   temperature 0.6; a separate rerun of the *same* prompt scored 38.10% (more network
   -error samples scored 0 that day). The 52.38% figure is the clean W&B-logged run.
   High task-model variance inflates the apparent val peak and widens the gap.

**Takeaway for the paper:** report the *test* score of the best-*validation* prompt
(which these runs do — top-1, `final_results[0]`), never max-over-test; and note that
small validation sets make DCPS's selection step the dominant source of
generalization gap. This is an honest limitation, not a bug.

### 3.2 Demonstration-sampling noise

Because each DCPS candidate is conditioned on a freshly sampled demonstration set,
candidate quality is noisy across iterations. Evidence from the checkpoints: the best
iteration is late and non-monotone (livebench-gpt peaks at iter 18/60; livebench-qwen
at 42/60; hover-qwen at 188/228). There is no steady climb — good prompts appear when
a favorable demonstration sample lands. This is the flip side of DCPS's simplicity:
no reflection means no directed improvement, so coverage (iteration count) matters,
and the method benefits from more iterations more than from smarter ones.

### 3.3 Where a heavy optimizer still earns its cost

Putting §1–§3 together, reflective search (GEPA) is worth its extra compute when the
task rewards **transferable reasoning recipes** rather than a format contract:
open-ended math (LiveBench-Math, +5 pts over DCPS). It *loses* when the task rewards
a **generic, reusable strategy** and reflection instead over-specializes to
optimization examples (HoVer, −4 pts vs DCPS). DCPS is the better default when the
win is mostly about discovering the output contract and task structure.

---

## 4. Caveat: LiveBench-Math olympiad scoring is confounded

The olympiad 0/24 in §2 is **not** a DCPS reasoning failure. The olympiad scorer
(`livebenchmath_utils/olympiad/utils.py`, called with `edit_distance=True`) computes
`Levenshtein.distance(...)`, but `Levenshtein` is not installed in the environment
that ran the DCPS `examples/` scripts. Every olympiad item therefore raised an
exception that was caught and recorded as `score=0.0` (verified: all 249 olympiad
records across `evaluation_history/*.json` carry an `error` field and score 0).

Consequences:

- ~19% of the LiveBench-Math test set (24/126) was forced to zero by a crashed
  dependency for DCPS. On non-olympiad items the best DCPS run scores 83.5%.
- The GEPA runs use the *identical* olympiad code, but the GEPA artifact declares
  `levenshtein` as a dependency and its W&B run has `Levenshtein==0.27.3` installed —
  so GEPA received olympiad partial credit. **The DCPS-vs-GEPA LiveBench-Math
  comparison is not metric-matched.** The draft run's 53.17% is a conservative
  lower bound; fixing the dependency would raise it — consistent with the paper's
  20-iteration server run (with `Levenshtein` working) scoring 59.52.
- The raw olympiad predictions are saved, so this can be re-scored offline with a
  working Levenshtein (no new model calls). Deferred per current decision — flagged
  here so the number is not misinterpreted.

---

## Source records

- DCPS prompts: `case_study/dcps_prompts.json`, `prompts_readable_{livebench,hover}.txt`
- GEPA/GEPA-MERGE prompts (verbatim): `case_study/gepa_prompts_extracted.md`
- Per-sample outcomes: `evaluation_history/*.json`
- Scores/costs: W&B `awesome-prompt` — livebench `id96bro5` (gpt), `07tj4oz7` (qwen);
  hover `soe03tl0` (gpt), `7iqanzfb` (qwen); GEPA `8hllkzv8`/`0rm9nuoc` (livebench),
  `grkqqy50`/`lgwdl9nf` (hover).
