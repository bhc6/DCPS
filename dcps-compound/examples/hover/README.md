# HoVer — Artifact-Aligned 4-Stage Multi-Hop Retrieval

Strict replication of the HoVer experiment from the GEPA paper
(*GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*,
Agrawal et al., 2025), aligned to `gepa-artifact` commit
`cbefbc1aa0f43dd39874ec4bf42211365dbda42e`.

This is the **multi-instruction** counterpart to `examples/ifbench/`:
HoVer's `HoverMultiHop` program has **four** LLM-driven predictors and
GEPA's optimisation surface jointly optimises all four instructions.

## Task

[HoVer](https://hover-nlp.github.io/) is a fact-verification benchmark
where each claim must be checked against multiple Wikipedia documents
that have to be retrieved through a multi-hop reasoning chain. We use
the **3-hop subset** (examples whose set of unique supporting docs has
size 3) and score retrieval recall over those gold supporting docs.

## Architecture (artifact `HoverMultiHop`)

```
HOP 1  BM25 retrieve k=7  using raw claim
       └─ summarize1            (claim, passages -> summary)              # Stage 1
HOP 2  create_query_hop2        (claim, summary_1 -> query)               # Stage 2
       BM25 retrieve k=7
       └─ summarize2            (claim, context, passages -> summary)     # Stage 3
HOP 3  create_query_hop3        (claim, summary_1, summary_2 -> query)    # Stage 4
       BM25 retrieve k=10
```

Final score: gold supporting-doc titles ⊆ titles found in the union of
the 3 hops' retrieved docs (titles are `dspy.evaluate.normalize_text`
normalised).

| Aspect | Detail |
|--------|--------|
| **Optimisation surface** | 4 predictor instructions (jointly) |
| **Train / Val / Test** | 150 / 300 / 300 (artifact `lite` mode, seed 1 trim) |
| **Source** | HuggingFace `hover` train, 3-hop only, shuffled seed 0 |
| **Splits** | test = 0–40 %, val = 40–80 %, train = 80–100 % of shuffled pool |
| **Metric** | `discrete_retrieval_eval` (artifact) — full subset coverage |
| **Retriever** | BM25S over Wikipedia 2017 abstracts (`bm25s` + `Stemmer`) |

## Files

| File | Description |
|------|-------------|
| `artifact_aligned.py` | Data, program (`build_hover_program`), 4-stage tag parser, metric — re-exports from `gepa-artifact` |
| `baseline_test.py`    | Unoptimised `HoverMultiHop` test-set evaluation (DSPy default ChainOfThought instructions) |
| `dynamic_fewshot.py`  | Dynamic 4-stage few-shot prompt search (default: qwen3-8b solver + generator) |
| `dynamic_fewshot_gpt41mini.py` | Same pipeline, `openrouter/openai/gpt-4.1-mini` solver + generator |

## Multi-instruction optimisation

A single iteration produces a **dict** of 4 instructions, one per
predictor stage. The generator LM is asked to emit them in a single
tagged response:

```
<<<SUMMARIZE1>>>
...stage 1 instruction...
<<<QUERY_HOP2>>>
...stage 2 instruction...
<<<SUMMARIZE2>>>
...stage 3 instruction...
<<<QUERY_HOP3>>>
...stage 4 instruction...
```

`parse_four_stage_prompt` enforces:

- All four tags present.
- Tags appear in the canonical execution order.
- Every section is non-empty after stripping.

If parsing fails, the iteration is scored 0 (same fallback policy as
IFBench's two-stage variant). The four instructions are then installed
on the corresponding `ChainOfThought` predictors of a fresh
`HoverMultiHop` via `predict.signature.with_instructions(...)`. This
matches the GEPA paper's joint optimisation surface exactly.

## Prerequisites

1. **Clone `gepa-artifact`** next to this repo (the adapter looks for
   `./gepa-artifact/` first, then `<repo>/gepa-artifact/`).
2. **HuggingFace `hover` dataset**: loaded with `trust_remote_code=True`
   on first call. Requires network access.
3. **BM25 index over wiki.abstracts.2017** (~5 GB download + index
   build). On the first call to any HoVer program forward,
   `gepa_artifact.benchmarks.hover.hover_program.init_retriever` will
   download `wiki.abstracts.2017.tar.gz` (from the DSPy cache mirror)
   into `gepa-artifact/gepa_artifact/benchmarks/hover/` and build the
   `bm25s` index there. Subsequent runs reuse the on-disk index and a
   `diskcache`-backed retrieval cache.
4. **Python deps** (already declared by `gepa-artifact`):
   `bm25s`, `PyStemmer`, `ujson`, `diskcache`, `datasets`, `tqdm`.
   They are not currently in this repo's `pyproject.toml` extras.
   Install ad-hoc, or add a `hover-artifact` extra mirroring
   `ifbench-artifact`.
5. **API keys** (`.env`): set one of
   - `OPENROUTER_API_KEY_HOVER` (preferred for the dynamic-fewshot scripts)
   - `OPENROUTER_API_KEY_HOVER_BASE` (preferred for `baseline_test.py`)
   - `OPENROUTER_API_KEY` (fallback)

## Run

Baseline (unoptimised default ChainOfThought instructions):

```bash
uv run python -m examples.hover.baseline_test --model gpt41mini
uv run python -m examples.hover.baseline_test --model qwen3-8b
```

Dynamic 4-stage few-shot prompt search:

```bash
uv run python -m examples.hover.dynamic_fewshot                 # qwen3-8b
uv run python -m examples.hover.dynamic_fewshot_gpt41mini       # gpt-4.1-mini
```

CLI flags (both variants):

```
--num-iterations    20    # number of (4-stage prompt set) candidates to try
--num-fewshot        3    # claims sampled into the metaprompt per iteration
--top-k              1    # how many top-val candidates to test on testset
--val-sample-size   30    # fixed head-slice of the 300-example artifact valset
--num-threads       16    # dspy.Evaluate parallelism
```

## Notes on the multi-instruction surface

- `build_hover_program(instructions=None)` returns the artifact default
  program (DSPy auto-generated ChainOfThought instructions). Pass a
  partial `{stage_name: text}` dict to override only some predictors —
  useful for ablations isolating one stage at a time.
- `artifact_default_instructions()` returns the four DSPy default
  instruction strings as installed on a fresh `HoverMultiHop`. They are
  also logged to wandb under `stage_instructions` for the baseline run.
- The fixed validation pool size (30) and the fact that we use a
  *head-slice* (not random subset) of the 300-example artifact valset
  match the IFBench / LiveBench-Math / AIME-v2 dynamic-fewshot
  variants in this repo, ensuring cross-task comparability of
  validation noise.
