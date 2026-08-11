# HotpotQA — GEPA Paper Replication

Strict replication of the HotpotQA experiment from the GEPA paper (*GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, Agrawal et al., 2025).

## Task

[HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018) is a multi-hop question-answering dataset with 113K Wikipedia-based question-answer pairs requiring reasoning over multiple supporting documents.

## Paper Setup

| Aspect | Detail |
|--------|--------|
| **Program** | HoVerMultiHop-style: 3-hop loop with `generate_query` → BM25 retrieve → `append_notes`, last hop replaced by `answer_question` |
| **Retrieval** | BM25 over Wikipedia 2017 abstracts (~5.2M documents) |
| **Optimiser** | `dspy.GEPA` — evolves all predictor instructions via reflective mutation |
| **Feedback** | Per-predictor textual feedback identifying missing documents / retrieval gaps |
| **Splits** | 150 train / 300 val / 300 test |
| **Metric** | Token-level F1 (HotpotQA official) |

## Architecture

```
MultiHopQA (dspy.Module)
├── generate_query   — ChainOfThought("question, notes -> query")      [×3 hops]
├── append_notes     — ChainOfThought("question, notes, context -> new_notes")  [hops 1-2]
└── answer_question  — ChainOfThought("question, notes, context -> answer")     [hop 3]
```

GEPA optimises the **instructions** of all three predictors jointly using per-predictor feedback.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — configures LMs, runs baseline eval, GEPA optimisation, and final eval |
| `utils.py` | MultiHopQA module, BM25 retriever, GEPA-compatible metric, dataset loading |

## Configuration

Edit the constants at the top of `main.py`:

```python
SOLVER_LM_MODEL = "openrouter/openai/gpt-4.1-mini"
REFLECTION_LM_MODEL = "openrouter/openai/gpt-5.1"
MAX_METRIC_CALLS = 500
NUM_THREADS = 16
```

## Prerequisites

1. **API key** — set `OPENROUTER_API_KEY` in the project root `.env` file.
2. **wandb** (optional) — set `WANDB_API_KEY` in `.env` to enable experiment tracking.
3. **Wikipedia corpus** — downloaded automatically on first run (~500 MB compressed).

## Run

```bash
uv run python -m examples.hotpotqa.main
```

First run will:
1. Download & index Wikipedia 2017 abstracts (one-time, ~2-3 min)
2. Load HotpotQA dataset (150/300/300)
3. Evaluate baseline (unoptimised) MultiHopQA on test set
4. Run `dspy.GEPA` optimisation (budget: `MAX_METRIC_CALLS`)
5. Evaluate optimised program on test set
6. Print comparison and optimised predictor instructions

## Output

Results are saved to `outputs/hotpotqa/<run_name>/`, including:
- GEPA optimisation logs
- wandb tracking (if enabled)
- Optimised predictor instructions
