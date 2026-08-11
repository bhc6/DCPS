# LiveBench-Math — GEPA Paper Replication

Strict replication of the LiveBench-Math experiment from the GEPA paper (*GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, Agrawal et al., 2025).

## Task

[LiveBench](https://livebench.ai/) (White et al., 2025) is a cross-domain benchmark consisting of regularly updated questions. We use the **math subset** of LiveBench questions retrieved on July 30, 2025. This set of questions (n=368) covers three task types:

| Task | Count | Answer format |
|------|-------|---------------|
| `AMPS_Hard` | 150 | LaTeX expression in `\boxed{}` |
| `math_comp` | 146 | 3-digit integer |
| `olympiad` | 72 | Comma-separated expression indices |

## Paper Setup

| Aspect | Detail |
|--------|--------|
| **Program** | Single-step `ChainOfThought("question -> answer")` |
| **Optimiser** | `dspy.GEPA` — evolves the predictor instructions via reflective mutation |
| **Feedback** | Textual feedback with correct/incorrect answer and task type |
| **Total** | 368 examples |
| **Split** | Shuffled with `random.Random(0)`, sliced at `int(tot*0.33)` / `int(tot*0.66)` &rarr; **121 / 121 / 126** (matches `gepa-artifact`) |
| **Metric** | Official `calculate_livebench_score` (sub-task aware, strips `<think>…</think>` before scoring) |

## Architecture

```
MathSolver (dspy.Module)
└── solve — ChainOfThought("question -> answer")
```

GEPA optimises the **instructions** of the `solve` predictor.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — LM config, baseline eval, GEPA optimisation, final eval |
| `utils.py` | MathSolver module, answer extraction, metric, dataset loading |

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

## Run

```bash
uv run python -m examples.livebench_math.main
```

The script will:
1. Load LiveBench math subset (368 examples from HuggingFace `livebench/math`)
2. Shuffle with seed 0 and split equally into train/val/test
3. Evaluate baseline (unoptimised) MathSolver on test set
4. Run `dspy.GEPA` optimisation (budget: `MAX_METRIC_CALLS`)
5. Evaluate optimised program on test set
6. Print comparison and optimised predictor instructions

## Answer Evaluation

Scoring is delegated to the **official LiveBench grading pipeline**, vendored
from `gepa-artifact` under `livebenchmath_utils/`:

- **math_comp** (`amc`/`smc`/sub-task starting with these) &rarr; `mathcontest_process_results_with_feedback`
- **math_comp** (`aime` sub-task) &rarr; `aime_process_results`
- **olympiad** (`imo`/`usamo`) &rarr; `proof_rearrangement_process_results(edit_distance=True)`
- **AMPS_Hard** &rarr; `amps_hard_process_results` (sympy-based LaTeX equivalence)

`<think>…</think>` blocks are stripped from the model output before grading,
matching the artifact exactly. This requires `sympy`, `antlr4-python3-runtime==4.11`,
and `lark` to be installed:

```bash
uv pip install sympy 'antlr4-python3-runtime==4.11' lark
```

## Output

Results are saved to `outputs/livebench_math/<run_name>/`, including:
- GEPA optimisation logs
- wandb tracking (if enabled)
- Optimised predictor instructions

## Dynamic few-shot baseline (`dynamic_fewshot.py`)

A non-GEPA baseline ported from `examples/aime_math/dynamic_fewshot.py`. Each
iteration samples few-shot examples from the trainset, asks a generator LM to
produce a candidate system prompt, and evaluates on a val subsample; the top-K
candidates by val score are then evaluated on the full testset.

```bash
uv run python -m examples.livebench_math.dynamic_fewshot
```

This is useful as a comparison point to quantify how much of the reported
accuracy comes from reflective evolution (GEPA) vs. simple meta-prompting.
