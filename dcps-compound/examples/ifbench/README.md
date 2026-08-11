# IFBench — GEPA Paper Replication

Strict replication of the IFBench experiment from the GEPA paper (*GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, Agrawal et al., 2025).

## Task

[IFBench](https://github.com/allenai/IFBench) (Pyatkin et al., 2025) evaluates language models' ability to follow precise output constraints (e.g., "answer only with yes or no", "mention a word at least three times"). It introduces 58 new, out-of-distribution constraint types to test generalisation beyond IFEval.

## Paper Setup

| Aspect | Detail |
|--------|--------|
| **Program** | 2-stage: `answer_query` (draft) → `rewrite_answer` (constrained rewrite) |
| **Optimiser** | `dspy.GEPA` — evolves both predictor instructions via reflective mutation |
| **Feedback** | Per-predictor textual feedback listing satisfied and failed constraints |
| **Train/Val** | 150 / 300 from IF-RLVR Train (`allenai/IF_multi_constraints_upto5`) |
| **Test** | 294 from IFBench test (`allenai/IFBench_test`) — unseen constraint types |
| **Metric** | Fraction of constraints satisfied per example (programmatic verification) |

## Architecture

```
ConstrainedRewrite (dspy.Module)
├── answer_query     — ChainOfThought("query -> draft_answer")
└── rewrite_answer   — ChainOfThought("query, draft_answer, constraints -> answer")
```

GEPA optimises the **instructions** of both predictors, using per-predictor feedback that reports exactly which constraints passed and which failed.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — LM config, baseline eval, GEPA optimisation, final eval |
| `utils.py` | ConstrainedRewrite module, GEPA-compatible metric, dataset loading |
| `constraints.py` | Constraint verification registry covering IFEval, IFTrain, and IFBench types |

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
uv run python -m examples.ifbench.main
```

The script will:
1. Load IF-RLVR training data (150 train / 300 val)
2. Load IFBench test data (294 examples with unseen constraint types)
3. Evaluate baseline (unoptimised) ConstrainedRewrite on test set
4. Run `dspy.GEPA` optimisation (budget: `MAX_METRIC_CALLS`)
5. Evaluate optimised program on test set
6. Print comparison and optimised predictor instructions

## Constraint Verification

`constraints.py` implements 100+ constraint verifiers covering:
- **IFEval** (25 original types): keywords, length, format, punctuation, case, etc.
- **IFTrain** (29 new training types): copy, counting, letter frequency, etc.
- **IFBench** (58 new test types): count, words, sentence, format, ratio, repeat, custom

Each verifier returns `(passed: bool, description: str)`, enabling rich textual feedback for GEPA reflection.

## Output

Results are saved to `outputs/ifbench/<run_name>/`, including:
- GEPA optimisation logs
- wandb tracking (if enabled)
- Optimised predictor instructions
