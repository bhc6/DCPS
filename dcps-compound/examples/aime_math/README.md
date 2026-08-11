# AIME Math

Optimize a math-solving prompt for AIME competition problems. The solver LLM (GPT-4.1-mini with chain-of-thought) is fixed — GEPA optimizes only the system prompt.

## Dataset

- **Train + Val**: `AI-MO/aimo-validation-aime` (AIME 2022–2024), split 50/50
- **Test**: `MathArena/aime_2025` (AIME 2025)

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install datasets dspy litellm
uv pip install wandb  # only needed if you enable WANDB
uv pip install -e .  # must come after dspy to avoid PyPI overwrite
```

## Run

```bash
export OPENROUTER_API_KEY=...
uv run python -m examples.aime_math.main
```

## Enable Weights & Biases (Optional)

```bash
export USE_WANDB=true
export WANDB_API_KEY=...
export WANDB_PROJECT=gepa-aime-math
# optional
export WANDB_ENTITY=your_team_or_user
export WANDB_RUN_NAME=aime-2025-run-01
```

Then run the same command:

```bash
uv run python -m examples.aime_math.main
```

After optimization, the script evaluates both the baseline and best-found prompt on the AIME 2025 test set and prints the improvement.

## Dynamic few-shot baseline (`dynamic_fewshot.py`)

`dynamic_fewshot.py` is a **non-GEPA baseline**: at each iteration it

1. randomly samples `NUM_FEWSHOT_EXAMPLES` problems from trainset,
2. feeds them into a metaprompt asking a generator LM for a new system prompt,
3. scores the candidate on a random val subsample,

then picks the top-K candidates by val score and evaluates them on test.

Run:

```bash
uv run python -m examples.aime_math.dynamic_fewshot
```

### Porting to other gepa-artifact benchmarks

The method is task-agnostic — it only needs (a) a single-string prompt, (b) an
example schema it can render into a few-shot block, and (c) a numeric metric.
Adapting to another benchmark in the paper requires swapping three things:

| Benchmark | Drop-in? | What to replace |
|---|---|---|
| `livebench_math` | Yes — see `examples/livebench_math/dynamic_fewshot.py` | `load_*_dataset`, metric, few-shot formatter (fields: `question` / `answer` / `task`), solver LM config |
| `ifbench` | Yes | Add `constraints` to the few-shot formatter; metric → constraint-checking metric |
| `arc_agi` | Yes | Few-shot formatter must render input/output grids as text |
| `hotpotqa` | **No — not directly.** The paper uses a multi-module `MultiHopQA` program; a single string candidate cannot represent per-predictor instructions. Would require generalizing candidate to `dict[predictor_name -> instruction]`. |

For an already-adapted example, see
[`examples/livebench_math/dynamic_fewshot.py`](../livebench_math/dynamic_fewshot.py).
