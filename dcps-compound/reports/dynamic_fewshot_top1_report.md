# Dynamic Few-Shot Experiments: Top-1 Comparison Report

Date: 2026-05-02

This report summarizes the AIME-Math, LiveBench-Math, and IFBench dynamic few-shot experiments, with a strict comparison rule: **only the validation-selected Top-1 prompt participates in the main comparison**. IFBench has been switched to the original `gepa-artifact` implementation and must be rerun before its scores are included in the headline comparison.

For dynamic few-shot runs, Top-1 means the first prompt after sorting all generated candidate prompts by validation score in descending order. The reported test score is the test performance of that validation-selected Rank-1 prompt. Scores from the best test result among the top-5 validation-selected prompts are not used for the main comparison.

## Main takeaway

| Benchmark | Model | Method | Selection rule | Val used for selection | Top-1 val | Top-1 test |
|---|---|---|---|---:|---:|---:|
| AIME-Math | Qwen3-8B | Dynamic few-shot v2 LiteLLM | validation Rank-1 | 15 / 45 | 80.00% | 63.33% |
| AIME-Math | GPT-4.1-mini | Dynamic few-shot v2 LiteLLM | validation Rank-1 | 15 / 45 | 60.00% | 43.33% |
| LiveBench-Math | Qwen3-8B | Dynamic few-shot | validation Rank-1 | 30 / 121 | 36.67% | 33.33% |
| LiveBench-Math | GPT-4.1-mini | Dynamic few-shot | validation Rank-1 | 30 / 121 | 36.67% | 30.16% |
| IFBench | Qwen3-8B | Original DSPy baseline | no validation selection | none | pending rerun | pending rerun |
| IFBench | Qwen3-8B | Dynamic few-shot | validation Rank-1 | 30 / 300 | pending rerun | pending rerun |
| IFBench | GPT-4.1-mini | Original DSPy baseline | no validation selection | none | pending rerun | pending rerun |
| IFBench | GPT-4.1-mini | Dynamic few-shot | validation Rank-1 | 30 / 300 | pending rerun | pending rerun |

## Top-1 policy

- **Dynamic few-shot selection**: generate 20 candidate prompts, score each on the validation pool, sort by validation score, then evaluate only the Rank-1 prompt for the main comparison.
- **Tie handling**: use the order printed by the final report table, which follows the script's sorted `top_results` order.
- **Excluded from main comparison**: top-5 average scores and best-test-within-top-5 scores.
- **Baseline handling**: original DSPy baselines do not perform validation-based prompt selection and are evaluated directly on the held-out test set.

This policy is important because selecting the best test score among top-5 validation prompts would use test-set information for model selection. Therefore, best-of-top-5 is reported only as diagnostic information when mentioned, not as the headline comparison value.

## Validation-set design

| Benchmark | Full train | Full validation | Validation used for dynamic selection | Test | Validation strategy |
|---|---:|---:|---:|---:|---|
| AIME-Math v2 | 45 | 45 | 15 | 150 | fixed head-slice of paper-aligned validation set |
| LiveBench-Math | 121 | 121 | 30 | 126 | fixed head-slice of paper-aligned validation set |
| IFBench dynamic | 150 | 300 | 30 | 294 | fixed head-slice of paper-aligned validation set |
| IFBench original baseline | 150 | 300 | 0 | 294 | no validation selection; test-only evaluation |

### Why this matters

The dynamic few-shot experiments include a validation-selection stage, while the original DSPy baseline does not. This is a substantive methodological difference and should be stated in the paper.

The fixed validation pool reduces prompt-selection noise because every candidate prompt is evaluated on the same validation examples. However, the fixed pool is smaller than the full validation split, especially for IFBench where 30 examples represent only 10% of the 300-example validation set.

## Benchmark-specific details

### AIME-Math v2

The AIME-Math v2 dynamic few-shot experiments use the paper-aligned split from `AI-MO/aimo-validation-aime`:

- **Shuffle**: `random.Random(0)`
- **Train**: first 45 examples
- **Full validation**: remaining 45 examples
- **Validation pool for prompt selection**: first 15 examples of the full validation split
- **Test**: `MathArena/aime_2025`, 30 examples replicated 5 times, giving 150 test examples for avg@5-style estimation

#### AIME-Math Top-1 results

| Model | Log file | Top-1 iteration | Top-1 val | Top-1 test | Notes |
|---|---|---:|---:|---:|---|
| Qwen3-8B | `logs_v2_qwen.txt` | 18 | 80.00% | 63.33% | validation Rank-1; best-test among top-5 was not used |
| GPT-4.1-mini | `logs_v2_gpt41mini.txt` | 1 | 60.00% | 43.33% | validation Rank-1; best-test among top-5 was not used |

#### AIME-Math diagnostic values not used in main comparison

| Model | Best test within top-5 | Reason excluded |
|---|---:|---|
| Qwen3-8B | 67.33% | selected by test performance among top-5, not by validation Rank-1 |
| GPT-4.1-mini | 49.33% | selected by test performance among top-5, not by validation Rank-1 |

Older AIME dynamic few-shot logs also exist (`logs_fewshot_dspy.txt`, `logs_fewshot_litellm.txt`), but those runs used per-iteration random validation sampling from the 45-example validation set. They should not be mixed with the v2 fixed-validation results in the main comparison.

### LiveBench-Math

The LiveBench-Math experiments use `livebench/math` with the artifact-aligned split:

- **Total examples**: 368
- **Shuffle**: `random.Random(0)`
- **Train**: 121 examples
- **Full validation**: 121 examples
- **Validation pool for prompt selection**: first 30 examples of the full validation split
- **Test**: 126 examples

#### LiveBench-Math Top-1 results

| Model | Log file | Top-1 iteration | Top-1 val | Top-1 test | Notes |
|---|---|---:|---:|---:|---|
| Qwen3-8B | `logs_lb_qwen.txt` | 3 | 36.67% | 33.33% | validation Rank-1 |
| GPT-4.1-mini | `logs_lb_gpt41mini.txt` | 16 | 36.67% | 30.16% | validation Rank-1 |

#### LiveBench-Math diagnostic values not used in main comparison

| Model | Best test within top-5 | Reason excluded |
|---|---:|---|
| Qwen3-8B | 33.33% | same as Top-1 in this run |
| GPT-4.1-mini | 32.54% | selected by test performance among top-5, not by validation Rank-1 |

### IFBench

The IFBench experiments now use the original `gepa-artifact` implementation, aligned to artifact commit `cbefbc1aa0f43dd39874ec4bf42211365dbda42e`:

- **Train/validation source**: `gepa-artifact/gepa_artifact/benchmarks/IFBench/data/IFBench_train.jsonl`
- **Test source**: `gepa-artifact/gepa_artifact/benchmarks/IFBench/data/IFBench_test.jsonl`
- **Train pool**: `train_val_set[300:600]`
- **Train**: artifact trim policy, `random.Random(1).sample(train_pool, 150)`
- **Full validation**: `train_val_set[:300]`
- **Test**: 294 examples, below the artifact test cap of 300
- **Program schema**: artifact `prompt -> response` two-stage program
- **Metric**: artifact `gepa_artifact.benchmarks.IFBench.ifbench_metric.metric_with_feedback`

#### IFBench original DSPy baseline

The original baseline uses the artifact-style two-stage DSPy program:

- **Stage 1 instruction**: `Respond to the query`
- **Stage 2 instruction**: `Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response.`

It loads the full 300-example validation split for split consistency and logging, but it does not use validation for prompt selection. It evaluates directly on the 294-example IFBench test split.

| Model | Log file | Validation selection | Test score |
|---|---|---|---:|
| Qwen3-8B | pending rerun | none | pending rerun |
| GPT-4.1-mini | pending rerun | none | pending rerun |

#### IFBench dynamic few-shot Top-1 results

| Model | Log file | Top-1 iteration | Top-1 val | Top-1 test | Delta vs original DSPy baseline |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |
| GPT-4.1-mini | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |

#### IFBench diagnostic values not used in main comparison

| Model | Best test within top-5 | Reason excluded |
|---|---:|---|
| Qwen3-8B | pending rerun | rerun required after artifact-aligned IFBench switch |
| GPT-4.1-mini | pending rerun | rerun required after artifact-aligned IFBench switch |

Previous IFBench numbers from `logs_if_base_qwen.txt`, `logs_if_base_gpt41mini.txt`, `logs_if_qwen.txt`, and `logs_if_gpt41mini.txt` were produced before switching to the artifact-aligned IFBench JSONL/schema/metric adapter. They are retained only as historical diagnostics and should not be compared against the original paper or used in the headline table.

## Model and method notes

### Alignment policy

The experiments are intended to match the original paper's experimental protocol in all comparison-relevant aspects: data source and split, held-out test usage, model family, decoding parameters, prompt/program structure, metric, and validation-based selection rule. The use of API endpoints is an implementation detail and does not by itself invalidate comparability, provided the served model and inference parameters match the paper configuration. IFBench uses a local adapter over `gepa-artifact` data/program/metric rather than the earlier HuggingFace/custom-checker utility stack.

For IFBench, the original paper protocol keeps IFBench as a held-out test set so the optimizer does not access the new, unseen test constraints before final evaluation. The dynamic few-shot baseline follows this held-out-test principle: few-shot examples are sampled from the training split, candidate prompts are selected on validation examples, and the 294-example IFBench test split is used only for final evaluation.

### Qwen3-8B configuration

The Qwen3-8B runs use API inference with the paper-aligned sampling settings:

- **Model**: `openrouter/qwen/qwen3-8b`
- **Temperature**: 0.6
- **Top-p**: 0.95
- **Top-k**: 20
- **Max tokens**: 16384
- **Provider**: Alibaba pinned where configured for Qwen3-8B API routing

### GPT-4.1-mini configuration

The GPT-4.1-mini runs use API inference with the paper-aligned sampling settings:

- **Model**: `openrouter/openai/gpt-4.1-mini`
- **Temperature**: 1.0
- **Max tokens**: 16384

Some GPT-4.1-mini scripts still pass `top_p=0.95` in AIME variants; this should be noted if reporting exact run configuration. For IFBench, GPT-4.1-mini uses temperature 1.0 and does not use Qwen-specific top-p/top-k/provider settings.

## W&B run references

| Experiment | W&B run |
|---|---|
| AIME v2 Qwen3-8B | `https://wandb.ai/awesome-prompt/aime-math-litellm-agnostic-nb/runs/5nnpwzju` |
| AIME v2 GPT-4.1-mini | `https://wandb.ai/awesome-prompt/aime-math-litellm-agnostic-nb/runs/4fx08hfi` |
| LiveBench Qwen3-8B | `https://wandb.ai/awesome-prompt/livebench-math-dynamic-fewshot/runs/7jl5dta0` |
| LiveBench GPT-4.1-mini | `https://wandb.ai/awesome-prompt/livebench-math-dynamic-fewshot/runs/h6lucked` |
| IFBench Qwen3-8B dynamic | pending artifact-aligned rerun; previous pre-alignment run was `https://wandb.ai/awesome-prompt/ifbench-dynamic-fewshot/runs/9ogkl1i5` |
| IFBench GPT-4.1-mini dynamic | pending artifact-aligned rerun; previous pre-alignment run was `https://wandb.ai/awesome-prompt/ifbench-dynamic-fewshot/runs/w5dkndzl` |
| IFBench Qwen3-8B original baseline | pending artifact-aligned rerun; previous pre-alignment run was `https://wandb.ai/awesome-prompt/ifbench-baseline-test/runs/lxer1u42` |
| IFBench GPT-4.1-mini original baseline | pending artifact-aligned rerun; previous pre-alignment run was `https://wandb.ai/awesome-prompt/ifbench-baseline-test/runs/vfjynj07` |

## Suggested paper wording

```text
All experiments are aligned with the original paper protocol in the comparison-relevant dimensions: train/validation/test separation, held-out test evaluation, model family, decoding parameters, prompt/program structure, and validation-based selection. For IFBench, data/program/metric are taken from the original GEPA artifact. The API route is an implementation detail and is not treated as a methodological difference, provided the same model and inference parameters are used.

For dynamic few-shot prompt search, we preserve the benchmark-specific paper-aligned train/validation/test splits, but use a fixed subset of the validation split for candidate prompt selection. Specifically, AIME-Math v2 uses the first 15 examples from the 45-example validation split, LiveBench-Math uses the first 30 examples from the 121-example validation split, and IFBench uses the first 30 examples from the 300-example validation split. The selected validation pool is reused for all prompt-generation iterations, ensuring that candidate prompts are compared on exactly the same examples.

For reporting dynamic few-shot performance, we use the validation-selected Top-1 prompt only. That is, candidate prompts are ranked by validation score, and the Rank-1 prompt is evaluated on the held-out test set. We do not select the best test score among the top-k validation-selected prompts for the main comparison.

The original DSPy baselines do not perform validation-based prompt selection; they use the unoptimized prompt/program and are evaluated directly on the test split.
```

## Final conclusion

Under the strict Top-1-by-validation rule:

- **AIME-Math**: Qwen3-8B dynamic few-shot v2 outperforms GPT-4.1-mini in this run, 63.33% vs 43.33%.
- **LiveBench-Math**: Qwen3-8B dynamic few-shot outperforms GPT-4.1-mini in this run, 33.33% vs 30.16%.
- **IFBench**: pending rerun under the artifact-aligned IFBench JSONL/schema/metric adapter.

The best-test-within-top-5 numbers are useful diagnostics, but they should not be used as headline comparisons because they implicitly select using test performance.
