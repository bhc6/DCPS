"""Dynamic few-shot prompt search baseline for LiveBench-Math.

Mirrors ``examples/aime_math/dynamic_fewshot.py`` but adapted to the LiveBench
math benchmark, and aligned byte-for-byte with the evaluation harness in
``gepa-artifact/gepa_artifact/benchmarks/livebench_math``:

  - Dataset: ``livebench/math`` (368 examples, seed-0 shuffle, split at
    ``int(tot*0.33)`` / ``int(tot*0.66)`` → 121 / 121 / 126).
  - Program: single-step ``ChainOfThought("question -> answer")``.
  - Metric : official ``calculate_livebench_score`` (sub-task aware, strips
    ``<think>…</think>`` before scoring).
  - Examples carry the raw HF row as ``question_d`` so the metric can grade.

Algorithm (unchanged from AIME version):
  1. Randomly sample a few-shot set from trainset.
  2. Ask a generator LM to produce a candidate system prompt conditioned on
     those few-shots + a base prompt.
  3. Evaluate the candidate on a random val subsample.
  4. After N iterations, select top-K by val score and evaluate on testset.
"""

import os
import random
import json
from typing import List
from datetime import datetime

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

from examples.livebench_math.utils import (
    MathSolver,
    load_livebench_math_dataset,
)
from examples.livebench_math.livebenchmath_utils.metric import (
    calculate_livebench_score,
)
from examples.livebench_math._run_tracking import (
    PHASE_GENERATE,
    PHASE_OPT_EVAL,
    PHASE_TEST_EVAL,
    PHASES,
    USAGE,
    add_snapshots,
    append_checkpoint,
    checkpoint_path,
    empty_bucket,
    ensure_usage_callback,
    load_checkpoint,
    sum_buckets,
)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _build_solver(instructions: str) -> MathSolver:
    """Return a fresh MathSolver whose `solve` predictor uses `instructions`."""
    solver = MathSolver()
    solver.solve.predict.signature.instructions = instructions
    return solver


def evaluate_on_dataset(prompt: str, dataset, num_threads: int = 16, record_history: bool = False, output_dir: str = None, model_config: dict = None) -> float:
    """Evaluate a candidate prompt on a dataset (accuracy in [0, 1]).

    Uses the official ``calculate_livebench_score`` so numbers are directly
    comparable with ``gepa-artifact`` / the paper. Any per-example exception
    (LM failure, metric crash, etc.) is caught and scored as 0.
    
    Args:
        prompt: The prompt to test
        dataset: The dataset to evaluate on
        num_threads: Number of threads for evaluation
        record_history: Whether to record detailed history for each example
        output_dir: Directory to save history JSON file (required if record_history=True)
        model_config: Dictionary containing model configuration (model, temperature, etc.)
    
    Returns:
        score: The evaluation score (0-1)
        history: Optional dict with detailed history if record_history=True
    """
    solver = _build_solver(prompt)

    # Storage for detailed history
    history_data = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "model_config": model_config or {},
        "examples": []
    } if record_history else None

    def dspy_metric(example, prediction, trace=None):
        try:
            score, _ = calculate_livebench_score(
                example["question_d"], getattr(prediction, "answer", ""), debug=False
            )
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "question": example.question,
                    "task": example.task,
                    "subtask": example.question_d.get("subtask", "N/A"),
                    "ground_truth": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": float(score),
                })
            return float(score)
        except Exception as e:
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "question": example.question,
                    "task": example.task,
                    "subtask": example.question_d.get("subtask", "N/A"),
                    "ground_truth": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": 0.0,
                    "error": str(e),
                })
            return 0.0

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=dspy_metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )
    result = evaluator(solver)
    score = result.score / 100.0
    
    # Record LM statistics
    if record_history and history_data is not None:
        lm = dspy.settings.lm
        if lm and hasattr(lm, 'history'):
            total_cost = 0
            total_input_tokens = 0
            total_output_tokens = 0
            for trace in lm.history:
                total_cost += trace.get("cost", 0) or 0
                total_input_tokens += trace.get("usage", {}).get("prompt_tokens", 0)
                total_output_tokens += trace.get("usage", {}).get("completion_tokens", 0)
            
            history_data["lm_statistics"] = {
                "total_cost": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "num_calls": len(lm.history),
            }
        
        history_data["overall_score"] = score
        history_data["num_correct"] = sum(1 for ex in history_data["examples"] if ex.get("score") == 1.0)

        # Save to JSON file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_history_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"  History saved to: {filepath}")

    return score, history_data if record_history else (score, None)


# ---------------------------------------------------------------------------
# Few-shot sampling & metaprompt
# ---------------------------------------------------------------------------


def sample_fewshot_examples(trainset: List, num_examples: int = 3) -> str:
    """Randomly sample few-shot examples from the training set.

    LiveBench examples only have `question`, `answer`, `task` — no written
    solution — so the formatter is simpler than the AIME version.
    """
    sampled = random.sample(trainset, min(num_examples, len(trainset)))

    out = ""
    for i, ex in enumerate(sampled, 1):
        subtask = ex.question_d.get("subtask", ex.task)
        out += f"Example {i} (task={ex.task}, subtask={subtask}):\n"
        out += f"Question: {ex.question}\n"
        out += f"Answer: {ex.answer}\n\n"
    return out.strip()


def create_metaprompt(fewshot_examples: str, base_prompt: str) -> str:
    """Metaprompt that asks a generator LM for a math-solving system prompt."""
    return f"""You are an expert prompt engineer for AI systems that solve LiveBench math problems.

LiveBench-Math contains three task types:
  - AMPS_Hard : LaTeX expression expected inside \\boxed{{...}}
  - math_comp : a 3-digit integer (0-999)
  - olympiad  : a comma-separated list of expression indices

Based on the base prompt and few-shot examples below, design an effective system prompt that will guide an AI to solve similar math problems accurately. The prompt should:

1. Be clear and specific about the expected final-answer format for each task.
2. Encourage step-by-step reasoning before emitting the final answer.
3. Reference patterns shown in the examples.
4. Instruct the AI to always place the final answer inside \\boxed{{...}}.

Here are the few-shot examples to analyze:

{fewshot_examples}

base prompt:
{base_prompt}

Now, generate a prompt that incorporates insights from these examples:"""


def generate_prompt_with_llm(metaprompt: str) -> str:
    """Use the configured (generator) LM to produce a candidate prompt."""

    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(
            desc="The metaprompt for generating a math solving prompt."
        )
        generated_prompt = dspy.OutputField(
            desc="The generated prompt for solving math problems."
        )

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    try:
        result = generator(metaprompt=metaprompt)
        return result.generated_prompt or ""
    except Exception as exc:
        print(f"[generate_prompt_with_llm] generator failed: {exc!r}")
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic few-shot prompt search for LiveBench-Math")
    parser.add_argument("--record-history", action="store_true", default=False, help="Record detailed evaluation history to JSON files")
    parser.add_argument("--output-dir", type=str, default="evaluation_history", help="Directory to save history JSON files")
    parser.add_argument("--num-iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--num-fewshot", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top prompts to select (default 1, matches hover; >1 only for diagnostics)")
    parser.add_argument("--val-sample-size", type=int, default=30, help="Validation sample size per iteration")
    parser.add_argument("--num-threads", type=int, default=16, help="Threads for evaluation")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore and delete any existing checkpoint; start fresh from iteration 1.",
    )
    args = parser.parse_args()

    load_dotenv()

    ensure_usage_callback()

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # Configuration
    NUM_ITERATIONS = args.num_iterations
    NUM_FEWSHOT_EXAMPLES = args.num_fewshot
    TOP_K = args.top_k
    VAL_SAMPLE_SIZE = args.val_sample_size
    NUM_THREADS = args.num_threads
    SOLVER_MODEL = "openrouter/openai/gpt-4.1-mini"
    GENERATOR_MODEL = "openrouter/openai/gpt-4.1-mini"
    RUN_NAME = f"dynamic_fewshot_gpt41mini_{NUM_ITERATIONS}iter_{NUM_FEWSHOT_EXAMPLES}shot"
    BASE_PROMPT = (
        "You are a helpful assistant. You are given a math question and you need "
        "to solve it step by step. Always place the final answer inside \\boxed{}."
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in your environment or .env file."
        )

    # gpt-4.1-mini sampling matches gepa-artifact's LM_CONFIGS: temperature=1.0.
    # timeout=600 + num_retries=2 guard against OpenRouter intermittently
    # dropping the streaming connection mid-response ("peer closed connection
    # without sending complete message body"), which otherwise hangs the whole
    # evaluation forever. Neither changes the sampling distribution.
    solver_lm = dspy.LM(
        SOLVER_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        num_retries=2,
        timeout=600,
        cache=False,
        extra_body={
            "top_k": 20,
        },
    )
    generator_lm = dspy.LM(
        GENERATOR_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        num_retries=2,
        timeout=600,
        cache=False,
        extra_body={
            "top_k": 20,
        },
    )
    # Disable ChatAdapter -> JSONAdapter fallback to avoid Qwen3-8B thinking mode
    # + JSON mode incompatibility (400 error). Failures are caught and scored 0.
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )

    trainset, full_valset, testset = load_livebench_math_dataset()

    # ---- Fixed validation pool: deterministic head-slice of the paper-aligned
    # valset, reused for every iteration. The underlying dataset split is
    # untouched; we just stop per-iter random resampling so candidate prompts
    # are scored on the *same* problems for fair comparison.
    valset = full_valset[:VAL_SAMPLE_SIZE]
    print(f"[OK] Fixed validation pool: {len(valset)} / {len(full_valset)} (head-slice of paper valset)")

    wandb.init(
        project="livebench-math-dynamic-fewshot",
        name=RUN_NAME,
        config={
            "num_iterations": NUM_ITERATIONS,
            "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
            "top_k_prompts": TOP_K,
            "val_sample_size": VAL_SAMPLE_SIZE,
            "solver_model": SOLVER_MODEL,
            "generator_model": GENERATOR_MODEL,
            "base_prompt": BASE_PROMPT,
            "sampling_temperature": solver_lm.kwargs.get("temperature"),
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "trainset_size": len(trainset),
            "full_valset_size": len(full_valset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "val_strategy": "fixed_head_slice_of_paper_valset",
            "split_alignment": "identical_to_paper",
        },
    )

    print("Starting dynamic few-shot prompt generation (LiveBench-Math)...")
    print(
        f"Iterations: {NUM_ITERATIONS}, Few-shot examples: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}"
    )
    print(
        f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}"
    )
    print(f"Validation sample size per iteration: {VAL_SAMPLE_SIZE}")
    print(f"History recording: {args.record_history}")
    if args.record_history:
        print(f"Output directory: {args.output_dir}")

    # Prepare model configuration for history recording
    model_config = {
        "model": SOLVER_MODEL,
        "temperature": solver_lm.kwargs.get("temperature"),
        "top_p": solver_lm.kwargs.get("top_p"),
        "top_k": solver_lm.kwargs.get("extra_body", {}).get("top_k"),
        "max_tokens": solver_lm.kwargs.get("max_tokens"),
        "num_retries": solver_lm.kwargs.get("num_retries"),
    }

    # ---- Resume support -------------------------------------------------
    ckpt_path = checkpoint_path(RUN_NAME)
    if args.no_resume and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[checkpoint] no-resume; removed existing checkpoint {ckpt_path.name}")

    completed = [] if args.no_resume else load_checkpoint(RUN_NAME)
    completed = [r for r in completed if r["iteration"] <= NUM_ITERATIONS]
    done_iterations = {r["iteration"] for r in completed}
    if completed:
        print(
            f"[checkpoint] resuming from {ckpt_path.name}: "
            f"{len(done_iterations)} iteration(s) already done, "
            f"continuing at iteration {len(done_iterations) + 1}"
        )

    all_results = list(completed)
    best_val_score = max((r["val_score"] for r in completed), default=0.0)

    # Cumulative wandb table — re-logged each iteration so UI updates live.
    prompt_table_cols = ["iteration", "val_score", "prompt", "fewshot_examples"]
    prompt_table_rows: list[list] = [
        [r["iteration"], r["val_score"], r["prompt"], r["fewshot_examples"]]
        for r in completed
    ]

    # Per-phase usage carried over from already-completed iterations so the
    # running totals stay monotonic across a resumed run. Old records lacking
    # "usage_by_phase" simply contribute zeros.
    carried_usage = {p: empty_bucket() for p in PHASES}
    for r in completed:
        u = r.get("usage_by_phase")
        if u:
            carried_usage = add_snapshots(carried_usage, u)

    for iteration in range(NUM_ITERATIONS):
        if (iteration + 1) in done_iterations:
            continue
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        usage_before = USAGE.snapshot()
        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)
        print(f"Sampled {NUM_FEWSHOT_EXAMPLES} examples from training set")

        metaprompt = create_metaprompt(fewshot_examples, BASE_PROMPT)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=generator_lm,
        )
        with USAGE.phase(PHASE_GENERATE):
            generated_prompt = generate_prompt_with_llm(metaprompt)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=solver_lm,
        )

        print(f"Generated prompt: {generated_prompt}")

        # Fixed val pool reused across iterations (built once before the loop)
        if not generated_prompt:
            print("[iteration] empty generated prompt, scoring 0.")
            val_score = 0.0
        else:
            try:
                with USAGE.phase(PHASE_OPT_EVAL):
                    val_score, _ = evaluate_on_dataset(
                        generated_prompt,
                        valset,
                        num_threads=NUM_THREADS,
                        record_history=args.record_history,
                        output_dir=args.output_dir if args.record_history else None,
                        model_config=model_config if args.record_history else None
                    )
            except Exception as exc:
                print(f"[iteration] evaluation failed: {exc!r}; scoring 0.")
                val_score = 0.0
        print(
            f"Validation score (on {len(valset)} fixed samples): {val_score:.2%}"
        )

        best_val_score = max(best_val_score, val_score)

        usage_after = USAGE.snapshot()
        iter_by_phase = USAGE.delta_since(usage_before, usage_after)
        gen_b = iter_by_phase[PHASE_GENERATE]
        eval_b = iter_by_phase[PHASE_OPT_EVAL]
        iter_total = sum_buckets(iter_by_phase, (PHASE_GENERATE, PHASE_OPT_EVAL))
        cum_total = sum_buckets(add_snapshots(carried_usage, usage_after),
                                (PHASE_GENERATE, PHASE_OPT_EVAL))
        print(
            f"  [generate] in {gen_b['prompt_tokens']:,} / out {gen_b['completion_tokens']:,} tok, "
            f"${gen_b['cost_usd']:.4f}  |  "
            f"[opt-eval] in {eval_b['prompt_tokens']:,} / out {eval_b['completion_tokens']:,} tok, "
            f"${eval_b['cost_usd']:.4f}  ({eval_b['calls']} calls)"
        )
        print(
            f"  iter total {iter_total['total_tokens']:,} tok ${iter_total['cost_usd']:.4f}  |  "
            f"cumulative opt {cum_total['total_tokens']:,} tok ${cum_total['cost_usd']:.4f}"
        )

        # Single-commit log (no explicit step=) + cumulative table so the
        # wandb UI updates in near real-time instead of at run end. Best-effort:
        # a transient wandb I/O error must never crash the run (the checkpoint
        # below is the source of truth).
        prompt_table_rows.append([
            iteration + 1,
            val_score,
            generated_prompt,
            fewshot_examples,
        ])
        try:
            wandb.log({
                "iteration": iteration + 1,
                "val_score": val_score,
                "best_val_score": best_val_score,
                "iter_generate_input_tokens": gen_b["prompt_tokens"],
                "iter_generate_output_tokens": gen_b["completion_tokens"],
                "iter_generate_cost_usd": gen_b["cost_usd"],
                "iter_generate_calls": gen_b["calls"],
                "iter_opt_eval_input_tokens": eval_b["prompt_tokens"],
                "iter_opt_eval_output_tokens": eval_b["completion_tokens"],
                "iter_opt_eval_cost_usd": eval_b["cost_usd"],
                "iter_opt_eval_calls": eval_b["calls"],
                "iter_total_tokens": iter_total["total_tokens"],
                "iter_cost_usd": iter_total["cost_usd"],
                "cumulative_opt_input_tokens": cum_total["prompt_tokens"],
                "cumulative_opt_output_tokens": cum_total["completion_tokens"],
                "cumulative_opt_total_tokens": cum_total["total_tokens"],
                "cumulative_opt_cost_usd": cum_total["cost_usd"],
                "cumulative_opt_calls": cum_total["calls"],
                "iteration_prompts": wandb.Table(
                    columns=prompt_table_cols,
                    data=list(prompt_table_rows),
                ),
            })
        except Exception as exc:
            print(f"[wandb] log failed for iteration {iteration + 1} (continuing): {exc!r}")

        record = {
            "iteration": iteration + 1,
            "prompt": generated_prompt,
            "val_score": val_score,
            "fewshot_examples": fewshot_examples,
            "usage_by_phase": iter_by_phase,
        }
        all_results.append(record)
        append_checkpoint(RUN_NAME, record)

    all_results.sort(key=lambda x: x["val_score"], reverse=True)
    top_results = all_results[:TOP_K]

    print(f"\n=== Top {TOP_K} Prompts from Validation ===")
    for i, result in enumerate(top_results):
        print(f"\nRank {i + 1} (Iteration {result['iteration']}):")
        print(f"Validation Score: {result['val_score']:.2%}")
        print(f"Prompt: {result['prompt']}")

    top_k_val_table = wandb.Table(
        columns=["rank", "iteration", "val_score", "prompt"],
        data=[[i + 1, r["iteration"], r["val_score"], r["prompt"]]
              for i, r in enumerate(top_results)],
    )
    try:
        wandb.log({"top_k_validation": top_k_val_table})
    except Exception as exc:
        print(f"[wandb] top_k_validation log failed (continuing): {exc!r}")

    print("\n=== Final Testing on Test Set ===")
    final_results = []

    final_cols = [
        "rank", "iteration", "val_score", "test_score", "val_test_gap", "prompt",
    ]
    final_rows: list[list] = []

    for i, result in enumerate(top_results):
        print(f"\nTesting Rank {i + 1} on test set...")
        try:
            with USAGE.phase(PHASE_TEST_EVAL):
                test_score, _ = evaluate_on_dataset(
                    result["prompt"],
                    testset,
                    num_threads=NUM_THREADS,
                    record_history=args.record_history,
                    output_dir=args.output_dir if args.record_history else None,
                    model_config=model_config if args.record_history else None
                )
        except Exception as exc:
            print(f"  [test] evaluation failed: {exc!r}; scoring 0.")
            test_score = 0.0
        print(f"  Test Score: {test_score:.2%}")

        final_rows.append([
            i + 1,
            result["iteration"],
            result["val_score"],
            test_score,
            result["val_score"] - test_score,
            result["prompt"],
        ])

        try:
            wandb.log({
                "test_rank": i + 1,
                "test_score": test_score,
                "test_val_score": result["val_score"],
                "val_test_gap": result["val_score"] - test_score,
                "final_results": wandb.Table(columns=final_cols, data=list(final_rows)),
            })
        except Exception as exc:
            print(f"[wandb] test-rank log failed for rank {i + 1} (continuing): {exc!r}")

        final_results.append({
            "rank": i + 1,
            "iteration": result["iteration"],
            "prompt": result["prompt"],
            "val_score": result["val_score"],
            "test_score": test_score,
        })

    print("\n=== FINAL REPORT ===")
    print(f"Generated {NUM_ITERATIONS} prompts using dynamic few-shot sampling")
    print(f"Selected top {TOP_K} based on validation performance")
    print(f"Tested on {len(testset)} test examples\n")

    print("Rank | Iteration | Val Score | Test Score | Prompt")
    print("-" * 100)
    for result in final_results:
        print(
            f"{result['rank']:4} | {result['iteration']:9} | "
            f"{result['val_score']:9.2%} | {result['test_score']:10.2%} | "
            f"{result['prompt'][:30]}..."
        )

    # Fairness: report the test score of the prompt with the BEST VALIDATION
    # score, not the best test score. final_results is ordered by descending
    # val_score (top_results was sorted that way), so rank 1 == highest-val
    # prompt. Picking max(test_score) here would be selecting on the test set,
    # which leaks the test set into model selection. Matches hover's top_k=1.
    best_result = final_results[0]
    print("\nBest Overall Performance:")
    print(f"Iteration: {best_result['iteration']}")
    print(f"Validation Score: {best_result['val_score']:.2%}")
    print(f"Test Score: {best_result['test_score']:.2%}")
    print(f"Prompt: {best_result['prompt']}")

    avg_val_score = sum(r["val_score"] for r in final_results) / len(final_results)
    avg_test_score = sum(r["test_score"] for r in final_results) / len(final_results)
    avg_gap = avg_val_score - avg_test_score

    print("\nPerformance Summary:")
    print(f"Average Validation Score (Top {TOP_K}): {avg_val_score:.2%}")
    print(f"Average Test Score (Top {TOP_K}): {avg_test_score:.2%}")
    print(f"Validation-to-Test Gap: {avg_gap:.2%}")

    # final_results table is already logged incrementally in the test loop.

    # Full per-phase usage = carried (prior iterations) + this process's calls.
    final_snap = add_snapshots(carried_usage, USAGE.snapshot())
    gen = final_snap[PHASE_GENERATE]
    opt_eval = final_snap[PHASE_OPT_EVAL]
    test_eval = final_snap[PHASE_TEST_EVAL]
    optimize_total = sum_buckets(final_snap, (PHASE_GENERATE, PHASE_OPT_EVAL))
    grand_total = sum_buckets(final_snap, PHASES)

    def _fmt(label: str, b: dict) -> str:
        return (
            f"  {label:<22} input {b['prompt_tokens']:>12,}  output {b['completion_tokens']:>12,}  "
            f"total {b['total_tokens']:>12,}  calls {b['calls']:>7,}  ${b['cost_usd']:.4f}"
        )

    print("\n=== TOKEN / COST USAGE (whole run, incl. resumed iterations) ===")
    print("OPTIMIZATION:")
    print(_fmt("generate (prompts)", gen))
    print(_fmt("eval (validation)", opt_eval))
    print(_fmt("optimize subtotal", optimize_total))
    print("TESTING:")
    print(_fmt("eval (test set)", test_eval))
    print("GRAND TOTAL:")
    print(_fmt("all phases", grand_total))

    if wandb.run is not None:
        summary = {
            "best_test_score": best_result["test_score"],
            "best_val_score": best_result["val_score"],
            "best_iteration": best_result["iteration"],
            "best_prompt": best_result["prompt"],
            "avg_val_score_top_k": avg_val_score,
            "avg_test_score_top_k": avg_test_score,
            "val_test_gap": avg_val_score - avg_test_score,
            "total_iterations": NUM_ITERATIONS,
        }
        for label, b in (
            ("generate", gen),
            ("opt_eval", opt_eval),
            ("optimize_total", optimize_total),
            ("test_eval", test_eval),
            ("grand_total", grand_total),
        ):
            summary[f"usage_{label}_input_tokens"] = b["prompt_tokens"]
            summary[f"usage_{label}_output_tokens"] = b["completion_tokens"]
            summary[f"usage_{label}_total_tokens"] = b["total_tokens"]
            summary[f"usage_{label}_cost_usd"] = b["cost_usd"]
            summary[f"usage_{label}_calls"] = b["calls"]
        try:
            wandb.run.summary.update(summary)
        except Exception as exc:
            print(f"[wandb] summary update failed (continuing): {exc!r}")

    try:
        wandb.finish()
    except Exception as exc:
        print(f"[wandb] finish failed (ignored): {exc!r}")


if __name__ == "__main__":
    main()
