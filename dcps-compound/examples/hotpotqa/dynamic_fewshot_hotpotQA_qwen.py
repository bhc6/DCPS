"""DCPS (Demonstration-Conditioned Prompt Search) for HotpotQA — Qwen3-8B.

Faithful port of the run script from the paper artifact
(``dcps-artifact/examples/hotpotqa/dynamic_fewshot_hotpotQA_qwen.py``), which
produced the paper's DCPS-Compound HotpotQA Qwen3-8B cell (61.33).

Identical in mechanism to ``dynamic_fewshot_hotpotQA.py`` — one ``dspy.Predict``
call emits all four ``HotpotMultiHop`` system prompts from 5 random train
demonstrations, running argmax over a fixed 50-example validation subset, top-1
to test, EM metric. Differences from the GPT entrypoint, all from the archived
script:

  * Qwen3-8B sampling: temperature 0.6, top_p 0.95, top_k 20, provider pinned to
    alibaba, max_tokens 16384, num_retries 0.
  * NUM_THREADS defaults to 8, not 32 (the archived comment: "Lowered to avoid
    OOM on low-memory server").
  * LM ``history`` is cleared and ``gc.collect()`` called after every iteration,
    with the token baselines reset to 0 to keep the cumulative math consistent
    with the emptied history. Preserved as-is.

Deviations from the archive match the GPT entrypoint: artifact path resolution,
the unused ``EXP_MODE`` block dropped (it always ran the fixed Qwen LM pair), and
ITERATIONS env-configurable with default 20 (main table) rather than 120.

Run:
    uv run python -m examples.hotpotqa.dynamic_fewshot_hotpotQA_qwen
"""

import gc
import json
import os
import random
import sys
from pathlib import Path

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter


def _artifact_root() -> Path:
    """Locate the vendored GEPA benchmark suite (``gepa-artifact[-origin]``)."""
    here = Path(__file__).resolve()
    for base in (Path.cwd(), *here.parents):
        for name in ("gepa-artifact-origin", "gepa-artifact"):
            candidate = base / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "gepa-artifact not found. Clone https://github.com/gepa-ai/gepa-artifact "
        "into the repo root."
    )


sys.path.insert(0, str(_artifact_root()))

import dspy.evaluate  # noqa: E402
from gepa_artifact.benchmarks.hotpotQA.hotpot_data import HotpotQABench  # noqa: E402
from gepa_artifact.benchmarks.hotpotQA.hotpot_program import HotpotMultiHop  # noqa: E402


class GeneratePromptsHotpotQA(dspy.Signature):
    """
    You are an expert prompt engineer optimizing a 4-stage Multi-hop QA pipeline (HotpotQA).

    The pipeline answers complex questions by:
    1. summarize1: Summarizing the initially retrieved passages.
    2. create_query_hop2: Generating a new search query for a 2nd hop based on the question and first summary.
    3. summarize2: Summarizing the newly retrieved 2nd hop passages along with the previous context.
    4. final_answer: Generating the final concise answer based on all summaries.

    Your task is to write highly effective System Prompts for each of these 4 agents.
    The goal is to maximize Exact Match (EM) and F1 score against the gold answer.
    """

    dataset_context = dspy.InputField(desc="Description of the task and a few examples.")
    summary1_prompt = dspy.OutputField(desc="System prompt for the summarize1 agent.")
    query2_prompt = dspy.OutputField(desc="System prompt for the create_query_hop2 agent.")
    summary2_prompt = dspy.OutputField(desc="System prompt for the summarize2 agent.")
    answer_prompt = dspy.OutputField(desc="System prompt for the final_answer agent.")


def sample_fewshot_examples(trainset, n=5):
    samples = random.sample(trainset, min(n, len(trainset)))
    context = "Here are some examples of questions and their correct answers from the HotpotQA dataset:\n\n"
    for i, ex in enumerate(samples):
        context += f"Example {i + 1}:\n"
        context += f"Question: {ex.question}\n"
        context += f"Answer: {ex.answer}\n\n"
    return context


def get_token_usage(lms: list) -> tuple[int, int]:
    total_input = 0
    total_output = 0
    for lm in lms:
        for entry in getattr(lm, "history", []):
            response = entry.get("response", {})
            if response:
                if isinstance(response, dict):
                    usage = response.get("usage", {})
                    if usage:
                        total_input += usage.get("prompt_tokens", 0)
                        total_output += usage.get("completion_tokens", 0)
                elif hasattr(response, "usage") and response.usage:
                    total_input += getattr(response.usage, "prompt_tokens", 0)
                    total_output += getattr(response.usage, "completion_tokens", 0)
    return total_input, total_output


def main():
    load_dotenv()

    NUM_THREADS = int(os.getenv("NUM_THREADS", "8"))  # Lowered to avoid OOM on low-memory server
    ITERATIONS = int(os.getenv("ITERATIONS", "20"))
    NUM_FEWSHOT_EXAMPLES = 5
    VAL_SUBSET_SIZE = 50  # Subset size for fast evaluation

    api_key = os.getenv("OPENROUTER_API_KEY_HOTPOTQA") or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY_HOTPOTQA or OPENROUTER_API_KEY in your environment or .env file."
        )

    qwen_solver_lm = dspy.LM(
        "openrouter/qwen/qwen3-8b",
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
        num_retries=0,
        extra_body={"provider": {"only": ["alibaba"]}},
    )
    qwen_generator_lm = dspy.LM(
        "openrouter/qwen/qwen3-8b",
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
        num_retries=0,
        extra_body={"provider": {"only": ["alibaba"]}},
    )

    # Note: Colbert retriever is automatically configured in hotpot_program.py

    print("Loading HotpotQA dataset | Mode: QWEN | Fast Subset Evaluator")
    bench = HotpotQABench()
    print("init_dataset start...")
    bench.init_dataset()
    print("init_dataset done!")
    trainset = bench.train_set
    valset = bench.val_set
    testset = bench.test_set

    # Fix validation subset for consistent optimization signals. Seeding the
    # global RNG here also fixes the per-iteration demonstration draws.
    random.seed(42)
    val_subset = random.sample(valset, min(VAL_SUBSET_SIZE, len(valset)))
    print("val_subset done!")

    print("wandb.init start...")
    wandb.init(
        project="hotpotqa-dynamic-fewshot",
        name=f"hotpot_fixed_qwen_subset{VAL_SUBSET_SIZE}_{ITERATIONS}iter",
        config={
            "mode": "fixed_qwen",
            "iterations": ITERATIONS,
            "num_threads": NUM_THREADS,
            "num_fewshot": NUM_FEWSHOT_EXAMPLES,
        },
    )
    print("wandb.init done!")

    print("HotpotMultiHop init start...")
    program = HotpotMultiHop()
    print("HotpotMultiHop init done!")

    print("generator init start...")
    generator = dspy.Predict(GeneratePromptsHotpotQA)
    print("generator init done!")

    best_val_score = -1.0
    best_result = None
    best_model_name = None
    start_iteration = 0
    historical_input_tokens = 0
    historical_output_tokens = 0

    checkpoint_file = "checkpoint_qwen.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            start_iteration = checkpoint.get("iteration", 0)
            best_val_score = checkpoint.get("best_val_score", -1.0)
            best_result = checkpoint.get("best_result", None)
            best_model_name = checkpoint.get("best_model_name", None)
            historical_input_tokens = checkpoint.get("historical_input_tokens", 0)
            historical_output_tokens = checkpoint.get("historical_output_tokens", 0)
            print("\n==================================================")
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"Resuming from iteration {start_iteration + 1}")
            print(f"Best validation score so far: {best_val_score:.2f}")
            print("==================================================\n")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    tracked_lms = [qwen_solver_lm, qwen_generator_lm]
    opt_start_input_tokens, opt_start_output_tokens = get_token_usage(tracked_lms)
    prev_input_tokens = opt_start_input_tokens
    prev_output_tokens = opt_start_output_tokens

    for i in range(start_iteration, ITERATIONS):
        solver_lm = qwen_solver_lm
        generator_lm = qwen_generator_lm
        model_name = "qwen"

        print(f"\n--- Iteration {i + 1}/{ITERATIONS} (Model: {model_name}) ---")

        fewshot_context = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)

        # Generate 4 prompts
        dspy.configure(lm=generator_lm, adapter=ChatAdapter(), cache=False)
        try:
            gen_res = generator(dataset_context=fewshot_context)
            s1_p = gen_res.summary1_prompt
            q2_p = gen_res.query2_prompt
            s2_p = gen_res.summary2_prompt
            ans_p = gen_res.answer_prompt
            print(f"Summary1 Prompt: {s1_p[:50]}...")
            print(f"Query2 Prompt: {q2_p[:50]}...")
            print(f"Summary2 Prompt: {s2_p[:50]}...")
            print(f"Answer Prompt: {ans_p[:50]}...")
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        # Inject prompts into the 4 predictors
        try:
            program.summarize1.predictors()[0].signature.instructions = s1_p
            program.create_query_hop2.predictors()[0].signature.instructions = q2_p
            program.summarize2.predictors()[0].signature.instructions = s2_p
            program.final_answer.predictors()[0].signature.instructions = ans_p
        except Exception as e:
            print(f"Prompt injection failed: {e}")
            continue

        # Evaluate on subset
        dspy.configure(lm=solver_lm, adapter=ChatAdapter(), cache=False)
        evaluate = dspy.Evaluate(
            devset=val_subset, metric=dspy.evaluate.answer_exact_match,
            num_threads=NUM_THREADS, display_progress=True
        )
        val_score = float(evaluate(program))

        cumulative_input_tokens, cumulative_output_tokens = get_token_usage(tracked_lms)
        iteration_input_tokens = cumulative_input_tokens - prev_input_tokens
        iteration_output_tokens = cumulative_output_tokens - prev_output_tokens
        opt_input_tokens = cumulative_input_tokens - opt_start_input_tokens + historical_input_tokens
        opt_output_tokens = cumulative_output_tokens - opt_start_output_tokens + historical_output_tokens
        prev_input_tokens = cumulative_input_tokens
        prev_output_tokens = cumulative_output_tokens

        print(f"Validation Score (Subset {VAL_SUBSET_SIZE}): {val_score:.2f}")
        print(
            f"Iteration Tokens: Input={iteration_input_tokens}, Output={iteration_output_tokens}, "
            f"Total={iteration_input_tokens + iteration_output_tokens}"
        )
        print(
            f"Optimization Tokens So Far: Input={opt_input_tokens}, Output={opt_output_tokens}, "
            f"Total={opt_input_tokens + opt_output_tokens}"
        )

        wandb.log({
            "iteration": i + 1,
            "val_score": val_score,
            "model_name": model_name,
            "best_val_score": max(best_val_score, val_score),
            "optimization_iteration_input_tokens": iteration_input_tokens,
            "optimization_iteration_output_tokens": iteration_output_tokens,
            "optimization_iteration_total_tokens": iteration_input_tokens + iteration_output_tokens,
            "optimization_input_tokens_cumulative": opt_input_tokens,
            "optimization_output_tokens_cumulative": opt_output_tokens,
            "optimization_total_tokens_cumulative": opt_input_tokens + opt_output_tokens,
        })

        if val_score > best_val_score:
            best_val_score = val_score
            best_result = {
                "summary1_prompt": s1_p,
                "query2_prompt": q2_p,
                "summary2_prompt": s2_p,
                "answer_prompt": ans_p,
            }
            best_model_name = model_name

        # Save checkpoint
        checkpoint = {
            "iteration": i + 1,
            "best_val_score": best_val_score,
            "best_result": best_result,
            "best_model_name": best_model_name,
            "historical_input_tokens": opt_input_tokens,
            "historical_output_tokens": opt_output_tokens,
        }
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=4)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

        # Clear LM history to prevent OOM from accumulated histories. Baselines
        # reset to 0 so the cumulative math stays consistent with empty history.
        for lm in tracked_lms:
            if hasattr(lm, "history"):
                lm.history.clear()
        prev_input_tokens = 0
        prev_output_tokens = 0
        opt_start_input_tokens = 0
        opt_start_output_tokens = 0
        gc.collect()

    opt_end_input_tokens, opt_end_output_tokens = get_token_usage(tracked_lms)
    optimization_input_tokens = opt_end_input_tokens - opt_start_input_tokens + historical_input_tokens
    optimization_output_tokens = opt_end_output_tokens - opt_start_output_tokens + historical_output_tokens
    print(
        f"\n=== Optimization Token Usage ===\n"
        f"Input Tokens: {optimization_input_tokens}\n"
        f"Output Tokens: {optimization_output_tokens}\n"
        f"Total Tokens: {optimization_input_tokens + optimization_output_tokens}"
    )
    wandb.log({
        "optimization_input_tokens": optimization_input_tokens,
        "optimization_output_tokens": optimization_output_tokens,
        "optimization_total_tokens": optimization_input_tokens + optimization_output_tokens,
    })

    # Final Evaluation on Test Set (top-1: the running argmax candidate)
    if best_result:
        print("\n=== Running Final Evaluation on Test Set ===")
        print(f"Using best model: {best_model_name}")
        program.summarize1.predictors()[0].signature.instructions = best_result["summary1_prompt"]
        program.create_query_hop2.predictors()[0].signature.instructions = best_result["query2_prompt"]
        program.summarize2.predictors()[0].signature.instructions = best_result["summary2_prompt"]
        program.final_answer.predictors()[0].signature.instructions = best_result["answer_prompt"]

        final_solver_lm = qwen_solver_lm

        test_start_input_tokens, test_start_output_tokens = get_token_usage(tracked_lms)
        dspy.configure(lm=final_solver_lm, adapter=ChatAdapter(), cache=False)
        evaluate_test = dspy.Evaluate(
            devset=testset, metric=dspy.evaluate.answer_exact_match,
            num_threads=NUM_THREADS, display_progress=True
        )
        test_score = float(evaluate_test(program))
        test_end_input_tokens, test_end_output_tokens = get_token_usage(tracked_lms)
        test_input_tokens = test_end_input_tokens - test_start_input_tokens
        test_output_tokens = test_end_output_tokens - test_start_output_tokens

        print(f"Final Test Score: {test_score:.2f}")
        print(
            f"Test Tokens: Input={test_input_tokens}, Output={test_output_tokens}, "
            f"Total={test_input_tokens + test_output_tokens}"
        )
        wandb.log({
            "test_score": test_score,
            "best_model": best_model_name,
            "test_input_tokens": test_input_tokens,
            "test_output_tokens": test_output_tokens,
            "test_total_tokens": test_input_tokens + test_output_tokens,
        })

        output_file = "best_prompts_qwen.json"
        with open(output_file, "w") as f:
            json.dump(best_result, f, indent=4)
        print(f"Saved best prompts to {output_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
