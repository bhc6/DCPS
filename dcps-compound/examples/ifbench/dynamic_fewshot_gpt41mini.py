"""Dynamic few-shot prompt search baseline for IFBench — GPT-4.1-mini variant.

Same pipeline as ``examples/ifbench/dynamic_fewshot.py`` (qwen3-8b) but
with solver/generator switched to ``openrouter/openai/gpt-4.1-mini``, using
the paper-aligned ``gpt-41-mini-openrouter`` config from
``gepa-artifact/scripts/run_ifbench_api.sh``:

  temperature=1.0 (no top_p/top_k/provider pin).

Dataset split, fixed validation-pool strategy, program and metric are
identical to the qwen3-8b variant — every change is confined to the LM
configuration so results are directly comparable across model backbones.
"""

import argparse
import os
import random

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

from examples.ifbench.artifact_aligned import (
    IFBenchCoT2StageProgram,
    build_ifbench_program_two_stage,
    ifbench_metric,
    load_ifbench_dataset,
    parse_two_stage_prompt,
    two_stage_output_format,
)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _build_solver(stage1: str, stage2: str) -> IFBenchCoT2StageProgram:
    """Return a fresh two-stage IFBench program whose predictors use
    ``stage1`` (drafter) and ``stage2`` (finalizer) instructions, matching
    the GEPA optimization surface of optimizing the two predictor instructions
    independently."""
    return build_ifbench_program_two_stage(stage1, stage2)


def evaluate_on_dataset(stage1: str, stage2: str, dataset, num_threads: int = 16) -> float:
    """Evaluate a candidate ``(stage1, stage2)`` instruction pair using the official IFBench metric."""
    solver = _build_solver(stage1, stage2)
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=ifbench_metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )
    result = evaluator(solver)
    return result.score / 100.0


# ---------------------------------------------------------------------------
# Few-shot sampling & metaprompt (identical to qwen3-8b variant)
# ---------------------------------------------------------------------------


def sample_fewshot_examples(trainset, num_examples: int = 3) -> str:
    sampled = random.sample(trainset, min(num_examples, len(trainset)))
    out = ""
    for i, ex in enumerate(sampled, 1):
        out += f"Example {i}:\n"
        out += f"Prompt: {ex.prompt}\n"
        out += f"Instruction IDs: {ex.instruction_id_list}\n\n"
    return out.strip()


def create_metaprompt(fewshot_examples: str) -> str:
    return f"""You are an expert prompt engineer for AI systems.

Based on the few-shot examples below, design TWO system prompts for a two-stage assistant:

[Stage 1 — Drafter]
Writes an initial answer to the user's query (input: query; output: response).

[Stage 2 — Finalizer]
Takes the query AND the drafter's response, produces the final response that will be scored (inputs: query, response; output: final_response).

Each prompt should:

1. Identify patterns or strategies shown in the examples.
2. Encourage careful reasoning before producing the stage's output.
3. Improve accuracy and adherence to user requests.
4. Be general enough to apply to new examples from the same task distribution.
5. Avoid copying any specific example content into the prompt.

Output format (exact, no extra commentary, no code fences):

{two_stage_output_format()}

Here are the few-shot examples to analyse:

{fewshot_examples}

Now, generate the two prompts following the output format above:"""


def generate_prompt_pair_with_llm(metaprompt: str) -> tuple[str, str] | None:
    """Ask the generator LM to emit a ``(stage1, stage2)`` instruction pair.

    Returns ``None`` if generation fails or the output cannot be parsed into
    the two tagged sections.
    """
    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(
            desc="The metaprompt for generating a pair of IFBench constraint-following system prompts, one per stage."
        )
        generated_prompt = dspy.OutputField(
            desc="Two system prompts in the required two-stage tagged format, to be installed as the instructions for the two predictors of the artifact IFBench two-stage program."
        )

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    try:
        result = generator(metaprompt=metaprompt)
    except Exception as exc:
        print(f"[generate_prompt_pair_with_llm] generator failed: {exc!r}")
        return None
    return parse_two_stage_prompt(result.generated_prompt or "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Dynamic few-shot prompt search for IFBench (gpt-4.1-mini)")
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--num-fewshot", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--val-sample-size", type=int, default=30,
        help="Fixed validation pool size (head-slice of paper valset). Reused across all iterations.",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # --- Configuration ---
    NUM_ITERATIONS = args.num_iterations
    NUM_FEWSHOT_EXAMPLES = args.num_fewshot
    TOP_K = args.top_k
    VAL_SAMPLE_SIZE = args.val_sample_size
    NUM_THREADS = args.num_threads
    SOLVER_MODEL = "openrouter/openai/gpt-4.1-mini"
    GENERATOR_MODEL = "openrouter/openai/gpt-4.1-mini"
    API_KEY_ENV = "OPENROUTER_API_KEY_IFBENCH"

    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            f"Set {API_KEY_ENV} or OPENROUTER_API_KEY in your environment or .env file."
        )

    # Paper-aligned gpt-4.1-mini sampling (gepa-artifact run_ifbench_api.sh):
    # temperature=1.0. No top_p/top_k/provider pin.
    solver_lm = dspy.LM(
        SOLVER_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=16384,
        num_retries=0,
        cache=False,
    )
    generator_lm = dspy.LM(
        GENERATOR_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=16384,
        num_retries=0,
        cache=False,
    )
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )

    # --- Load data (artifact-aligned split, NOT modified) ---
    trainset, full_valset, testset = load_ifbench_dataset()

    # --- Fixed validation pool ---
    valset = full_valset[:VAL_SAMPLE_SIZE]
    print(f"[OK] Fixed validation pool: {len(valset)} / {len(full_valset)} (head-slice of paper valset)")

    wandb.init(
        project="ifbench-dynamic-fewshot",
        name=f"dynamic_fewshot_gpt41mini_{NUM_ITERATIONS}iter_{NUM_FEWSHOT_EXAMPLES}shot",
        config={
            "task": "ifbench",
            "num_iterations": NUM_ITERATIONS,
            "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
            "top_k_prompts": TOP_K,
            "val_sample_size": VAL_SAMPLE_SIZE,
            "solver_model": SOLVER_MODEL,
            "generator_model": GENERATOR_MODEL,
            "metaprompt_style": "generic_fewshot_only",
            "sampling_temperature": solver_lm.kwargs.get("temperature"),
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "trainset_size": len(trainset),
            "full_valset_size": len(full_valset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "val_strategy": "fixed_head_slice_of_paper_valset",
            "split_alignment": "identical_to_gepa_artifact",
            "data_source": "gepa-artifact/gepa_artifact/benchmarks/IFBench/data",
            "metric": "gepa_artifact.benchmarks.IFBench.ifbench_metric.metric_with_feedback",
            "api_key_env": API_KEY_ENV,
        },
    )

    print("Starting dynamic few-shot prompt generation (IFBench, gpt-4.1-mini)...")
    print(f"Iterations: {NUM_ITERATIONS}, Few-shot: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}")
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    all_results = []
    best_val_score = 0.0
    prompt_table_cols = ["iteration", "val_score", "stage1", "stage2", "fewshot_examples"]
    prompt_table_rows: list[list] = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)
        print(f"Sampled {NUM_FEWSHOT_EXAMPLES} examples from training set")

        metaprompt = create_metaprompt(fewshot_examples)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=generator_lm,
        )
        prompt_pair = generate_prompt_pair_with_llm(metaprompt)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=solver_lm,
        )

        if prompt_pair is None:
            print("[iteration] generator produced unparseable output, scoring 0.")
            stage1, stage2 = "", ""
            val_score = 0.0
        else:
            stage1, stage2 = prompt_pair
            print(f"Generated stage1: {stage1[:200]}...")
            print(f"Generated stage2: {stage2[:200]}...")
            try:
                val_score = evaluate_on_dataset(stage1, stage2, valset, num_threads=NUM_THREADS)
            except Exception as exc:
                print(f"[iteration] evaluation failed: {exc!r}; scoring 0.")
                val_score = 0.0
        print(f"Validation score (on {len(valset)} fixed samples): {val_score:.2%}")

        best_val_score = max(best_val_score, val_score)

        prompt_table_rows.append([iteration + 1, val_score, stage1, stage2, fewshot_examples])
        wandb.log({
            "iteration": iteration + 1,
            "val_score": val_score,
            "best_val_score": best_val_score,
            "iteration_prompts": wandb.Table(columns=prompt_table_cols, data=list(prompt_table_rows)),
        })

        all_results.append({
            "iteration": iteration + 1,
            "stage1": stage1,
            "stage2": stage2,
            "val_score": val_score,
            "fewshot_examples": fewshot_examples,
        })

    all_results.sort(key=lambda x: x["val_score"], reverse=True)
    top_results = all_results[:TOP_K]

    print(f"\n=== Top {TOP_K} Prompt Pairs from Validation ===")
    for i, r in enumerate(top_results):
        print(f"\nRank {i + 1} (Iteration {r['iteration']}): Val={r['val_score']:.2%}")
        print(f"Stage1: {r['stage1'][:200]}...")
        print(f"Stage2: {r['stage2'][:200]}...")

    top_k_val_table = wandb.Table(
        columns=["rank", "iteration", "val_score", "stage1", "stage2"],
        data=[[i + 1, r["iteration"], r["val_score"], r["stage1"], r["stage2"]] for i, r in enumerate(top_results)],
    )
    wandb.log({"top_k_validation": top_k_val_table})

    print("\n=== Final Testing on Test Set ===")
    final_results = []
    final_cols = ["rank", "iteration", "val_score", "test_score", "val_test_gap", "stage1", "stage2"]
    final_rows: list[list] = []

    for i, r in enumerate(top_results):
        print(f"\nTesting Rank {i + 1} on test set...")
        try:
            test_score = evaluate_on_dataset(r["stage1"], r["stage2"], testset, num_threads=NUM_THREADS)
        except Exception as exc:
            print(f"  [test] evaluation failed: {exc!r}; scoring 0.")
            test_score = 0.0
        print(f"  Test Score: {test_score:.2%}")

        final_rows.append([
            i + 1, r["iteration"], r["val_score"],
            test_score, r["val_score"] - test_score, r["stage1"], r["stage2"],
        ])
        wandb.log({
            "test_rank": i + 1,
            "test_score": test_score,
            "test_val_score": r["val_score"],
            "val_test_gap": r["val_score"] - test_score,
            "final_results": wandb.Table(columns=final_cols, data=list(final_rows)),
        })
        final_results.append({
            "rank": i + 1,
            "iteration": r["iteration"],
            "stage1": r["stage1"],
            "stage2": r["stage2"],
            "val_score": r["val_score"],
            "test_score": test_score,
        })

    print("\n=== FINAL REPORT ===")
    print(f"Generated {NUM_ITERATIONS} (stage1, stage2) prompt pairs using dynamic few-shot sampling")
    print(f"Selected top {TOP_K} based on validation performance")
    print(f"Tested on {len(testset)} test examples\n")

    print("Rank | Iter | Val Score | Test Score | Stage1")
    print("-" * 100)
    for r in final_results:
        print(f"{r['rank']:4} | {r['iteration']:4} | {r['val_score']:9.2%} | "
              f"{r['test_score']:10.2%} | {r['stage1'][:30]}...")

    best = max(final_results, key=lambda x: x["test_score"])
    avg_val = sum(r["val_score"] for r in final_results) / len(final_results)
    avg_test = sum(r["test_score"] for r in final_results) / len(final_results)

    print(f"\nBest: Iter {best['iteration']}, Val={best['val_score']:.2%}, Test={best['test_score']:.2%}")
    print(f"Best stage1: {best['stage1']}")
    print(f"Best stage2: {best['stage2']}")
    print(f"\nAvg Val (Top {TOP_K}): {avg_val:.2%}, Avg Test: {avg_test:.2%}, Gap: {avg_val - avg_test:.2%}")

    if wandb.run is not None:
        wandb.run.summary.update({
            "best_test_score": best["test_score"],
            "best_val_score": best["val_score"],
            "best_iteration": best["iteration"],
            "best_stage1": best["stage1"],
            "best_stage2": best["stage2"],
            "avg_val_score_top_k": avg_val,
            "avg_test_score_top_k": avg_test,
            "val_test_gap": avg_val - avg_test,
            "total_iterations": NUM_ITERATIONS,
            "optimization_surface": "two_predictor_instructions",
        })

    wandb.finish()


if __name__ == "__main__":
    main()
