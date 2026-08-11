import os
import random
from typing import List

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

from examples.aime_math.utils import evaluate_on_dataset, load_math_dataset, math_metric, run_llm
from gepa.optimize_anything import SideInfo


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate a candidate on a single example."""
    prediction = run_llm(example, candidate)
    score, feedback = math_metric(example, prediction)

    side_info = {
        "score": score,
        "input": example.problem,
        "prompt": candidate,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }

    return score, side_info


def sample_fewshot_examples(trainset: List, num_examples: int = 3) -> str:
    """Randomly sample few-shot examples from training set."""
    sampled_examples = random.sample(trainset, min(num_examples,
                                                   len(trainset)))

    fewshot_text = ""
    for i, example in enumerate(sampled_examples, 1):
        fewshot_text += f"Example {i}:\n"
        fewshot_text += f"Problem: {example.problem}\n"
        if hasattr(example, 'solution') and example.solution:
            fewshot_text += f"Solution: {example.solution}\n"
        fewshot_text += f"Answer: {example.answer}\n\n"

    return fewshot_text.strip()


def create_metaprompt(fewshot_examples: str, base_prompt: str) -> str:
    """Create a metaprompt for generating math problem solving prompts."""
    return f"""You are an expert prompt engineer for AI systems that solve competition math problems. 

Based on the base prompt and few-shot examples below, design an effective prompt that will guide an AI to solve similar math problems accurately. The prompt should:

1. Encourage step-by-step mathematical reasoning
2. Reference the patterns or strategies shown in the examples
3. Focus on problem-solving techniques and accuracy
4. CRITICAL: DO NOT include any specific output formatting instructions (like JSON, XML, or specific headers). The system will automatically handle parsing the output.

Here are the few-shot examples to analyze:

{fewshot_examples}

base prompt:
{base_prompt}

Now, generate a prompt that incorporates insights from these examples:"""


def generate_prompt_with_llm(metaprompt: str) -> str:
    """Use LLM to generate a prompt based on the metaprompt.

    Returns an empty string on any failure so the caller can record a 0-scored
    iteration instead of crashing the whole run.
    """

    # Create a temporary signature for prompt generation
    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(
            desc="The metaprompt for generating a math solving prompt.")
        generated_prompt = dspy.OutputField(
            desc="The generated prompt for solving math problems.")

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    try:
        result = generator(metaprompt=metaprompt)
        return result.generated_prompt or ""
    except Exception as exc:
        print(f"[generate_prompt_with_llm] generator failed: {exc!r}")
        return ""


def main():
    load_dotenv()

    # Disable all DSPy caching (disk + memory) for fresh results every run
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # Configuration
    NUM_ITERATIONS = 20  # Number of prompt generation iterations
    NUM_FEWSHOT_EXAMPLES = 3  # Number of few-shot examples to sample
    TOP_K = 5  # Number of top prompts to test on final test set
    VAL_SAMPLE_SIZE = 15  # Number of validation examples to randomly sample per iteration
    # Note: avg@5 is implemented by replicating the test set 5x in load_math_dataset()
    SOLVER_MODEL = "openrouter/qwen/qwen3-8b"  # Model for solving math problems
    GENERATOR_MODEL = "openrouter/qwen/qwen3-8b"  # Model for generating prompts
    BASE_PROMPT = """You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'."""

    api_key = os.getenv("OPENROUTER_API_KEY_AIME_MATH_FEWSHOT")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY_AIME_MATH_FEWSHOT in your environment or .env file.")

    # Configure solver LM for math problems (Evaluation)
    # Paper-aligned sampling: temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, num_retries=2
    solver_lm = dspy.LM(
        SOLVER_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        num_retries=0,
        extra_body={
            "top_k": 20,
            'provider': {
                'only': ['alibaba'],
            },
        },
    )

    # Configure generator LM for prompt generation
    generator_lm = dspy.LM(
        GENERATOR_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        num_retries=0,
        extra_body={
            "top_k": 20,
            'provider': {
                'only': ['alibaba'],
            },
        },
    )

    # Disable ChatAdapter -> JSONAdapter fallback to avoid Qwen3-8B thinking mode
    # + JSON mode incompatibility (400 error). Failures are caught and scored 0.
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )

    trainset, valset, testset = load_math_dataset()

    # Initialize wandb
    wandb.init(
        project="aime-math-dynamic-fewshot",
        name=f"dynamic_fewshot_{NUM_ITERATIONS}iter_{NUM_FEWSHOT_EXAMPLES}shot",
        config={
            "num_iterations": NUM_ITERATIONS,
            "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
            "top_k_prompts": TOP_K,
            "val_sample_size": VAL_SAMPLE_SIZE,
            "solver_model": SOLVER_MODEL,
            "generator_model": GENERATOR_MODEL,
            "base_prompt": BASE_PROMPT,
            "sampling_temperature": solver_lm.kwargs.get("temperature"),
            "sampling_top_p": solver_lm.kwargs.get("top_p"),
            "sampling_top_k": solver_lm.kwargs.get("extra_body", {}).get("top_k"),
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "test_avg_k": len(testset) // (len(testset) // 5) if len(testset) > 0 else 5,
        },
    )

    # Define custom x-axes so wandb draws line charts correctly
    wandb.define_metric("iteration")
    wandb.define_metric("val_score", step_metric="iteration")
    wandb.define_metric("best_val_score", step_metric="iteration")
    wandb.define_metric("test_rank")
    wandb.define_metric("test_score", step_metric="test_rank")
    wandb.define_metric("test_val_score", step_metric="test_rank")
    wandb.define_metric("val_test_gap", step_metric="test_rank")

    print(f"Starting dynamic few-shot prompt generation...")
    print(
        f"Iterations: {NUM_ITERATIONS}, Few-shot examples: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}"
    )
    print(
        f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}"
    )
    print(f"Validation sample size per iteration: {VAL_SAMPLE_SIZE}")

    # Generate and test prompts
    all_results = []
    best_val_score = 0.0

    # Table to store generated prompts and scores, logged once at the end
    prompt_table_cols = ["iteration", "val_score", "prompt", "fewshot_examples"]
    prompt_table_rows: list[list] = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        # Sample few-shot examples from training set
        fewshot_examples = sample_fewshot_examples(trainset,
                                                   NUM_FEWSHOT_EXAMPLES)
        print(f"Sampled {NUM_FEWSHOT_EXAMPLES} examples from training set")

        # Create metaprompt
        metaprompt = create_metaprompt(fewshot_examples, BASE_PROMPT)

        # Switch to generator LM and generate prompt
        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=generator_lm,
        )
        generated_prompt = generate_prompt_with_llm(metaprompt)

        # Switch back to solver LM for evaluation
        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=solver_lm,
        )

        print(f"Generated prompt: {generated_prompt}")

        # Randomly sample a subset from validation set for faster evaluation.
        # Any per-iteration failure is logged and scored as 0 so the run
        # continues across all NUM_ITERATIONS.
        val_sample = random.sample(valset, min(VAL_SAMPLE_SIZE, len(valset)))
        if not generated_prompt:
            print("[iteration] empty generated prompt, scoring 0.")
            val_score = 0.0
        else:
            try:
                val_score = evaluate_on_dataset(generated_prompt, val_sample)
            except Exception as exc:
                print(f"[iteration] evaluation failed: {exc!r}; scoring 0.")
                val_score = 0.0
        print(
            f"Validation score (on {len(val_sample)} samples): {val_score:.2%}"
        )

        # Track best validation score
        best_val_score = max(best_val_score, val_score)

        prompt_table_rows.append([
            iteration + 1,
            val_score,
            generated_prompt,
            fewshot_examples,
        ])
        
        # Log scalar metrics for real-time line charts
        wandb.log({
            "iteration": iteration + 1,
            "val_score": val_score,
            "best_val_score": best_val_score,
        })

        # Store results
        all_results.append({
            'iteration': iteration + 1,
            'prompt': generated_prompt,
            'val_score': val_score,
            'fewshot_examples': fewshot_examples
        })

    # Log the full iteration table once at the end of the loop
    wandb.log({
        "iteration_prompts": wandb.Table(
            columns=prompt_table_cols,
            data=prompt_table_rows,
        )
    })

    # Sort by validation score and select top K
    all_results.sort(key=lambda x: x['val_score'], reverse=True)
    top_results = all_results[:TOP_K]

    print(f"\n=== Top {TOP_K} Prompts from Validation ===")
    for i, result in enumerate(top_results):
        print(f"\nRank {i+1} (Iteration {result['iteration']}):")
        print(f"Validation Score: {result['val_score']:.2%}")
        print(f"Prompt: {result['prompt']}")

    # Log top-K validation results table
    top_k_val_table = wandb.Table(
        columns=["rank", "iteration", "val_score", "prompt"],
        data=[[i + 1, r["iteration"], r["val_score"], r["prompt"]]
              for i, r in enumerate(top_results)],
    )
    wandb.log({"top_k_validation": top_k_val_table})

    # Final testing on test set (testset is already replicated 5x => single eval = avg@5)
    print(f"\n=== Final Testing on Test Set (avg@5 via replicated testset) ===")
    final_results = []

    # Table for final test results, logged once at the end
    final_cols = [
        "rank", "iteration", "val_score", "test_score", "val_test_gap", "prompt",
    ]
    final_rows: list[list] = []

    for i, result in enumerate(top_results):
        print(f"\nTesting Rank {i+1} on test set...")
        try:
            test_score = evaluate_on_dataset(result['prompt'], testset)
        except Exception as exc:
            print(f"  [test] evaluation failed: {exc!r}; scoring 0.")
            test_score = 0.0
        print(f"  Test Score (avg@5): {test_score:.2%}")

        final_rows.append([
            i + 1,
            result["iteration"],
            result["val_score"],
            test_score,
            result["val_score"] - test_score,
            result["prompt"],
        ])

        # Log scalar metrics for real-time line charts
        wandb.log({
            "test_rank": i + 1,
            "test_score": test_score,
            "test_val_score": result["val_score"],
            "val_test_gap": result["val_score"] - test_score,
        })

        final_results.append({
            'rank': i + 1,
            'iteration': result['iteration'],
            'prompt': result['prompt'],
            'val_score': result['val_score'],
            'test_score': test_score,
        })

    # Log the final test results table once
    wandb.log({
        "final_results": wandb.Table(columns=final_cols, data=final_rows)
    })

    # Final report
    print(f"\n=== FINAL REPORT ===")
    print(
        f"Generated {NUM_ITERATIONS} prompts using dynamic few-shot sampling")
    print(f"Selected top {TOP_K} based on validation performance")
    print(f"Tested on {len(testset)} test examples (avg@5 via replication)\n")

    print("Rank | Iteration | Val Score | Test Score (avg@5) | Prompt")
    print("-" * 100)
    for result in final_results:
        print(
            f"{result['rank']:4} | {result['iteration']:9} | {result['val_score']:9.2%} | {result['test_score']:18.2%} | {result['prompt'][:30]}..."
        )

    # Best overall performance
    best_result = max(final_results, key=lambda x: x['test_score'])
    print(f"\n🏆 Best Overall Performance:")
    print(f"Iteration: {best_result['iteration']}")
    print(f"Validation Score: {best_result['val_score']:.2%}")
    print(f"Test Score (avg@5): {best_result['test_score']:.2%}")
    print(f"Prompt: {best_result['prompt']}")

    # Performance summary
    avg_val_score = sum(r['val_score'] for r in final_results) / len(final_results)
    avg_test_score = sum(r['test_score'] for r in final_results) / len(final_results)
    avg_gap = avg_val_score - avg_test_score

    print(f"\n📊 Performance Summary:")
    print(f"Average Validation Score (Top {TOP_K}): {avg_val_score:.2%}")
    print(f"Average Test Score avg@5 (Top {TOP_K}): {avg_test_score:.2%}")
    print(f"Validation-to-Test Gap: {avg_gap:.2%}")

    # Log summary metrics
    if wandb.run is not None:
        wandb.run.summary.update({
            "best_test_score": best_result["test_score"],
            "best_val_score": best_result["val_score"],
            "best_iteration": best_result["iteration"],
            "best_prompt": best_result["prompt"],
            "avg_val_score_top_k": avg_val_score,
            "avg_test_score_top_k": avg_test_score,
            "val_test_gap": avg_val_score - avg_test_score,
            "total_iterations": NUM_ITERATIONS,
        })

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
