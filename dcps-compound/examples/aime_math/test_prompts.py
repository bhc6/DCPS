import os
import argparse
from typing import List

import dspy
from dotenv import load_dotenv

from examples.aime_math.utils import evaluate_on_dataset, load_math_dataset


def test_prompts(
    prompts: List[str],
    # model: str = "openrouter/qwen/qwen3-8b",
    model: str = "openai/gpt-4.1-mini",
    dataset: str = "test",
    num_samples: int = None,
    temperature: float = 0.6,
    max_tokens: int = 16384,
    top_p: float = 0.95,
    top_k: int = 20,
    num_replicas: int = 1,
    record_history: bool = False,
    output_dir: str = "evaluation_history",
) -> None:
    """Test one or more prompts on AIME math dataset.

    Aligned with paper (gepa-artifact) evaluation:
      - Qwen3-8b: temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384
      - Test set is already replicated 5x inside load_math_dataset() (avg@5).
        A single dspy.Evaluate call over it therefore produces avg@5.
      - num_replicas here is an EXTRA replication on top of that (default 1).

    Args:
        prompts: List of prompt strings to test
        model: Model name to use for evaluation
        dataset: Which dataset to use ('train', 'val', or 'test')
        num_samples: Number of samples to evaluate (None for all)
        temperature: Sampling temperature (paper default 0.6 for Qwen3-8b)
        max_tokens: Max output tokens (paper default 16384)
        top_p: Nucleus sampling (paper default 0.95)
        top_k: Top-k sampling (paper default 20)
        num_replicas: Replication factor for avg@k (paper uses 5)
        record_history: Whether to record detailed evaluation history to JSON files
        output_dir: Directory to save history JSON files (default: evaluation_history)
    """
    load_dotenv()

    # Disable all DSPy caching (disk + memory) for fresh results every run
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment or .env file.")

    # Configure model (paper-aligned sampling params)
    # Paper serves Qwen3-8b locally via vllm with thinking enabled (default, unlimited).
    # We mirror this on OpenRouter:
    #   - Do not override `reasoning` (thinking stays on, no limit).
    #   - Pin providers to avoid flaky routing that yields empty `content`.
    solver_lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_retries=0,
        cache=False,
        extra_body={
            "top_k": top_k,
            # 'provider': {
            #     'only': ['alibaba'],
            # },
        },
    )
    dspy.configure(lm=solver_lm)

    # Load dataset
    trainset, valset, testset = load_math_dataset()
    
    if dataset == "train":
        dataset_to_use = trainset
        dataset_name = "training"
    elif dataset == "val":
        dataset_to_use = valset
        dataset_name = "validation"
    else:  # test
        dataset_to_use = testset
        dataset_name = "test"

    # Limit samples if specified.
    # For test set (5x replicated), truncate by unique problems so every
    # problem keeps exactly 5 copies and avg@5 stays valid.
    if num_samples is not None:
        if dataset == "test":
            # num_samples refers to unique problems; keep 5 copies of each
            unique_problems = num_samples
            dataset_to_use = dataset_to_use[:unique_problems] * 5
            print(f"Using first {unique_problems} unique problems from {dataset_name} set "
                  f"({len(dataset_to_use)} evals with 5x replication)")
        else:
            dataset_to_use = dataset_to_use[:num_samples]
            print(f"Using first {num_samples} samples from {dataset_name} set")
    else:
        print(f"Using full {dataset_name} set ({len(dataset_to_use)} samples)")

    # Built-in 5x replication already done in load_math_dataset() for test set.
    # `num_replicas` here is an EXTRA replication factor on top of that.
    built_in_replicas = 5 if dataset == "test" else 1
    effective_avg_k = built_in_replicas * num_replicas
    if num_replicas > 1:
        unique_size = len(dataset_to_use)
        dataset_to_use = list(dataset_to_use) * num_replicas
        print(f"Extra {num_replicas}x replication: {unique_size} -> {len(dataset_to_use)} evaluations")

    print(f"\n{'='*80}")
    print(f"Model Configuration (paper-aligned):")
    print(f"  Model: {model}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Top-k: {top_k}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Evaluation: avg@{effective_avg_k} "
          f"(built-in {built_in_replicas}x x extra {num_replicas}x = {len(dataset_to_use)} evals)")
    print(f"{'='*80}")

    # Test each prompt once over the (already-replicated) dataset
    results = []
    histories = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Testing Prompt {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print()

        try:
            # Prepare model configuration for history recording
            model_config = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "num_retries": solver_lm.kwargs.get('num_retries', 0),
            }
            
            score, history = evaluate_on_dataset(
                prompt, 
                dataset_to_use,
                record_history=record_history,
                output_dir=output_dir if record_history else None,
                model_config=model_config if record_history else None
            )
            print(f"  Score (avg@{effective_avg_k}): {score:.2%}")
            if history and record_history:
                lm_stats = history.get("lm_statistics", {})
                print(f"  LM Cost: ${lm_stats.get('total_cost', 0):.4f}")
                print(f"  Input Tokens: {lm_stats.get('total_input_tokens', 0)}")
                print(f"  Output Tokens: {lm_stats.get('total_output_tokens', 0)}")
                print(f"  Num Correct: {history.get('num_correct', 0)}/{len(history.get('examples', []))}")
                histories.append(history)
        except Exception as e:
            print(f"  Error: {e}")
            score = 0.0
            history = None
        results.append((prompt, score))

    # Summary
    print(f"\n{'='*80}")
    print("=== SUMMARY ===")
    print()

    for i, (prompt, score) in enumerate(results, 1):
        print(f"Prompt {i}: {score:.2%}")
        print(f"  {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        print()

    if len(results) > 1:
        best_prompt, best_score = max(results, key=lambda x: x[1])
        worst_prompt, worst_score = min(results, key=lambda x: x[1])
        overall_avg = sum(s for _, s in results) / len(results)

        print(f"Best score: {best_score:.2%}")
        print(f"Worst score: {worst_score:.2%}")
        print(f"Overall average score: {overall_avg:.2%}")
        print()
        print("Best prompt:")
        print(best_prompt)


def main():
    parser = argparse.ArgumentParser(
        description="Test one or more prompts on AIME math dataset"
    )
    parser.add_argument(
        "prompts",
        nargs="+",
        help="One or more prompt strings to test (wrap in quotes if containing spaces)"
    )
    parser.add_argument(
        "--model",
        default="openrouter/qwen/qwen3-8b",
        help="Model name to use (default: openrouter/qwen/qwen3-8b)"
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset to use (default: test)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (paper default: 0.6)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max output tokens (paper default: 16384)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p / nucleus sampling (paper default: 0.95)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (paper default: 20)"
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="EXTRA replication factor on top of the built-in 5x in load_math_dataset (default: 1)"
    )
    parser.add_argument(
        "--record-history",
        action="store_true",
        default=False,
        help="Record detailed evaluation history to JSON files (default: False)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_history",
        help="Directory to save history JSON files (default: evaluation_history)"
    )
    args = parser.parse_args()

    test_prompts(
        prompts=args.prompts,
        model=args.model,
        dataset=args.dataset,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        num_replicas=args.num_replicas,
        record_history=args.record_history,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    # Example usage (also works via command line)
    if len(os.sys.argv) == 1:
        # No command line arguments, run example
        example_prompts = [
            "Solve the math problem carefully. Break down the steps and provide the final answer as a single number.",
            "Solve this step by step and give the final numerical answer.",
        ]
        print("No arguments provided. Running example prompts...")
        print("Usage: python test_prompts.py \"prompt1\" \"prompt2\" [--dataset val] [--num-samples 10]")
        print()
        test_prompts(example_prompts, num_samples=5, dataset="val")
    else:
        main()
