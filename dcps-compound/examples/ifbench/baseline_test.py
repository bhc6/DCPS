"""IFBench original DSPy baseline test.

This script evaluates the unoptimised IFBench two-stage DSPy program from the
GEPA artifact, using the original Signature instructions as the baseline prompt:

  1. ``GenerateResponse``:
     "Respond to the query"
  2. ``EnsureCorrectResponse``:
     "Ensure the response is correct and adheres to the given constraints. Your
      response will be used as the final response."

Dataset, program schema, and metric are loaded through
``examples.ifbench.artifact_aligned`` so they match ``gepa-artifact``.

API key priority:
  OPENROUTER_API_KEY_IFBENCH_BASE -> OPENROUTER_API_KEY
"""

import argparse
import os

import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

import dspy

from examples.ifbench.artifact_aligned import IFBenchCoT2StageProgram, ifbench_metric, load_ifbench_dataset


def build_lm(model_name: str, api_key: str, api_base: str | None):
    if model_name == "qwen3-8b":
        return dspy.LM(
            "openrouter/qwen/qwen3-8b",
            api_key=api_key,
            api_base=api_base,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
            num_retries=0,
            cache=False,
            extra_body={
                "top_k": 20,
                "provider": {"only": ["alibaba"]},
            },
        )

    if model_name == "gpt41mini":
        return dspy.LM(
            "openrouter/openai/gpt-4.1-mini",
            api_key=api_key,
            api_base=api_base,
            temperature=1.0,
            max_tokens=16384,
            num_retries=0,
            cache=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="IFBench original DSPy baseline test")
    parser.add_argument("--model", choices=["qwen3-8b", "gpt41mini"], required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    api_key_env = "OPENROUTER_API_KEY_IFBENCH_BASE"
    api_key = os.getenv(api_key_env) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} or OPENROUTER_API_KEY in your environment or .env file.")

    lm = build_lm(args.model, api_key, api_base)
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=lm,
    )

    print("Loading IFBench datasets...")
    trainset, valset, testset = load_ifbench_dataset()
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    print("Original baseline instructions:")
    print("  GenerateResponse: Respond to the query")
    print(
        "  EnsureCorrectResponse: Ensure the response is correct and adheres to the given constraints. "
        "Your response will be used as the final response."
    )

    wandb.init(
        project="ifbench-baseline-test",
        name=f"original_dspy_baseline_{args.model}",
        config={
            "task": "ifbench",
            "model_key": args.model,
            "model": lm.model,
            "api_key_env": api_key_env,
            "num_threads": args.num_threads,
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "program": "artifact_aligned_IFBenchCoT2StageProgram",
            "data_source": "gepa-artifact/gepa_artifact/benchmarks/IFBench/data",
            "metric": "gepa_artifact.benchmarks.IFBench.ifbench_metric.metric_with_feedback",
            "generate_response_instruction": "Respond to the query",
            "ensure_correct_response_instruction": (
                "Ensure the response is correct and adheres to the given constraints. "
                "Your response will be used as the final response."
            ),
            "sampling_temperature": lm.kwargs.get("temperature"),
            "top_p": lm.kwargs.get("top_p"),
            "max_tokens": lm.kwargs.get("max_tokens"),
            "extra_body": lm.kwargs.get("extra_body"),
        },
    )

    program = IFBenchCoT2StageProgram()
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=ifbench_metric,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )

    print("\nEvaluating original DSPy baseline on IFBench test set...")
    result = evaluator(program)
    score = result.score / 100.0
    print(f"\nOriginal DSPy baseline test score: {score:.2%}")

    wandb.log({"test_score": score})
    if wandb.run is not None:
        wandb.run.summary.update({"test_score": score})
    wandb.finish()


if __name__ == "__main__":
    main()
