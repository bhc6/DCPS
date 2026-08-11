"""HoVer original DSPy baseline test.

Evaluates the unoptimised artifact ``HoverMultiHop`` 4-predictor
multi-hop retrieval program on the artifact-aligned HoVer 3-hop test
set, with the four predictors using their default DSPy ChainOfThought
instructions (i.e. the auto-generated ``Given the fields ..., produce
the fields ...`` instruction strings).

Dataset, program schema, and metric are loaded through
``examples.hover.artifact_aligned`` so they match ``gepa-artifact``.

API key priority:
  OPENROUTER_API_KEY_HOVER_BASE -> OPENROUTER_API_KEY
"""

import argparse
import os

import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

import dspy

from examples.hover.artifact_aligned import (
    STAGE_NAMES,
    artifact_default_instructions,
    build_hover_program,
    hover_metric,
    load_hover_dataset,
)


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
    parser = argparse.ArgumentParser(description="HoVer original DSPy baseline test")
    parser.add_argument("--model", choices=["qwen3-8b", "gpt41mini"], required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    api_key_env = "OPENROUTER_API_KEY_HOVER_BASE"
    api_key = os.getenv(api_key_env) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} or OPENROUTER_API_KEY in your environment or .env file.")

    lm = build_lm(args.model, api_key, api_base)
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=lm,
    )

    print("Loading HoVer datasets (artifact-aligned 3-hop split)...")
    trainset, valset, testset = load_hover_dataset()
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    defaults = artifact_default_instructions()
    print("Original baseline instructions (DSPy ChainOfThought defaults):")
    for stage in STAGE_NAMES:
        snippet = defaults[stage].replace("\n", " ")[:120]
        print(f"  {stage}: {snippet}")

    wandb.init(
        project="hover-baseline-test",
        name=f"original_dspy_baseline_{args.model}",
        config={
            "task": "hover",
            "model_key": args.model,
            "model": lm.model,
            "api_key_env": api_key_env,
            "num_threads": args.num_threads,
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "program": "artifact_aligned_HoverMultiHop",
            "data_source": "gepa-artifact/gepa_artifact/benchmarks/hover (HuggingFace 'hover' 3-hop)",
            "metric": "gepa_artifact.benchmarks.hover.hover_utils.discrete_retrieval_eval",
            "stage_instructions": {s: defaults[s] for s in STAGE_NAMES},
            "sampling_temperature": lm.kwargs.get("temperature"),
            "top_p": lm.kwargs.get("top_p"),
            "max_tokens": lm.kwargs.get("max_tokens"),
            "extra_body": lm.kwargs.get("extra_body"),
        },
    )

    program = build_hover_program()
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=hover_metric,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )

    print("\nEvaluating original DSPy baseline on HoVer test set...")
    result = evaluator(program)
    score = result.score / 100.0
    print(f"\nOriginal DSPy baseline test score: {score:.2%}")

    wandb.log({"test_score": score})
    if wandb.run is not None:
        wandb.run.summary.update({"test_score": score})
    wandb.finish()


if __name__ == "__main__":
    main()
