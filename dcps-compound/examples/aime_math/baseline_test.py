"""
Artifact-aligned AIME baseline test: evaluate the default artifact instruction on the AIME test set.

This provides the "before optimization" reference score so we can measure
the improvement gained by dynamic few-shot prompt generation.
"""

import os
from typing import Any, cast

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

from examples.aime_math.artifact_aligned import (
    artifact_default_instruction,
    evaluate_on_dataset,
    load_aime_dataset,
)


def main():
    load_dotenv()

    # Disable all DSPy caching for fresh results
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # ---- Configuration (matches the optimization experiments) ----
    SOLVER_MODEL = "openrouter/qwen/qwen3-8b"
    API_KEY_ENV = "OPENROUTER_API_KEY"

    baseline_instruction = artifact_default_instruction()

    api_key = os.getenv(API_KEY_ENV)
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} in your environment or .env file.")

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
            "provider": {"only": ["alibaba"]},
        },
    )
    extra_body = cast(dict[str, Any], solver_lm.kwargs.get("extra_body") or {})

    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )

    trainset, valset, testset = load_aime_dataset()

    # ---- Init wandb ----
    wandb.init(
        project="aime-math-baseline",
        name=f"baseline_{SOLVER_MODEL.split('/')[-1]}",
        config={
            "solver_model": SOLVER_MODEL,
            "baseline_instruction": baseline_instruction,
            "instruction_source": "gepa-artifact/gepa_artifact/benchmarks/AIME/AIME_program.py",
            "program": "artifact_aligned_CoT_GenerateResponse",
            "metric": "artifact_aligned_AIME_integer_exact_match",
            "trainset_size": len(trainset),
            "full_valset_size": len(valset),
            "testset_size": len(testset),
            "test_avg_k": len(testset) // (len(testset) // 5) if len(testset) > 0 else 5,
            "temperature": solver_lm.kwargs.get("temperature"),
            "top_p": solver_lm.kwargs.get("top_p"),
            "top_k": extra_body.get("top_k"),
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "num_retries": solver_lm.kwargs.get("num_retries"),
            "provider": extra_body.get("provider"),
        },
    )

    # ---- Evaluate artifact default instruction on the full test set (avg@5) ----
    print(f"Evaluating artifact default instruction on test set ({len(testset)} examples, avg@5)...")
    print(f"Instruction: {baseline_instruction}\n")

    test_score = evaluate_on_dataset(baseline_instruction, testset)
    print("\n=== Baseline Result ===")
    print(f"Test Score (avg@5): {test_score:.2%}")
    print(f"Instruction: {baseline_instruction}")

    # Log results table with prompt
    results_table = wandb.Table(
        columns=["instruction", "test_score", "model", "testset_size"],
        data=[[baseline_instruction, test_score, SOLVER_MODEL, len(testset)]],
    )
    wandb.log({
        "test_score": test_score,
        "results": results_table,
    })

    if wandb.run:
        wandb.run.summary.update({
            "test_score": test_score,
            "baseline_instruction": baseline_instruction,
            "solver_model": SOLVER_MODEL,
        })
    wandb.finish()


if __name__ == "__main__":
    main()
