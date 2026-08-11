"""
Artifact-aligned AIME baseline test (DSPy) with ``openrouter/openai/gpt-4.1-mini``.

GPT-4.1-mini variant of ``baseline_test.py``. Uses a dedicated API key
(``OPENROUTER_API_KEY_AIME_MATH_GPT41MINI``) for rigorous paper-style
controlled comparison against the optimized prompts.
"""

import os

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

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    # ---- Configuration ----
    SOLVER_MODEL = "openrouter/openai/gpt-4.1-mini"
    API_KEY_ENV = "OPENROUTER_API_KEY_AIME_MATH_GPT41MINI"

    baseline_instruction = artifact_default_instruction()

    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} or OPENROUTER_API_KEY in your environment or .env file.")

    solver_lm = dspy.LM(
        SOLVER_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        num_retries=0,
        cache=False,
    )

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
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "num_retries": solver_lm.kwargs.get("num_retries"),
            "api_key_env": API_KEY_ENV,
        },
    )

    # ---- Evaluate artifact default instruction on the full test set (avg@5) ----
    print(f"Evaluating artifact default instruction on test set ({len(testset)} examples, avg@5)...")
    print(f"Instruction: {baseline_instruction}\n")

    test_score = evaluate_on_dataset(baseline_instruction, testset)
    print(f"\n=== Baseline Result ===")
    print(f"Test Score (avg@5): {test_score:.2%}")
    print(f"Instruction: {baseline_instruction}")

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
