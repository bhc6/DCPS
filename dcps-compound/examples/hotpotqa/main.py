"""HotpotQA GEPA optimisation — strict paper replication.

Uses dspy.GEPA to optimise the MultiHopQA program's predictor instructions,
with per-predictor textual feedback, BM25 retrieval, and 150/300/300 splits.
"""

import os

import dspy
from dotenv import load_dotenv

from examples.hotpotqa.utils import (
    MultiHopQA,
    hotpotqa_metric,
    init_retriever,
    load_hotpotqa_dataset,
)

# ---------------------------------------------------------------------------
# Configuration — change these to switch models / budget
# ---------------------------------------------------------------------------
SOLVER_LM_MODEL = "openrouter/openai/gpt-4.1-mini"
REFLECTION_LM_MODEL = "openrouter/openai/gpt-5.1"
MAX_METRIC_CALLS = 500
NUM_THREADS = 16


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in your environment or .env file.")

    # --- LMs ---
    solver_lm = dspy.LM(
        SOLVER_LM_MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=0.7,
        max_tokens=3000,
    )
    reflection_lm = dspy.LM(
        REFLECTION_LM_MODEL,
        api_key=api_key,
        api_base=api_base,
        max_tokens=8000,
    )
    dspy.configure(lm=solver_lm)

    # --- Retriever ---
    print("Initialising BM25 retriever...")
    init_retriever()

    # --- Dataset (paper: 150 / 300 / 300) ---
    print("Loading HotpotQA dataset...")
    trainset, valset, testset = load_hotpotqa_dataset()
    print(
        f"Dataset sizes — Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}"
    )

    # --- Run name ---
    solver_short = SOLVER_LM_MODEL.rsplit("/", 1)[-1]
    reflect_short = REFLECTION_LM_MODEL.rsplit("/", 1)[-1]
    run_name = f"{solver_short}-{reflect_short}-{MAX_METRIC_CALLS}"
    log_dir = f"outputs/hotpotqa/{run_name}"

    # --- Baseline evaluation ---
    student = MultiHopQA()
    print("\nEvaluating baseline (unoptimised) on test set...")
    baseline_eval = dspy.Evaluate(
        devset=testset,
        metric=hotpotqa_metric,
        num_threads=NUM_THREADS,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )
    baseline_result = baseline_eval(student)
    baseline_score = baseline_result.score
    print(f"Baseline F1: {baseline_score:.2f}%")

    # --- dspy.GEPA optimisation ---
    print("\nStarting dspy.GEPA optimisation...")
    gepa = dspy.GEPA(
        metric=hotpotqa_metric,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_lm=reflection_lm,
        num_threads=NUM_THREADS,
        log_dir=log_dir,
        track_stats=True,
        track_best_outputs=True,
        use_wandb=True,
        wandb_api_key=os.getenv("WANDB_API_KEY"),
        wandb_init_kwargs={
            "project": "gepa-hotpotqa",
            "name": run_name,
            "config": {
                "task": "hotpotqa",
                "solver_lm": SOLVER_LM_MODEL,
                "reflection_lm": REFLECTION_LM_MODEL,
                "train_size": len(trainset),
                "val_size": len(valset),
                "test_size": len(testset),
                "max_metric_calls": MAX_METRIC_CALLS,
            },
        },
    )

    optimized_program = gepa.compile(
        student=MultiHopQA(),
        trainset=trainset,
        valset=valset,
    )

    # --- Optimised evaluation ---
    print("\nEvaluating optimised program on test set...")
    opt_eval = dspy.Evaluate(
        devset=testset,
        metric=hotpotqa_metric,
        num_threads=NUM_THREADS,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )
    opt_result = opt_eval(optimized_program)
    opt_score = opt_result.score
    print(f"Optimised F1: {opt_score:.2f}%")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Baseline F1:  {baseline_score:.2f}%")
    print(f"Optimised F1: {opt_score:.2f}%")
    print(f"Improvement:  {opt_score - baseline_score:+.2f}%")
    print(f"{'='*60}")

    # Print optimised instructions
    print("\nOptimised predictor instructions:")
    for name, pred in optimized_program.named_predictors():
        print(f"\n--- {name} ---")
        print(pred.signature.instructions)


if __name__ == "__main__":
    main()
