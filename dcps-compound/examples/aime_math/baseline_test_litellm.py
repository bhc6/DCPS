"""
Baseline test (LiteLLM, no DSPy) — evaluate BASE_PROMPT on the AIME test set.

This provides a DSPy-free baseline that uses the same evaluation pipeline as
``dynamic_fewshot_litellm.py``, ensuring a fair comparison:
  * Same model, temperature, top_p, top_k, max_tokens
  * Same dataset split and avg@5 test methodology
  * Same answer extraction (\\boxed{} / ### / fallback)
  * No ChatAdapter — no parsing failures

Configuration is strictly aligned with the original GEPA paper:
  * Model: qwen3-8b (temperature=0.6, top_p=0.95, top_k=20)
  * Dataset: AI-MO/aimo-validation-aime + MathArena/aime_2025 × 5
  * Split: 50/50 train/val, seed=0
"""

import os
import re
import random
import concurrent.futures
from dataclasses import dataclass
from typing import Optional

import litellm
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MathExample:
    problem: str
    answer: str
    solution: str = ""


# ---------------------------------------------------------------------------
# Dataset loading (identical to dynamic_fewshot_litellm.py & original paper)
# ---------------------------------------------------------------------------

def load_math_dataset():
    """Load AIME math datasets, returning (trainset, valset, testset)."""
    train_split = []
    train_raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for item in train_raw:
        train_split.append(MathExample(
            problem=item["problem"],
            answer=item["answer"],
            solution=item.get("solution", ""),
        ))
    random.Random(0).shuffle(train_split)

    test_split = []
    test_raw = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_raw:
        test_split.append(MathExample(problem=item["problem"], answer=item["answer"]))

    n = len(train_split)
    trainset = train_split[: n // 2]
    valset = train_split[n // 2:]
    testset = test_split * 5  # Replicate 5x for avg@5
    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Answer extraction (identical to dynamic_fewshot_litellm.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[int]:
    """Extract integer answer from model output.

    Priority: '### <number>' > \\boxed{} > last integer in text.
    """
    if not text:
        return None

    # Try '### <number>' format
    hash_matches = re.findall(r'###\s*(\d+)', text)
    if hash_matches:
        try:
            return int(hash_matches[-1])
        except ValueError:
            pass

    # Try \boxed{...} — take the last match
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed_matches:
        raw = boxed_matches[-1]
        raw = re.sub(r'\\text(?:bf|rm|it|sf)?\{([^}]*)\}', r'\1', raw)
        raw = raw.strip().strip('()., ')
        try:
            return int(raw)
        except ValueError:
            pass

    # Fallback: last integer in the text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], model: str, api_key: str, api_base: str, **kwargs) -> str:
    """Call LLM via litellm and return the response text."""
    response = litellm.completion(
        model=model, messages=messages,
        api_key=api_key, api_base=api_base,
        **kwargs,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single(
    example: MathExample, system_prompt: str,
    model: str, api_key: str, api_base: str, **llm_kwargs,
) -> tuple[float, str, Optional[int], int]:
    """Evaluate a single example. Returns (score, feedback, predicted, correct)."""
    correct = int(example.answer)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.problem},
        ]
        response = call_llm(messages, model, api_key, api_base, **llm_kwargs)
        predicted = extract_answer(response)
        if predicted is None:
            return 0.0, f"Could not extract answer. Correct: {correct}", None, correct
        score = 1.0 if predicted == correct else 0.0
        return score, f"{'Correct' if score else 'Wrong'}. Predicted={predicted}, Correct={correct}", predicted, correct
    except Exception as e:
        return 0.0, f"Error: {e}", None, correct


def evaluate_on_dataset(
    system_prompt: str, dataset: list[MathExample],
    model: str, api_key: str, api_base: str,
    num_threads: int = 8, **llm_kwargs,
) -> tuple[float, list[dict]]:
    """Evaluate system_prompt on dataset in parallel. Returns (accuracy, per_example_results)."""
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                evaluate_single, ex, system_prompt, model, api_key, api_base, **llm_kwargs
            ): i
            for i, ex in enumerate(dataset)
        }
        correct_count = 0
        pbar = tqdm(total=len(dataset), desc="Evaluating")
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            score, feedback, predicted, correct_ans = future.result()
            results.append({
                "index": idx,
                "score": score,
                "predicted": predicted,
                "correct": correct_ans,
                "feedback": feedback,
            })
            correct_count += int(score)
            pbar.set_postfix_str(f"{correct_count}/{len(results)} ({correct_count / len(results):.1%})")
            pbar.update(1)
        pbar.close()

    accuracy = sum(r["score"] for r in results) / len(results) if results else 0.0
    return accuracy, sorted(results, key=lambda r: r["index"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    # ---- Configuration (strictly aligned with original paper) ----
    SOLVER_MODEL = "openrouter/qwen/qwen3-8b"
    API_KEY_ENV = "OPENROUTER_API_KEY_AIME_MATH_AGNOSTIC_NB"

    BASE_PROMPT = (
        "Solve the problem and provide the answer in the correct format."
    )

    api_key = os.getenv(API_KEY_ENV)
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} in your environment or .env file.")

    llm_kwargs = dict(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "top_k": 20,
            "provider": {"only": ["alibaba"]},
        },
    )

    NUM_THREADS = 8

    # ---- Load data ----
    trainset, valset, testset = load_math_dataset()

    # ---- Init wandb ----
    wandb.init(
        project="aime-math-baseline-litellm",
        name=f"baseline_litellm_{SOLVER_MODEL.split('/')[-1]}",
        config={
            "solver_model": SOLVER_MODEL,
            "base_prompt": BASE_PROMPT,
            "base_prompt_source": "Original paper AIME Signature docstring",
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "test_avg_k": 5,
            "num_threads": NUM_THREADS,
            "framework": "litellm (no DSPy)",
            "answer_extraction": "### > \\boxed{} > last integer",
            **{k: v for k, v in llm_kwargs.items() if k != "extra_body"},
            "top_k": llm_kwargs["extra_body"]["top_k"],
            "provider": llm_kwargs["extra_body"]["provider"],
        },
    )

    # ---- Evaluate on validation set first ----
    print(f"Evaluating BASE_PROMPT on validation set ({len(valset)} examples)...")
    print(f"Prompt: {BASE_PROMPT}\n")

    val_score, val_results = evaluate_on_dataset(
        BASE_PROMPT, valset, SOLVER_MODEL, api_key, api_base,
        num_threads=NUM_THREADS, **llm_kwargs,
    )
    print(f"\nValidation Score: {val_score:.2%}")

    wandb.log({"val_score": val_score})

    # ---- Evaluate on test set (avg@5) ----
    print(f"\nEvaluating BASE_PROMPT on test set ({len(testset)} examples, avg@5)...")

    test_score, test_results = evaluate_on_dataset(
        BASE_PROMPT, testset, SOLVER_MODEL, api_key, api_base,
        num_threads=NUM_THREADS, **llm_kwargs,
    )
    print(f"\n{'=' * 50}")
    print(f"=== Baseline Result (LiteLLM, no DSPy) ===")
    print(f"{'=' * 50}")
    print(f"Validation Score:     {val_score:.2%}")
    print(f"Test Score (avg@5):   {test_score:.2%}")
    print(f"Val-Test Gap:         {val_score - test_score:.2%}")
    print(f"Prompt: {BASE_PROMPT}")

    # ---- Log comprehensive results to wandb ----

    # Per-example results table
    per_example_table = wandb.Table(
        columns=["index", "score", "predicted", "correct", "feedback"],
        data=[[r["index"], r["score"], r["predicted"], r["correct"], r["feedback"]] for r in test_results],
    )

    # Summary results table
    results_table = wandb.Table(
        columns=["prompt", "val_score", "test_score", "val_test_gap", "model", "testset_size", "framework"],
        data=[[
            BASE_PROMPT, val_score, test_score, val_score - test_score,
            SOLVER_MODEL, len(testset), "litellm (no DSPy)",
        ]],
    )

    wandb.log({
        "test_score": test_score,
        "val_test_gap": val_score - test_score,
        "results": results_table,
        "per_example_results": per_example_table,
    })

    if wandb.run:
        wandb.run.summary.update({
            "val_score": val_score,
            "test_score": test_score,
            "val_test_gap": val_score - test_score,
            "base_prompt": BASE_PROMPT,
            "solver_model": SOLVER_MODEL,
            "framework": "litellm (no DSPy)",
        })

    wandb.finish()
    print("\nDone. Results logged to wandb.")


if __name__ == "__main__":
    main()
