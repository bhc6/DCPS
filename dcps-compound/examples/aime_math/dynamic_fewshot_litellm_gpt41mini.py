"""
Dynamic Few-Shot Prompt Optimization — DSPy-free version using LiteLLM.

GPT-4.1-mini variant of ``dynamic_fewshot_litellm.py``: uses
``openrouter/openai/gpt-4.1-mini`` for both solver and generator and drops
the OpenRouter ``provider`` pin (no longer Qwen-specific).
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
# Dataset loading
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
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[int]:
    """Extract integer answer from model output."""
    if not text:
        return None

    hash_matches = re.findall(r'###\s*(\d+)', text)
    if hash_matches:
        try:
            return int(hash_matches[-1])
        except ValueError:
            pass

    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed_matches:
        raw = boxed_matches[-1]
        raw = re.sub(r'\\text(?:bf|rm|it|sf)?\{([^}]*)\}', r'\1', raw)
        raw = raw.strip().strip('()., ')
        try:
            return int(raw)
        except ValueError:
            pass

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
) -> tuple[float, str]:
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.problem},
        ]
        response = call_llm(messages, model, api_key, api_base, **llm_kwargs)
        predicted = extract_answer(response)
        correct = int(example.answer)
        if predicted is None:
            return 0.0, f"Could not extract answer. Correct: {correct}"
        score = 1.0 if predicted == correct else 0.0
        return score, f"{'Correct' if score else 'Wrong'}. Predicted={predicted}, Correct={correct}"
    except Exception as e:
        return 0.0, f"Error: {e}"


def evaluate_on_dataset(
    system_prompt: str, dataset: list[MathExample],
    model: str, api_key: str, api_base: str,
    num_threads: int = 8, **llm_kwargs,
) -> float:
    scores: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                evaluate_single, ex, system_prompt, model, api_key, api_base, **llm_kwargs
            ): i
            for i, ex in enumerate(dataset)
        }
        correct = 0
        pbar = tqdm(total=len(dataset), desc="Evaluating")
        for future in concurrent.futures.as_completed(futures):
            score, _ = future.result()
            scores.append(score)
            correct += int(score)
            pbar.set_postfix_str(f"{correct}/{len(scores)} ({correct / len(scores):.1%})")
            pbar.update(1)
        pbar.close()
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Prompt generation helpers
# ---------------------------------------------------------------------------

def sample_fewshot_examples(trainset: list[MathExample], num_examples: int = 3) -> str:
    sampled = random.sample(trainset, min(num_examples, len(trainset)))
    parts: list[str] = []
    for i, ex in enumerate(sampled, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Problem: {ex.problem}")
        if ex.solution:
            parts.append(f"Solution: {ex.solution}")
        parts.append(f"Answer: {ex.answer}\n")
    return "\n".join(parts).strip()


def create_metaprompt(fewshot_examples: str) -> str:
    return f"""You are an expert prompt engineer for AI systems.

Based on the few-shot examples below, design an effective prompt that will guide an AI to solve math problems accurately. 

Here are the few-shot examples to analyze:

{fewshot_examples}

Now, generate a prompt that incorporates insights from these examples:"""


def generate_prompt_with_llm(
    metaprompt: str, model: str, api_key: str, api_base: str, **llm_kwargs,
) -> str:
    try:
        return call_llm(
            [{"role": "user", "content": metaprompt}],
            model, api_key, api_base, **llm_kwargs,
        ).strip()
    except Exception as e:
        print(f"[generate_prompt_with_llm] failed: {e!r}")
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    # ---- Configuration ----
    NUM_ITERATIONS = 20
    NUM_FEWSHOT_EXAMPLES = 3
    TOP_K = 5
    VAL_SAMPLE_SIZE = 15
    NUM_THREADS = 8
    SOLVER_MODEL = "openrouter/openai/gpt-4.1-mini"
    GENERATOR_MODEL = "openrouter/openai/gpt-4.1-mini"
    API_KEY_ENV = "OPENROUTER_API_KEY_AIME_MATH_GPT41MINI"

    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} or OPENROUTER_API_KEY in your environment or .env file.")

    llm_kwargs = dict(
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
    )

    # ---- Load data ----
    trainset, valset, testset = load_math_dataset()

    # ---- Init wandb ----
    wandb.init(
        project="aime-math-litellm-agnostic-nb",
        name=f"litellm_gpt41mini_{NUM_ITERATIONS}iter_{NUM_FEWSHOT_EXAMPLES}shot",
        config={
            "num_iterations": NUM_ITERATIONS,
            "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
            "top_k_prompts": TOP_K,
            "val_sample_size": VAL_SAMPLE_SIZE,
            "num_threads": NUM_THREADS,
            "solver_model": SOLVER_MODEL,
            "generator_model": GENERATOR_MODEL,
            "base_prompt": "(none)",
            "framework": "litellm (no DSPy, no base prompt)",
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "testset_size": len(testset),
        },
    )
    wandb.define_metric("iteration")
    wandb.define_metric("val_score", step_metric="iteration")
    wandb.define_metric("best_val_score", step_metric="iteration")
    wandb.define_metric("test_rank")
    wandb.define_metric("test_score", step_metric="test_rank")
    wandb.define_metric("test_val_score", step_metric="test_rank")
    wandb.define_metric("val_test_gap", step_metric="test_rank")

    print("Starting dynamic few-shot prompt generation (LiteLLM, gpt-4.1-mini)...")
    print(f"Iterations: {NUM_ITERATIONS}, Few-shot: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}")
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    print(f"Validation sample size: {VAL_SAMPLE_SIZE}, Threads: {NUM_THREADS}")

    # ---- Optimization loop ----
    all_results: list[dict] = []
    best_val_score = 0.0
    prompt_table_cols = ["iteration", "val_score", "best_val_score", "prompt", "metaprompt", "fewshot_examples"]
    prompt_table_rows: list[list] = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)
        metaprompt = create_metaprompt(fewshot_examples)
        generated_prompt = generate_prompt_with_llm(
            metaprompt, GENERATOR_MODEL, api_key, api_base, **llm_kwargs,
        )
        print(f"Generated prompt: {generated_prompt[:200]}...")

        val_sample = random.sample(valset, min(VAL_SAMPLE_SIZE, len(valset)))
        if not generated_prompt:
            print("[iteration] empty prompt, scoring 0.")
            val_score = 0.0
        else:
            try:
                val_score = evaluate_on_dataset(
                    generated_prompt, val_sample,
                    SOLVER_MODEL, api_key, api_base,
                    num_threads=NUM_THREADS, **llm_kwargs,
                )
            except Exception as e:
                print(f"[iteration] evaluation failed: {e!r}; scoring 0.")
                val_score = 0.0

        print(f"Validation score ({len(val_sample)} samples): {val_score:.2%}")
        best_val_score = max(best_val_score, val_score)

        prompt_table_rows.append([
            iteration + 1, val_score, best_val_score,
            generated_prompt, metaprompt, fewshot_examples,
        ])
        wandb.log({
            "iteration": iteration + 1,
            "val_score": val_score,
            "best_val_score": best_val_score,
        })
        all_results.append({
            "iteration": iteration + 1,
            "prompt": generated_prompt,
            "val_score": val_score,
            "metaprompt": metaprompt,
            "fewshot_examples": fewshot_examples,
        })

    wandb.log({"iteration_prompts": wandb.Table(columns=prompt_table_cols, data=prompt_table_rows)})

    # ---- Select top-K and test ----
    all_results.sort(key=lambda x: x["val_score"], reverse=True)
    top_results = all_results[:TOP_K]

    print(f"\n=== Top {TOP_K} Prompts from Validation ===")
    for i, r in enumerate(top_results):
        print(f"\nRank {i+1} (Iter {r['iteration']}): Val={r['val_score']:.2%}")
        print(f"Prompt: {r['prompt'][:100]}...")

    wandb.log({
        "top_k_validation": wandb.Table(
            columns=["rank", "iteration", "val_score", "prompt"],
            data=[[i + 1, r["iteration"], r["val_score"], r["prompt"]] for i, r in enumerate(top_results)],
        )
    })

    print(f"\n=== Final Testing (avg@5 via replicated testset) ===")
    final_results: list[dict] = []
    final_cols = ["rank", "iteration", "val_score", "test_score", "val_test_gap", "prompt"]
    final_rows: list[list] = []

    for i, r in enumerate(top_results):
        print(f"\nTesting Rank {i+1}...")
        try:
            test_score = evaluate_on_dataset(
                r["prompt"], testset,
                SOLVER_MODEL, api_key, api_base,
                num_threads=NUM_THREADS, **llm_kwargs,
            )
        except Exception as e:
            print(f"  [test] failed: {e!r}; scoring 0.")
            test_score = 0.0
        print(f"  Test Score (avg@5): {test_score:.2%}")

        final_rows.append([
            i + 1, r["iteration"], r["val_score"],
            test_score, r["val_score"] - test_score, r["prompt"],
        ])
        wandb.log({
            "test_rank": i + 1,
            "test_score": test_score,
            "test_val_score": r["val_score"],
            "val_test_gap": r["val_score"] - test_score,
        })
        final_results.append({
            "rank": i + 1, "iteration": r["iteration"],
            "prompt": r["prompt"], "val_score": r["val_score"], "test_score": test_score,
        })

    wandb.log({"final_results": wandb.Table(columns=final_cols, data=final_rows)})

    # ---- Final report ----
    print(f"\n=== FINAL REPORT ===")
    print(f"Generated {NUM_ITERATIONS} prompts, selected top {TOP_K}")
    print(f"Tested on {len(testset)} examples (avg@5)\n")

    print("Rank | Iter | Val Score | Test Score | Prompt")
    print("-" * 80)
    for r in final_results:
        print(f"{r['rank']:4} | {r['iteration']:4} | {r['val_score']:9.2%} | {r['test_score']:10.2%} | {r['prompt'][:30]}...")

    best = max(final_results, key=lambda x: x["test_score"])
    print(f"\n🏆 Best: Iter {best['iteration']}, Val={best['val_score']:.2%}, Test={best['test_score']:.2%}")
    print(f"Prompt: {best['prompt']}")

    avg_val = sum(r["val_score"] for r in final_results) / len(final_results)
    avg_test = sum(r["test_score"] for r in final_results) / len(final_results)
    print(f"\n📊 Avg Val (Top {TOP_K}): {avg_val:.2%}, Avg Test: {avg_test:.2%}, Gap: {avg_val - avg_test:.2%}")

    if wandb.run:
        wandb.run.summary.update({
            "best_test_score": best["test_score"],
            "best_val_score": best["val_score"],
            "best_iteration": best["iteration"],
            "best_prompt": best["prompt"],
            "avg_val_score_top_k": avg_val,
            "avg_test_score_top_k": avg_test,
            "val_test_gap": avg_val - avg_test,
        })
    wandb.finish()


if __name__ == "__main__":
    main()
