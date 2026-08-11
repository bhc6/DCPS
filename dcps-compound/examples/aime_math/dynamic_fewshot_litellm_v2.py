"""
Dynamic Few-Shot Prompt Optimization — v2 (fixed validation), Qwen3-8B.

Key differences vs ``dynamic_fewshot_litellm.py`` (v1):
- Validation set is **fixed** across all iterations (no per-iter random
  resampling), so every candidate prompt is evaluated on the *same*
  problems — fair comparison, lower selection noise.
- Validation set size is capped at ``VAL_SAMPLE_SIZE = 15`` (matches v1).

The train/val split is **identical to the original paper** (and to v1):
  random.Random(0).shuffle(train_split)
  trainset = train_split[: n // 2]
  valset   = train_split[n // 2 :]
The only addition is taking a deterministic head slice of valset (size 15)
as the fixed eval pool — the underlying split is untouched.
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
# Dataset loading (paper-aligned, identical to dynamic_fewshot_litellm.py)
# ---------------------------------------------------------------------------

def load_math_dataset():
    """Load AIME math datasets, returning (trainset, valset, testset).

    Split logic is byte-identical to the original paper / v1 LiteLLM script:
    seed-0 shuffle, 50/50 split of the AIMO validation-AIME train split,
    test set replicated 5x for avg@5.
    """
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
    testset = test_split * 5  # avg@5
    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[int]:
    """Extract integer answer from model output.

    Tries '### <number>' first, then \\boxed{}, then falls back to
    the last standalone integer in the text.
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
) -> tuple[float, str]:
    """Evaluate a single example. Returns (score, feedback)."""
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
    """Evaluate system_prompt on dataset in parallel. Returns accuracy 0-1."""
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
    VAL_SAMPLE_SIZE = 15  # fixed pool size, matches v1 paper-aligned setup
    NUM_THREADS = 8
    SOLVER_MODEL = "openrouter/qwen/qwen3-8b"
    GENERATOR_MODEL = "openrouter/qwen/qwen3-8b"
    API_KEY_ENV = "OPENROUTER_API_KEY_AIME_MATH_V2"

    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} or OPENROUTER_API_KEY in your environment or .env file.")

    llm_kwargs = dict(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "top_k": 20,
            "provider": {"only": ["alibaba"]},
        },
    )

    # ---- Load data (paper-aligned split, NOT modified) ----
    trainset, full_valset, testset = load_math_dataset()

    # ---- Build a FIXED validation pool of size <= VAL_SAMPLE_SIZE ----
    # Deterministic head-slice of the paper-aligned valset. The underlying
    # split is untouched; we just stop resampling per-iteration so candidate
    # prompts are scored on the *same* problems for fair comparison.
    valset = full_valset[:VAL_SAMPLE_SIZE]

    print(f"\n[OK] Fixed validation pool: {len(valset)} problems (head-slice of paper valset)")
    print(f"   Train: {len(trainset)} | Full val: {len(full_valset)} | Test: {len(testset)}")

    # ---- Init wandb ----
    wandb.init(
        project="aime-math-litellm-agnostic-nb",
        name=f"litellm_v2_stratified_{NUM_ITERATIONS}iter_{NUM_FEWSHOT_EXAMPLES}shot",
        config={
            "version": "v2-fixed-val",
            "num_iterations": NUM_ITERATIONS,
            "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
            "top_k_prompts": TOP_K,
            "val_size": len(valset),
            "val_strategy": "fixed_head_slice_of_paper_valset",
            "val_sample_size": VAL_SAMPLE_SIZE,
            "split_alignment": "identical_to_paper_v1",
            "num_threads": NUM_THREADS,
            "solver_model": SOLVER_MODEL,
            "generator_model": GENERATOR_MODEL,
            "base_prompt": "(none)",
            "framework": "litellm (no DSPy, no base prompt, fixed val)",
            "trainset_size": len(trainset),
            "full_valset_size": len(full_valset),
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

    print("\nStarting dynamic few-shot prompt generation (LiteLLM v2, fixed val)...")
    print(f"Iterations: {NUM_ITERATIONS}, Few-shot: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}")
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

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

        # Evaluate on the FIXED validation set (no random sampling here)
        if not generated_prompt:
            print("[iteration] empty prompt, scoring 0.")
            val_score = 0.0
        else:
            try:
                val_score = evaluate_on_dataset(
                    generated_prompt, valset,
                    SOLVER_MODEL, api_key, api_base,
                    num_threads=NUM_THREADS, **llm_kwargs,
                )
            except Exception as e:
                print(f"[iteration] evaluation failed: {e!r}; scoring 0.")
                val_score = 0.0

        print(f"Validation score ({len(valset)} problems, fixed): {val_score:.2%}")
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
