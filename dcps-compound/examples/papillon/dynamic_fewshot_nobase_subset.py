"""DCPS (Demonstration-Conditioned Prompt Search) for PUPA / Papillon.

Faithful port of the run script from the paper artifact
(``dcps-artifact/examples/papillon/dynamic_fewshot_nobase_subset.py``), which
produced the paper's DCPS-Compound PUPA cells:

    pupa_gpt_nobase_subset45_20iter   -> test 94.09  (run p16uyuga)
    pupa_qwen_nobase_subset45_20iter  -> test 90.02  (run n7o1crf6)

Mechanism (the whole thing the audit isolates): propose -> evaluate -> argmax.
One ``dspy.ChainOfThought`` call emits BOTH stage prompts, conditioned only on
randomly sampled demonstrations. No base prompt ("nobase"), no reflection, no
textual gradient, no evolutionary merge. Selection is argmax of validation score
with TOP_K = 1, so exactly one candidate is scored on test.

Rollouts = (45 val + 1 test) x 20 iterations = 920.

Backbone is chosen by ``EXP_MODE`` (``qwen`` | ``gpt``). Note that unlike the
GEPA artifact's fixed-judge setup, here the untrusted external LLM and the LLM
judge follow the backbone: in qwen mode BOTH are qwen3-8b, not gpt-4.1-mini.
That is what the paper's runs did, so it is preserved here.

Two deviations from the archived script, neither affecting results:
  * ``gepa-artifact-origin`` -> resolved via _artifact_root() (this repo vendors
    the benchmark as ``gepa-artifact``).
  * The archived script drew its four LM keys from three unrelated env vars
    (``OPENROUTER_API_KEY_PUPA``, ``..._AIME_MATH_AGNOSTIC``,
    ``..._AIME_MATH_AGNOSTIC_NB``) purely to spread OpenRouter rate limits. Here
    each role reads its own optional var and falls back to a single key.

Run:
    EXP_MODE=gpt  uv run python -m examples.papillon.dynamic_fewshot_nobase_subset
    EXP_MODE=qwen uv run python -m examples.papillon.dynamic_fewshot_nobase_subset
"""

import os
import random
import sys
from pathlib import Path
from typing import List

import dspy
import wandb
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter


def _artifact_root() -> Path:
    """Locate the vendored GEPA benchmark suite (``gepa-artifact[-origin]``)."""
    here = Path(__file__).resolve()
    for base in (Path.cwd(), *here.parents):
        for name in ("gepa-artifact-origin", "gepa-artifact"):
            candidate = base / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "gepa-artifact not found. Clone https://github.com/gepa-ai/gepa-artifact "
        "into the repo root."
    )


sys.path.insert(0, str(_artifact_root()))

from gepa_artifact.benchmarks.papillon.papillon_data import Papillon  # noqa: E402
from gepa_artifact.benchmarks.papillon.papillon_program import PAPILLON  # noqa: E402
from gepa_artifact.benchmarks.papillon.papillon_utils import (  # noqa: E402
    compute_overall_score,
    llm_judge,
)


def sample_fewshot_examples(trainset: List, num_examples: int = 3) -> str:
    """Sample examples from PUPA training set for the metaprompt."""
    sampled_examples = random.sample(trainset, min(num_examples, len(trainset)))

    fewshot_text = ""
    for i, example in enumerate(sampled_examples, 1):
        fewshot_text += f"Example {i}:\n"
        fewshot_text += f"User Query: {example.user_query}\n"
        fewshot_text += f"Target Response: {example.target_response}\n"
        fewshot_text += f"PII to protect: {example.pii_str}\n\n"

    return fewshot_text.strip()


def create_metaprompt(fewshot_examples: str) -> str:
    """Create a metaprompt for generating redaction and response prompts (No Base Prompt)."""
    return f"""You are an expert prompt engineer specializing in privacy-preserving AI.
Your task is to generate system prompts for two crucial agents in a privacy pipeline:

1. **Redaction Agent**: Takes a private user query and creates a redacted version for an external LLM.
2. **Response Agent**: Takes the external LLM's response, the original private query, and the redacted request to provide the final answer.

Based on the examples below, design effective system prompts for BOTH agents.

### Few-shot Examples (Query -> Target Response/PII):
{fewshot_examples}

Now, generate concise and powerful system prompts for both agents:"""


def generate_prompts_with_llm(metaprompt: str) -> tuple[str, str]:
    """Use LLM to generate two prompts based on the metaprompt."""

    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(desc="Instructions and examples.")
        redaction_prompt = dspy.OutputField(desc="Optimized system prompt for the Redaction Agent.")
        response_prompt = dspy.OutputField(desc="Optimized system prompt for the Response Agent.")

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    try:
        result = generator(metaprompt=metaprompt)
        return (result.redaction_prompt or "", result.response_prompt or "")
    except Exception as exc:
        print(f"[generate_prompts_with_llm] failed: {exc!r}")
        return ("", "")


def get_token_usage(lms: list) -> tuple[int, int]:
    """Sum up input and output tokens from a list of DSPy LM objects."""
    total_input = 0
    total_output = 0
    for lm in lms:
        for entry in lm.history:
            response = entry.get("response", {})
            if response:
                # Handle cases where response is a dict (OpenRouter/LiteLLM style)
                if isinstance(response, dict):
                    usage = response.get("usage", {})
                    if usage:
                        total_input += usage.get("prompt_tokens", 0)
                        total_output += usage.get("completion_tokens", 0)
                # Handle cases where response is an object (standard OpenAI style)
                elif hasattr(response, "usage") and response.usage:
                    total_input += getattr(response.usage, "prompt_tokens", 0)
                    total_output += getattr(response.usage, "completion_tokens", 0)
    return total_input, total_output


def main():
    load_dotenv()

    # === Configuration (as in the paper runs) ===
    NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "20"))
    NUM_FEWSHOT_EXAMPLES = 3
    NUM_THREADS = int(os.getenv("NUM_THREADS", "32"))
    VAL_SUBSET_SIZE = 45  # Speed up iterations

    # Mode Selection: "qwen" or "gpt"
    MODE = os.getenv("EXP_MODE", "qwen").lower()

    # The untrusted external LLM and the judge follow the backbone (see module docstring).
    if MODE == "qwen":
        SOLVER_MODEL = "openrouter/qwen/qwen3-8b"
        GENERATOR_MODEL = "openrouter/qwen/qwen3-8b"
        UNTRUSTED_MODEL = "openrouter/qwen/qwen3-8b"
        JUDGE_MODEL = "openrouter/qwen/qwen3-8b"
        COMMON_KWARGS = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
            "extra_body": {"provider": {"only": ["alibaba"]}},
        }
    else:
        SOLVER_MODEL = "openrouter/openai/gpt-4.1-mini"
        GENERATOR_MODEL = "openrouter/openai/gpt-4.1-mini"
        UNTRUSTED_MODEL = "openrouter/openai/gpt-4.1-mini"
        JUDGE_MODEL = "openrouter/openai/gpt-4.1-mini"
        COMMON_KWARGS = {"temperature": 1.0, "max_tokens": 16384}

    # One key per role (optional); all fall back to a single OpenRouter key. The
    # archived script borrowed unrelated per-task keys to spread rate limits.
    DEFAULT_KEY = os.getenv("OPENROUTER_API_KEY_PUPA") or os.getenv("OPENROUTER_API_KEY")
    if not DEFAULT_KEY:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY_PUPA or OPENROUTER_API_KEY in your environment or .env file."
        )
    KEY_SOLVER = os.getenv("OPENROUTER_API_KEY_PUPA_SOLVER") or DEFAULT_KEY
    KEY_GENERATOR = os.getenv("OPENROUTER_API_KEY_PUPA_GENERATOR") or DEFAULT_KEY
    KEY_JUDGE = os.getenv("OPENROUTER_API_KEY_PUPA_JUDGE") or DEFAULT_KEY
    KEY_UNTRUSTED = os.getenv("OPENROUTER_API_KEY_PUPA_UNTRUSTED") or DEFAULT_KEY

    api_base = os.getenv("OPENROUTER_API_BASE")

    # Configure Models
    untrusted_lm = dspy.LM(UNTRUSTED_MODEL, api_key=KEY_UNTRUSTED, api_base=api_base, **COMMON_KWARGS)
    solver_lm = dspy.LM(SOLVER_MODEL, api_key=KEY_SOLVER, api_base=api_base, **COMMON_KWARGS)
    generator_lm = dspy.LM(GENERATOR_MODEL, api_key=KEY_GENERATOR, api_base=api_base, **COMMON_KWARGS)
    judge_lm = dspy.LM(JUDGE_MODEL, api_key=KEY_JUDGE, api_base=api_base, **COMMON_KWARGS)
    llm_judge.set_lm(judge_lm)

    # Initialize Benchmark
    print(f"Loading PUPA dataset (origin) | Mode: {MODE.upper()} | Subset ({VAL_SUBSET_SIZE}) | NoBase...")
    pupa_bench = Papillon()
    pupa_bench.init_dataset()
    trainset = pupa_bench.train_set
    valset_full = pupa_bench.val_set
    testset = pupa_bench.test_set

    # Fixed Subset Selection
    valset = random.Random(42).sample(valset_full, min(VAL_SUBSET_SIZE, len(valset_full)))
    print(f"Using fixed validation subset of size {len(valset)} (sampled with seed 42)")

    # Initialize Program
    program = PAPILLON(untrusted_model=untrusted_lm)

    # Initialize WandB
    wandb.init(
        project="pupa-dynamic-fewshot-origin",
        name=f"pupa_{MODE}_nobase_subset{VAL_SUBSET_SIZE}_{NUM_ITERATIONS}iter",
        config={
            "mode": MODE,
            "num_iterations": NUM_ITERATIONS,
            "val_size": len(valset),
            "solver_model": SOLVER_MODEL,
            "num_threads": NUM_THREADS,
            "experiment_type": "ablation_no_base_prompt_subset",
        },
    )

    best_val_score = 0.0
    all_results = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        # 1. Sample few-shot examples
        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)

        # 2. Generate prompts
        dspy.configure(lm=generator_lm, adapter=ChatAdapter(), cache=False)
        metaprompt = create_metaprompt(fewshot_examples)
        redact_p, resp_p = generate_prompts_with_llm(metaprompt)
        print(f"Redaction Prompt: {redact_p[:50]}...")
        print(f"Response Prompt: {resp_p[:50]}...")

        # 3. Evaluate
        if not redact_p or not resp_p:
            val_score = 0.0
        else:
            # Update BOTH predictors
            program.craft_redacted_request.predictors()[0].signature.instructions = redact_p
            program.respond_to_query.signature.instructions = resp_p

            dspy.configure(lm=solver_lm, adapter=ChatAdapter(), cache=False)
            evaluate = dspy.Evaluate(
                devset=valset, metric=compute_overall_score, num_threads=NUM_THREADS, display_progress=True
            )
            val_score = float(evaluate(program))

        # 4. Token Usage Tracking
        cumulative_input, cumulative_output = get_token_usage([solver_lm, generator_lm, judge_lm, untrusted_lm])

        print(f"Validation Score: {val_score:.2%}")
        print(f"Tokens so far: Input={cumulative_input}, Output={cumulative_output}")
        best_val_score = max(best_val_score, val_score)

        wandb.log({
            "iteration": iteration + 1,
            "val_score": val_score,
            "best_val_score": best_val_score,
            "input_tokens_cumulative": cumulative_input,
            "output_tokens_cumulative": cumulative_output,
            "total_tokens_cumulative": cumulative_input + cumulative_output,
            "redaction_prompt": redact_p,
            "response_prompt": resp_p,
        })

        all_results.append({
            "iteration": iteration + 1,
            "redaction_prompt": redact_p,
            "response_prompt": resp_p,
            "val_score": val_score,
        })

    # Final Test on Top 1 (selection = argmax validation)
    all_results.sort(key=lambda x: x["val_score"], reverse=True)
    best_result = all_results[0]
    print(f"\nBest Configuration from Iteration {best_result['iteration']}")

    program.craft_redacted_request.predictors()[0].signature.instructions = best_result["redaction_prompt"]
    program.respond_to_query.signature.instructions = best_result["response_prompt"]

    dspy.configure(lm=solver_lm, adapter=ChatAdapter(), cache=False)
    evaluate_test = dspy.Evaluate(
        devset=testset, metric=compute_overall_score, num_threads=NUM_THREADS, display_progress=True
    )
    test_score = float(evaluate_test(program))

    print(f"Final Test Score: {test_score:.2%}")
    wandb.run.summary["test_score"] = test_score
    wandb.finish()


if __name__ == "__main__":
    main()
