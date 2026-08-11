"""Dynamic few-shot prompt search baseline for HoVer — GPT-4.1-mini variant.

Same pipeline as ``examples/hover/dynamic_fewshot.py`` (qwen3-8b) but
with solver/generator switched to ``openrouter/openai/gpt-4.1-mini``,
using paper-aligned sampling (``temperature=1.0``, no top_p/top_k/
provider pin).

Dataset split, fixed validation-pool strategy, four-stage program, and
metric are identical to the qwen3-8b variant — every change is confined
to the LM configuration so results are directly comparable across model
backbones.
"""

import argparse
import os

import dspy
from dotenv import load_dotenv

from examples.hover.dynamic_fewshot import run


def main():
    parser = argparse.ArgumentParser(description="Dynamic few-shot prompt search for HoVer (gpt-4.1-mini)")
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--num-fewshot", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--val-sample-size", type=int, default=30,
        help="Fixed validation pool size (head-slice of paper valset). Reused across all iterations.",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore and delete any existing checkpoint; start fresh from iteration 1.",
    )
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    api_key_env = "OPENROUTER_API_KEY_HOVER"
    api_key = os.getenv(api_key_env) or os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} or OPENROUTER_API_KEY in your environment or .env file.")

    solver_model = "openrouter/openai/gpt-4.1-mini"
    generator_model = "openrouter/openai/gpt-4.1-mini"

    solver_lm = dspy.LM(
        solver_model,
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=16384,
        num_retries=0,
        cache=False,
    )
    generator_lm = dspy.LM(
        generator_model,
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=16384,
        num_retries=0,
        cache=False,
    )

    run(
        solver_lm=solver_lm,
        generator_lm=generator_lm,
        num_iterations=args.num_iterations,
        num_fewshot=args.num_fewshot,
        top_k=args.top_k,
        val_sample_size=args.val_sample_size,
        num_threads=args.num_threads,
        wandb_run_name=f"dynamic_fewshot_gpt41mini_{args.num_iterations}iter_{args.num_fewshot}shot",
        api_key_env_for_logging=api_key_env,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
