import argparse
import os

import dspy
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

from examples.livebench_math.utils import MathSolver, livebench_math_metric, load_livebench_math_dataset, evaluate_on_dataset


BASELINE_PROMPT = (
    """You are a knowledgeable math problem-solving assistant specialized in LiveBench-Math tasks. For each given problem, follow these instructions carefully:

1. **Step-by-step Reasoning:** Before giving the final answer, think through the problem step-by-step aloud. Provide clear, logical explanations and intermediate steps that demonstrate how you arrive at the solution.

2. **Final Answer Formatting:**
   - For **AMPS_Hard** tasks: The final answer must be a LaTeX expression placed strictly inside \boxed{...}. For example, \boxed{\frac{n+1}{2}^2}.
   - For **math_comp** tasks: The final answer must be an integer between 0 and 999, enclosed inside \boxed{...}. For example, \boxed{123}.
   - For **olympiad** tasks: The final answer must be a comma-separated list of integers corresponding to expression indices, strictly enclosed in \boxed{...}. For example, \boxed{3,5,1,2,4,7,6}.

3. **Following Patterns from Examples:**  
   - If the problem involves matching formulae (as in olympiad tasks), explicitly justify your choices step-by-step.  
   - For multiple-choice styled questions (math_comp), provide reasoning and select the best answer, then output only the integer final answer as specified.  
   - For complicated proofs or derivations (AMPS_Hard), write detailed reasoning and conclude with the boxed LaTeX formula.

4. **If Unsure, Make the Best Possible Guess:** Clearly state your uncertainty and justify your choice as best as you can.

Always conclude by outputting the final answer exactly in the correct \boxed{...} format and nothing else outside it.

Begin by carefully reading the problem, then produce your reasoning and final boxed answer accordingly."""
)


def configure_lm(
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
):
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment or .env file.")

    solver_lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_retries=0,
        cache=False,
        extra_body={
            "top_k": top_k,
            # "provider": {
            #     "only": ["alibaba"],
            # },
        },
    )
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )
    return solver_lm


def select_dataset(dataset_name: str, num_samples: int | None):
    trainset, valset, testset = load_livebench_math_dataset()
    datasets = {
        "train": trainset,
        "val": valset,
        "test": testset,
    }
    dataset = datasets[dataset_name]
    if num_samples is not None:
        dataset = dataset[:num_samples]
    return dataset, trainset, valset, testset


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned LiveBench-Math baseline test")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--dataset", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--record-history", action="store_true", default=False, help="Record detailed evaluation history to JSON files")
    parser.add_argument("--output-dir", type=str, default="evaluation_history", help="Directory to save history JSON files")
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    solver_lm = configure_lm(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    dataset, trainset, valset, testset = select_dataset(args.dataset, args.num_samples)

    program = MathSolver()
    program.solve.predict.signature.instructions = BASELINE_PROMPT

    print("=" * 80)
    print("LiveBench-Math baseline test (paper-aligned)")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Retries: {solver_lm.kwargs.get('num_retries')}")
    print("DSPy cache: disk=False, memory=False; LM cache=False")
    print("Adapter: ChatAdapter(use_json_adapter_fallback=False)")
    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}")
    print(f"Evaluating split: {args.dataset}, examples={len(dataset)}")
    print("Metric: official calculate_livebench_score via livebench_math_metric")
    print(f"History recording: {args.record_history}")
    if args.record_history:
        print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Prepare model configuration for history recording
    model_config = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "num_retries": solver_lm.kwargs.get('num_retries', 0),
    }

    score, history = evaluate_on_dataset(
        program, 
        dataset, 
        args.num_threads,
        record_history=args.record_history,
        output_dir=args.output_dir if args.record_history else None,
        model_config=model_config if args.record_history else None
    )
    print(f"Baseline score: {score:.2%}")
    
    if history and args.record_history:
        lm_stats = history.get("lm_statistics", {})
        print(f"LM Cost: ${lm_stats.get('total_cost', 0):.4f}")
        print(f"Input Tokens: {lm_stats.get('total_input_tokens', 0)}")
        print(f"Output Tokens: {lm_stats.get('total_output_tokens', 0)}")
        print(f"Num Correct: {history.get('num_correct', 0)}/{len(history.get('examples', []))}")


if __name__ == "__main__":
    main()
