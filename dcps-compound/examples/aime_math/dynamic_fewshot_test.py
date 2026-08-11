import os
import random
from typing import List

import dspy
from dotenv import load_dotenv

from examples.aime_math.utils import evaluate_on_dataset, load_math_dataset, math_metric, run_llm
from gepa.optimize_anything import SideInfo


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate a candidate on a single example."""
    prediction = run_llm(example, candidate)
    score, feedback = math_metric(example, prediction)

    side_info = {
        "score": score,
        "input": example.problem,
        "prompt": candidate,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }

    return score, side_info


def sample_fewshot_examples(trainset: List, num_examples: int = 3) -> str:
    """Randomly sample few-shot examples from training set."""
    sampled_examples = random.sample(trainset, min(num_examples,
                                                   len(trainset)))

    fewshot_text = ""
    for i, example in enumerate(sampled_examples, 1):
        fewshot_text += f"Example {i}:\n"
        fewshot_text += f"Problem: {example.problem}\n"
        if hasattr(example, 'solution') and example.solution:
            fewshot_text += f"Solution: {example.solution}\n"
        fewshot_text += f"Answer: {example.answer}\n\n"

    return fewshot_text.strip()


def create_metaprompt(fewshot_examples: str) -> str:
    """Create a metaprompt for generating math problem solving prompts."""
    return f"""You are an expert prompt engineer for AI systems that solve competition math problems. 

Based on the few-shot examples below, design an effective prompt that will guide an AI to solve similar math problems accurately. The prompt should:

1. Be clear and specific about the expected format
2. Encourage step-by-step reasoning
3. Reference the patterns shown in the examples
4. Guide the AI to provide final numerical answers

Here are the few-shot examples to analyze:

{fewshot_examples}

Now, generate a prompt that incorporates insights from these examples:"""


def generate_prompt_with_llm(metaprompt: str) -> str:
    """Use LLM to generate a prompt based on the metaprompt."""

    # Create a temporary signature for prompt generation
    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(
            desc="The metaprompt for generating a math solving prompt.")
        generated_prompt = dspy.OutputField(
            desc="The generated prompt for solving math problems.")

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    result = generator(metaprompt=metaprompt)
    return result.generated_prompt


def main():
    load_dotenv()

    # Configuration
    NUM_ITERATIONS = 20  # Number of prompt generation iterations
    NUM_FEWSHOT_EXAMPLES = 3  # Number of few-shot examples to sample
    TOP_K = 5  # Number of top prompts to test on final test set
    VAL_SAMPLE_SIZE = 10  # Number of validation examples to randomly sample per iteration

    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in your environment or .env file.")

    # Configure solver LM for math problems
    solver_lm = dspy.LM(
        "openrouter/openai/gpt-4.1-mini",
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=32000,
    )

    # Configure generator LM for prompt generation
    generator_lm = dspy.LM(
        "openrouter/openai/gpt-4.1-mini",  # Use a cheaper model for prompt generation
        api_key=api_key,
        api_base=api_base,
        temperature=0.7,
        max_tokens=1000,
    )

    dspy.configure(lm=solver_lm)

    trainset, valset, testset = load_math_dataset()

    print(f"Starting dynamic few-shot prompt generation...")
    print(
        f"Iterations: {NUM_ITERATIONS}, Few-shot examples: {NUM_FEWSHOT_EXAMPLES}, Top K: {TOP_K}"
    )
    print(
        f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}"
    )
    print(f"Validation sample size per iteration: {VAL_SAMPLE_SIZE}")

    # Generate and test prompts
    all_results = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        # Sample few-shot examples from training set
        fewshot_examples = sample_fewshot_examples(trainset,
                                                   NUM_FEWSHOT_EXAMPLES)
        print(f"Sampled {NUM_FEWSHOT_EXAMPLES} examples from training set")

        # Create metaprompt
        metaprompt = create_metaprompt(fewshot_examples)

        # Switch to generator LM and generate prompt
        dspy.configure(lm=generator_lm)
        generated_prompt = generate_prompt_with_llm(metaprompt)

        # Switch back to solver LM for evaluation
        dspy.configure(lm=solver_lm)

        print(f"Generated prompt: {generated_prompt}")

        # Randomly sample a subset from validation set for faster evaluation
        val_sample = random.sample(valset, min(VAL_SAMPLE_SIZE, len(valset)))
        val_score = evaluate_on_dataset(generated_prompt, val_sample)
        print(f"Validation score (on {len(val_sample)} samples): {val_score:.2%}")

        # Store results
        all_results.append({
            'iteration': iteration + 1,
            'prompt': generated_prompt,
            'val_score': val_score,
            'fewshot_examples': fewshot_examples
        })

    # Sort by validation score and select top K
    all_results.sort(key=lambda x: x['val_score'], reverse=True)
    top_results = all_results[:TOP_K]

    print(f"\n=== Top {TOP_K} Prompts from Validation ===")
    for i, result in enumerate(top_results):
        print(f"\nRank {i+1} (Iteration {result['iteration']}):")
        print(f"Validation Score: {result['val_score']:.2%}")
        print(f"Prompt: {result['prompt']}")

    # Final testing on test set
    print(f"\n=== Final Testing on Test Set ===")
    final_results = []

    for i, result in enumerate(top_results):
        print(f"\nTesting Rank {i+1} on test set...")
        test_score = evaluate_on_dataset(result['prompt'], testset)
        print(f"Test Score: {test_score:.2%}")

        final_results.append({
            'rank': i + 1,
            'iteration': result['iteration'],
            'prompt': result['prompt'],
            'val_score': result['val_score'],
            'test_score': test_score
        })

    # Final report
    print(f"\n=== FINAL REPORT ===")
    print(
        f"Generated {NUM_ITERATIONS} prompts using dynamic few-shot sampling")
    print(f"Selected top {TOP_K} based on validation performance")
    print(f"Tested on {len(testset[:50])} test examples\n")

    print("Rank | Iteration | Val Score | Test Score | Prompt")
    print("-" * 80)
    for result in final_results:
        print(
            f"{result['rank']:4} | {result['iteration']:9} | {result['val_score']:9.2%} | {result['test_score']:10.2%} | {result['prompt'][:50]}..."
        )

    # Best overall performance
    best_result = max(final_results, key=lambda x: x['test_score'])
    print(f"\n🏆 Best Overall Performance:")
    print(f"Iteration: {best_result['iteration']}")
    print(f"Validation Score: {best_result['val_score']:.2%}")
    print(f"Test Score: {best_result['test_score']:.2%}")
    print(f"Prompt: {best_result['prompt']}")

    # Performance analysis
    avg_val_score = sum(r['val_score']
                        for r in final_results) / len(final_results)
    avg_test_score = sum(r['test_score']
                         for r in final_results) / len(final_results)

    print(f"\n📊 Performance Summary:")
    print(f"Average Validation Score (Top {TOP_K}): {avg_val_score:.2%}")
    print(f"Average Test Score (Top {TOP_K}): {avg_test_score:.2%}")
    print(f"Validation-to-Test Gap: {(avg_val_score - avg_test_score):.2%}")


if __name__ == "__main__":
    main()
