import random
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Any
import dspy

from datasets import load_dataset


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()


predictor = dspy.ChainOfThought(GenerateResponse)


def run_llm(example, prompt: str):
    """Run the LLM on a single example with the given prompt."""
    predictor.predict.signature.instructions = prompt
    return predictor(problem=example.problem)


def math_metric(example, prediction):
    """Compute score and detailed feedback for math problems."""
    correct_answer, written_solution = int(example.answer), getattr(example, "solution", "")
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting. The correct answer is '{correct_answer}'.{solution_suffix}{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        return 0.0, feedback_text

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


def load_math_dataset():
    train_split = []
    test_split = []

    train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for item in train_load_dataset:
        question = item["problem"]
        solution = item["solution"]
        answer = item["answer"]

        train_split.append(dspy.Example(problem=question, solution=solution, answer=answer).with_inputs("problem"))

    random.Random(0).shuffle(train_split)

    test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_load_dataset:
        question = item["problem"]
        answer = item["answer"]

        test_split.append(dspy.Example(problem=question, answer=answer).with_inputs("problem"))

    train_size = len(train_split)
    trainset = train_split[: train_size // 2]
    valset = train_split[train_size // 2 :]
    # Replicate test set 5x for avg@5 (pass@1 low-variance estimation)
    # This matches the experiment reproduction config
    testset = test_split * 5

    return trainset, valset, testset


def evaluate_on_dataset(prompt, dataset, record_history: bool = False, output_dir: str = None, model_config: dict = None):
    """Evaluate a predictor on a dataset using dspy.Evaluate.

    Errors (unparseable outputs, LM failures, etc.) are swallowed per-example
    and scored as 0. ``max_errors`` is set to 999999 so the evaluator never
    aborts, and ``failure_score=0.0`` is set explicitly.

    Args:
        prompt: The prompt to test
        dataset: The dataset to evaluate on
        record_history: Whether to record detailed history for each example
        output_dir: Directory to save history JSON file (required if record_history=True)
        model_config: Dictionary containing model configuration (model, temperature, etc.)

    Returns:
        score: The evaluation score (0-1)
        history: Optional dict with detailed history if record_history=True
    """
    predictor.predict.signature.instructions = prompt

    # Storage for detailed history
    history_data = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "model_config": model_config or {},
        "examples": []
    } if record_history else None

    # Thread-safe counters for real-time wandb logging
    _lock = threading.Lock()
    _counters = {"done": 0, "correct": 0}

    def dspy_metric(example, prediction):
        """Adapter: dspy.Evaluate expects a numeric score, not (score, feedback)."""
        try:
            score, feedback = math_metric(example, prediction)
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "problem": example.problem,
                    "correct_answer": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": score,
                    "feedback": feedback,
                })
        except Exception as e:
            score = 0.0
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "problem": example.problem,
                    "correct_answer": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": 0.0,
                    "error": str(e),
                })

        # Real-time per-problem logging to wandb
        with _lock:
            _counters["done"] += 1
            _counters["correct"] += int(score)
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "running_correct": _counters["correct"],
                        "running_total": _counters["done"],
                        "running_accuracy": _counters["correct"] / _counters["done"],
                    })
            except Exception:
                pass  # wandb not available or not initialized

        return score

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=dspy_metric,
        num_threads=8,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )

    eval_result = evaluator(predictor)
    score = eval_result.score / 100.0

    # Record LM statistics
    if record_history and history_data is not None:
        lm = dspy.settings.lm
        if lm and hasattr(lm, 'history'):
            total_cost = 0
            total_input_tokens = 0
            total_output_tokens = 0
            for trace in lm.history:
                total_cost += trace.get("cost", 0) or 0
                total_input_tokens += trace.get("usage", {}).get("prompt_tokens", 0)
                total_output_tokens += trace.get("usage", {}).get("completion_tokens", 0)
            
            history_data["lm_statistics"] = {
                "total_cost": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "num_calls": len(lm.history),
            }
        
        history_data["overall_score"] = score
        history_data["num_correct"] = sum(1 for ex in history_data["examples"] if ex.get("score") == 1.0)

        # Save to JSON file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_history_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"  History saved to: {filepath}")

    if record_history:
        return score, history_data
    return score
