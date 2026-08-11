"""LiveBench-Math reproduction utilities — strict paper replication.

Paper setup (aligned with gepa-artifact/gepa_artifact/benchmarks/livebench_math):
- LiveBench math subset (n=368), shuffled with seed 0.
- Split indices: ``int(tot*0.33)`` / ``int(tot*0.66)`` → 121 / 121 / 126.
- Trim caps (150 / 300 / 300) from BenchmarkMeta never trigger for LiveBench.
- Single-step ChainOfThought("question -> answer") as the program.
- Metric: official LiveBench ``calculate_livebench_score`` (sub-task aware,
  strips ``<think>…</think>`` before scoring).
"""

import random
import json
import os
from datetime import datetime
from typing import Dict, List, Any

import dspy
from datasets import load_dataset

from examples.livebench_math.livebenchmath_utils.metric import (
    calculate_livebench_score,
)

# ---------------------------------------------------------------------------
# Scoring (delegates to the official LiveBench pipeline)
# ---------------------------------------------------------------------------


def check_math_answer(prediction: str, question_d: dict) -> tuple[float, str]:
    """Score a prediction against the raw LiveBench question dict.

    Thin wrapper over ``calculate_livebench_score`` so existing call sites have
    a stable, simple API. ``question_d`` must be the original HuggingFace row
    (containing ``turns``, ``ground_truth``, ``task``, optionally ``subtask``).
    """
    score, feedback = calculate_livebench_score(question_d, prediction, debug=True)
    return float(score), feedback


# ---------------------------------------------------------------------------
# Single-step ChainOfThought program
# ---------------------------------------------------------------------------


class GenerateResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""

    question = dspy.InputField()
    answer = dspy.OutputField()


class MathSolver(dspy.Module):
    """Single-step ChainOfThought math solver.

    Uses the same ``GenerateResponse`` signature (and seed instruction) as
    ``gepa-artifact``'s ``livebenchmath_program.program_cot``.
    """

    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(GenerateResponse)

    def forward(self, question: str):
        result = self.solve(question=question)
        return dspy.Prediction(answer=result.answer)


# ---------------------------------------------------------------------------
# GEPA feedback metric
# ---------------------------------------------------------------------------


def livebench_math_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric with textual feedback.

    Delegates to the official ``calculate_livebench_score`` (with ``debug=True``
    to get feedback text), matching ``gepa-artifact`` exactly.
    """
    prediction = getattr(pred, "answer", "")
    question_d = gold["question_d"]
    score, feedback_text = calculate_livebench_score(question_d, prediction, debug=True)
    return dspy.Prediction(score=float(score), feedback=feedback_text)


# ---------------------------------------------------------------------------
# Dataset loading (paper: 368 total, shuffled seed 0, split equally)
# ---------------------------------------------------------------------------


def load_livebench_math_dataset():
    """Load LiveBench math subset and split exactly like ``gepa-artifact``.

    368 examples shuffled with seed 0, then sliced at ``int(tot*0.33)`` and
    ``int(tot*0.66)`` → train=121, val=121, test=126.

    Each ``dspy.Example`` carries ``question_d`` (the raw HuggingFace row) so
    the official ``calculate_livebench_score`` metric can grade it.
    """
    print("Loading LiveBench math dataset...")
    ds = load_dataset("livebench/math", split="test")

    all_examples = []
    for item in ds:
        # Convert any non-primitive fields (e.g. datetime) so dspy.Example can
        # hash/serialize the row. We keep a dict with the exact keys the metric
        # reads (``turns``, ``ground_truth``, ``task``, ``subtask``).
        question_d = dict(item)
        all_examples.append(
            dspy.Example(
                question=item["turns"][0],
                answer=item["ground_truth"],
                task=item["task"],
                question_d=question_d,
            ).with_inputs("question")
        )

    assert len(all_examples) == 368, f"Expected 368 examples, got {len(all_examples)}"

    # Shuffle with seed 0 per paper artifact
    random.Random(0).shuffle(all_examples)

    # Split exactly as gepa-artifact: int(tot*0.33) / int(tot*0.66)
    tot = len(all_examples)
    trainset = all_examples[: int(tot * 0.33)]
    valset = all_examples[int(tot * 0.33) : int(tot * 0.66)]
    testset = all_examples[int(tot * 0.66) :]

    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Evaluation with history recording
# ---------------------------------------------------------------------------


def evaluate_on_dataset(program, dataset, num_threads: int, record_history: bool = False, output_dir: str = None, model_config: dict = None):
    """Evaluate a program on a dataset with optional history recording.

    Args:
        program: The DSPy program to evaluate
        dataset: The dataset to evaluate on
        num_threads: Number of threads for evaluation
        record_history: Whether to record detailed history for each example
        output_dir: Directory to save history JSON file (required if record_history=True)
        model_config: Dictionary containing model configuration (model, temperature, etc.)

    Returns:
        score: The evaluation score (0-1)
        history: Optional dict with detailed history if record_history=True
    """
    # Extract prompt/instruction from program
    prompt_instruction = ""
    if hasattr(program, 'solve') and hasattr(program.solve, 'predict') and hasattr(program.solve.predict, 'signature') and hasattr(program.solve.predict.signature, 'instructions'):
        prompt_instruction = program.solve.predict.signature.instructions
    elif hasattr(program, 'predict') and hasattr(program.predict, 'signature') and hasattr(program.predict.signature, 'instructions'):
        prompt_instruction = program.predict.signature.instructions

    # Extract dataset information
    dataset_info = {
        "size": len(dataset),
        "split": "unknown",
        "tasks": [],
    }
    if dataset:
        tasks = set()
        for ex in dataset:
            if hasattr(ex, 'task'):
                tasks.add(ex.task)
        dataset_info["tasks"] = sorted(list(tasks))

    # Storage for detailed history
    history_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": dataset_info,
        "num_threads": num_threads,
        "model_config": model_config or {},
        "prompt_instruction": prompt_instruction,
        "examples": []
    } if record_history else None

    def dspy_metric_with_history(example, prediction, trace=None):
        """Metric wrapper that records history if enabled."""
        try:
            result = livebench_math_metric(example, prediction, trace)
            score = float(result.score)
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "question": example.question,
                    "task": example.task,
                    "subtask": example.question_d.get("subtask", "N/A"),
                    "ground_truth": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": score,
                    "feedback": result.feedback,
                })
            return score
        except Exception as e:
            if record_history and history_data is not None:
                history_data["examples"].append({
                    "question": example.question,
                    "task": example.task,
                    "subtask": example.question_d.get("subtask", "N/A"),
                    "ground_truth": str(example.answer),
                    "predicted_answer": str(getattr(prediction, "answer", "")),
                    "score": 0.0,
                    "error": str(e),
                })
            return 0.0

    # Use the appropriate metric based on whether history recording is enabled
    metric = dspy_metric_with_history if record_history else livebench_math_metric

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )

    eval_result = evaluator(program)
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

    return score, history_data if record_history else (score, None)
