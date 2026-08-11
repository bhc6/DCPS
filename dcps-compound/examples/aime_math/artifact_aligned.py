import random
from typing import Any, cast

import dspy
from datasets import load_dataset


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    answer = dspy.OutputField()


program_cot = dspy.ChainOfThought(GenerateResponse)


def artifact_default_instruction() -> str:
    return GenerateResponse.instructions


def load_aime_dataset():
    train_raw = load_dataset("AI-MO/aimo-validation-aime")["train"]
    train_split = [
        dspy.Example({
            "problem": cast(Any, item)["problem"],
            "solution": cast(Any, item)["solution"],
            "answer": cast(Any, item)["answer"],
        }).with_inputs("problem")
        for item in train_raw
    ]
    random.Random(0).shuffle(train_split)

    test_raw = load_dataset("MathArena/aime_2025")["train"]
    test_split = [
        dspy.Example({
            "problem": cast(Any, item)["problem"],
            "answer": cast(Any, item)["answer"],
        }).with_inputs("problem")
        for item in test_raw
    ]

    split_idx = int(0.5 * len(train_split))
    trainset = train_split[:split_idx]
    valset = train_split[split_idx:]
    testset = test_split * 5
    return trainset, valset, testset


def aime_metric(example, prediction, trace=None):
    correct_answer = int(example["answer"])
    try:
        llm_answer = int(prediction.answer)
    except ValueError:
        return 0
    return int(correct_answer == llm_answer)


def aime_metric_with_feedback(example, prediction, trace=None):
    correct_answer = int(example["answer"])
    written_solution = example.get("solution", "")
    try:
        llm_answer = int(prediction.answer)
    except ValueError:
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. You responded with "
            f"'{prediction.answer}', which couldn't be parsed as a python integer. Please ensure "
            f"your answer is a valid integer without any additional text or formatting. The correct "
            f"answer is '{correct_answer}'."
        )
        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what "
                "takeaways you can learn from this solution to improve your future answers and approach "
                "to similar problems and ensure your final answer is a valid integer."
            )
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways "
            "you can learn from this solution to improve your future answers and approach to similar problems."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


def build_aime_program(instructions: str | None = None):
    program = dspy.ChainOfThought(GenerateResponse)
    if instructions is not None:
        signature = cast(Any, program.predict.signature)
        program.predict.signature = signature.with_instructions(instructions)
    return program


def evaluate_on_dataset(instructions: str | None, dataset, num_threads: int = 8) -> float:
    program = build_aime_program(instructions)
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=aime_metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=999999,
        failure_score=0.0,
    )
    result = evaluator(program)
    return result.score / 100.0
