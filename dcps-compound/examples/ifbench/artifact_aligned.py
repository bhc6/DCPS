import json
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import dspy

TRAIN_SIZE = 150
VAL_SIZE = 300
TEST_SIZE = 300


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root() -> Path:
    return Path.cwd() / "gepa-artifact" if (Path.cwd() / "gepa-artifact").exists() else _repo_root() / "gepa-artifact"


def _ifbench_artifact_dir() -> Path:
    return _artifact_root() / "gepa_artifact" / "benchmarks" / "IFBench"


def _load_jsonl_examples(path: Path) -> list[dspy.Example]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            examples.append(dspy.Example(**data).with_inputs("prompt"))
    return examples


def _trim_dataset(dataset, size: int):
    if size is None or size >= len(dataset):
        return dataset
    rng = random.Random()
    rng.seed(1)
    return rng.sample(dataset, size)


def load_ifbench_dataset(
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
):
    artifact_dir = _ifbench_artifact_dir()
    train_path = artifact_dir / "data" / "IFBench_train.jsonl"
    test_path = artifact_dir / "data" / "IFBench_test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing IFBench artifact JSONL files. Expected "
            f"{train_path} and {test_path}. Clone gepa-artifact with LFS/data files."
        )

    train_val_set = _load_jsonl_examples(train_path)
    test_set = _load_jsonl_examples(test_path)

    train_pool = train_val_set[300:600]
    val_set = train_val_set[:300]

    train_set = _trim_dataset(train_pool, train_size)
    val_set = _trim_dataset(val_set, val_size)
    test_set = _trim_dataset(test_set, test_size)

    return train_set, val_set, test_set


class GenerateResponse(dspy.Signature):
    """Respond to the query"""

    query = dspy.InputField()
    response = dspy.OutputField()


class EnsureCorrectResponse(dspy.Signature):
    """Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response."""

    query = dspy.InputField()
    response = dspy.InputField()
    final_response = dspy.OutputField()


class IFBenchCoT2StageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_response_module = dspy.ChainOfThought(GenerateResponse)
        self.ensure_correct_response_module = dspy.ChainOfThought(EnsureCorrectResponse)

    def forward(self, prompt: str):
        response = self.generate_response_module(query=prompt).response
        final_response = self.ensure_correct_response_module(query=prompt, response=response)
        return dspy.Prediction(response=final_response.final_response)


def build_ifbench_program(instructions: str | None = None) -> IFBenchCoT2StageProgram:
    """Build an artifact-aligned IFBench program.

    If ``instructions`` is provided, it is installed as the instructions for
    BOTH predictor stages. For stage-specific control, prefer
    ``build_ifbench_program_two_stage``.
    """
    return build_ifbench_program_two_stage(instructions, instructions)


def build_ifbench_program_two_stage(
    generate_instructions: str | None,
    ensure_instructions: str | None,
) -> IFBenchCoT2StageProgram:
    """Build an artifact-aligned IFBench program with per-stage instructions.

    ``generate_instructions`` is installed on
    ``generate_response_module.predict.signature`` and
    ``ensure_instructions`` on ``ensure_correct_response_module.predict.signature``.
    Passing ``None`` for either leaves the corresponding stage's artifact
    default instruction untouched. This matches the GEPA optimization surface
    of optimizing the two predictor instructions independently.
    """
    program = IFBenchCoT2StageProgram()
    if generate_instructions is not None:
        sig = cast(Any, program.generate_response_module.predict.signature)
        program.generate_response_module.predict.signature = sig.with_instructions(generate_instructions)
    if ensure_instructions is not None:
        sig = cast(Any, program.ensure_correct_response_module.predict.signature)
        program.ensure_correct_response_module.predict.signature = sig.with_instructions(ensure_instructions)
    return program


def artifact_default_generate_instructions() -> str:
    return GenerateResponse.instructions


def artifact_default_ensure_instructions() -> str:
    return EnsureCorrectResponse.instructions


_STAGE1_TAG = "<<<STAGE1>>>"
_STAGE2_TAG = "<<<STAGE2>>>"


def two_stage_output_format() -> str:
    """Return the exact output format the generator LM must follow."""
    return (
        f"{_STAGE1_TAG}\n"
        "...stage 1 (drafter) system prompt...\n"
        f"{_STAGE2_TAG}\n"
        "...stage 2 (finalizer) system prompt..."
    )


def parse_two_stage_prompt(text: str) -> tuple[str, str] | None:
    """Parse a generator LM output into ``(stage1, stage2)`` instructions.

    Returns ``None`` if either section is missing or empty. The parser looks
    for the ``<<<STAGE1>>>`` and ``<<<STAGE2>>>`` section markers that the
    metaprompt asks the generator to emit.
    """
    if not text:
        return None
    s1_idx = text.find(_STAGE1_TAG)
    s2_idx = text.find(_STAGE2_TAG)
    if s1_idx == -1 or s2_idx == -1 or s2_idx <= s1_idx:
        return None
    stage1 = text[s1_idx + len(_STAGE1_TAG):s2_idx].strip()
    stage2 = text[s2_idx + len(_STAGE2_TAG):].strip()
    if not stage1 or not stage2:
        return None
    return stage1, stage2


def _artifact_metric_with_feedback() -> Callable:
    artifact_root = _artifact_root()
    if str(artifact_root) not in sys.path:
        sys.path.insert(0, str(artifact_root))
    module = import_module("gepa_artifact.benchmarks.IFBench.ifbench_metric")
    return module.metric_with_feedback


def ifbench_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return _artifact_metric_with_feedback()(gold, pred, trace)
