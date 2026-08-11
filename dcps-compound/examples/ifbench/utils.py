"""IFBench reproduction utilities — strict paper replication.

Paper setup:
- 2-stage DSPy program: answer user query, then rewrite to follow constraints.
- Per-predictor textual feedback identifying satisfied/failed constraints.
- IF-RLVR Train for train/val (150/300), IFBench test for test (294).
"""

import ast
import random

import dspy
from datasets import load_dataset

from examples.ifbench.constraints import check_all_constraints

# ---------------------------------------------------------------------------
# 2-stage DSPy program (answer → constrained rewrite)
# ---------------------------------------------------------------------------


class ConstrainedRewrite(dspy.Module):
    """2-stage instruction-following program.

    Stage 1 (answer_query): Generate an initial answer to the user's query.
    Stage 2 (rewrite_answer): Rewrite the answer to satisfy output constraints.
    """

    def __init__(self):
        super().__init__()
        self.answer_query = dspy.ChainOfThought(
            "query -> draft_answer"
        )
        self.rewrite_answer = dspy.ChainOfThought(
            "query, draft_answer, constraints -> answer"
        )

    def forward(self, query: str, constraints: str = ""):
        draft = self.answer_query(query=query)
        rewritten = self.rewrite_answer(
            query=query,
            draft_answer=draft.draft_answer,
            constraints=constraints,
        )
        return dspy.Prediction(
            answer=rewritten.answer,
            draft_answer=draft.draft_answer,
        )


# ---------------------------------------------------------------------------
# GEPA feedback metric (per-predictor aware)
# ---------------------------------------------------------------------------


def ifbench_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric with per-predictor feedback.

    Checks all constraints programmatically and reports which ones
    passed and which failed.
    """
    response = getattr(pred, "answer", "")
    instruction_ids = gold.instruction_ids
    kwargs_list = gold.kwargs_list

    score, results = check_all_constraints(response, instruction_ids, kwargs_list)

    passed = [(iid, desc) for iid, ok, desc in results if ok]
    failed = [(iid, desc) for iid, ok, desc in results if not ok]

    # --- Per-predictor feedback ---
    if pred_name == "answer_query":
        if score >= 0.8:
            feedback = (
                f"Good initial answer. The draft helped achieve {score:.0%} constraint satisfaction. "
                f"Passed: {len(passed)}/{len(results)}."
            )
        else:
            feedback = (
                f"The initial answer needs improvement for constraint satisfaction ({score:.0%}). "
                f"Failed constraints: {'; '.join(desc for _, desc in failed[:3])}. "
                f"Generate a draft that anticipates the output constraints."
            )
        return dspy.Prediction(score=score, feedback=feedback)

    if pred_name == "rewrite_answer":
        if score == 1.0:
            feedback = (
                f"Perfect! All {len(results)} constraints satisfied."
            )
        elif score >= 0.5:
            feedback = (
                f"Partial success: {len(passed)}/{len(results)} constraints satisfied. "
                f"Failed: {'; '.join(desc for _, desc in failed[:3])}. "
                f"Rewrite more carefully to satisfy all output constraints."
            )
        else:
            feedback = (
                f"Poor constraint adherence: {len(passed)}/{len(results)} passed. "
                f"Failed: {'; '.join(desc for _, desc in failed[:5])}. "
                f"Read each constraint carefully and ensure the rewrite satisfies every one."
            )
        return dspy.Prediction(score=score, feedback=feedback)

    # --- Program-level feedback (default) ---
    if score == 1.0:
        feedback = f"All {len(results)} constraints satisfied."
    elif score >= 0.5:
        feedback = (
            f"Partially satisfied: {len(passed)}/{len(results)} constraints. "
            f"Failed: {'; '.join(desc for _, desc in failed[:3])}."
        )
    else:
        feedback = (
            f"Low constraint satisfaction: {len(passed)}/{len(results)}. "
            f"Failed: {'; '.join(desc for _, desc in failed[:5])}."
        )

    return dspy.Prediction(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# Dataset loading (paper: 150 train / 300 val from IF-RLVR, 294 test from IFBench)
# ---------------------------------------------------------------------------

TRAIN_SIZE = 150
VAL_SIZE = 300
TEST_SIZE = 294


def _parse_ifrlvr_example(item: dict) -> dspy.Example:
    """Parse an IF-RLVR train example into a dspy.Example."""
    user_msg = item["messages"][0]["content"]
    gt = ast.literal_eval(item["ground_truth"])
    instruction_ids = gt[0]["instruction_id"]
    kwargs_list = gt[0]["kwargs"]

    # Clean kwargs: replace None with {}
    kwargs_list = [kw if kw is not None else {} for kw in kwargs_list]

    # Extract constraint text (tab-separated)
    constraint_text = item["constraint"]

    return dspy.Example(
        query=user_msg,
        constraints=constraint_text,
        instruction_ids=instruction_ids,
        kwargs_list=kwargs_list,
    ).with_inputs("query", "constraints")


def _parse_ifbench_example(item: dict) -> dspy.Example:
    """Parse an IFBench test example into a dspy.Example."""
    prompt = item["prompt"]
    instruction_ids = item["instruction_id_list"]
    kwargs_list = item["kwargs"]

    # Clean kwargs
    kwargs_list = [
        {k: v for k, v in kw.items() if v is not None} if kw else {}
        for kw in kwargs_list
    ]

    # Extract constraint descriptions from the prompt
    # The constraints are appended to the prompt text
    return dspy.Example(
        query=prompt,
        constraints="(constraints are embedded in the query)",
        instruction_ids=instruction_ids,
        kwargs_list=kwargs_list,
    ).with_inputs("query", "constraints")


def load_ifbench_dataset(
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
):
    """Load IF-RLVR train + IFBench test and split into train/val/test."""

    # --- Train/Val from IF-RLVR ---
    print("Loading IF-RLVR training data...")
    ifrlvr = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    indices = list(range(len(ifrlvr)))
    random.Random(42).shuffle(indices)

    total_trainval = train_size + val_size
    selected = indices[:total_trainval]

    train_examples = []
    val_examples = []

    for i, idx in enumerate(selected):
        ex = _parse_ifrlvr_example(ifrlvr[idx])
        if i < train_size:
            train_examples.append(ex)
        else:
            val_examples.append(ex)

    # --- Test from IFBench ---
    print("Loading IFBench test data...")
    ifbench = load_dataset("allenai/IFBench_test", split="train")

    test_examples = []
    for i in range(min(test_size, len(ifbench))):
        test_examples.append(_parse_ifbench_example(ifbench[i]))

    return train_examples, val_examples, test_examples
