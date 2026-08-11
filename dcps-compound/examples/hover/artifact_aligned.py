"""Artifact-aligned HoVer (3-hop multi-hop retrieval) adapter.

Mirrors the IFBench ``artifact_aligned.py`` design. All dataset, program,
and metric code is delegated to / re-exported from
``gepa-artifact/gepa_artifact/benchmarks/hover/`` so this experiment is
provably aligned to the GEPA paper artifact at commit
``cbefbc1aa0f43dd39874ec4bf42211365dbda42e``.

Key differences from IFBench:
- HoVer optimizes **four** predictor instructions independently, not two:
  ``summarize1``, ``create_query_hop2``, ``summarize2``, ``create_query_hop3``.
  The four-stage tagged output format and parser below reflect that.
- Dataset is loaded via the artifact ``hoverBench`` Benchmark class
  (HuggingFace ``hover`` 3-hop train, shuffled seed 0, split 40/40/20
  test/val/train, trimmed to 300/300/150 with seed 1).
- The retrieval program (``HoverMultiHop``) requires a local BM25S index
  over the 2017 Wikipedia abstracts dump (~5GB). On first call, the
  artifact downloads and indexes the corpus into
  ``gepa-artifact/gepa_artifact/benchmarks/hover/`` (see
  ``hover_program.initialize_bm25s_retriever_and_corpus``). All subsequent
  calls hit the on-disk index.
"""

import json
import random
import sys
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import dspy

# HoVer official release JSON. The HuggingFace ``hover`` dataset script no
# longer loads on modern ``datasets`` versions (script-based datasets are
# rejected, and the parquet auto-export decoder fails), so we fetch the
# canonical train file directly from the upstream GitHub repo.
_HOVER_TRAIN_URL = (
    "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
)
_HOVER_TRAIN_CACHE_NAME = "hover_train_release_v1.1.json"

TRAIN_SIZE = 150
VAL_SIZE = 300
TEST_SIZE = 300

# Stage names match the predictor attribute names on ``HoverMultiHop``.
STAGE_NAMES: tuple[str, str, str, str] = (
    "summarize1",
    "create_query_hop2",
    "summarize2",
    "create_query_hop3",
)

# Per-stage tags used in the four-stage tagged metaprompt output format.
_STAGE_TAGS: dict[str, str] = {
    "summarize1": "<<<SUMMARIZE1>>>",
    "create_query_hop2": "<<<QUERY_HOP2>>>",
    "summarize2": "<<<SUMMARIZE2>>>",
    "create_query_hop3": "<<<QUERY_HOP3>>>",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root() -> Path:
    return Path.cwd() / "gepa-artifact" if (Path.cwd() / "gepa-artifact").exists() else _repo_root() / "gepa-artifact"


def _ensure_artifact_on_path() -> None:
    artifact_root = _artifact_root()
    if not artifact_root.exists():
        raise FileNotFoundError(
            f"gepa-artifact directory not found at {artifact_root}. "
            "Clone https://github.com/gepa-ai/gepa-artifact next to this repo."
        )
    if str(artifact_root) not in sys.path:
        sys.path.insert(0, str(artifact_root))


def _hover_artifact_module():
    _ensure_artifact_on_path()
    return import_module("gepa_artifact.benchmarks.hover")


def _hover_program_module():
    _ensure_artifact_on_path()
    return import_module("gepa_artifact.benchmarks.hover.hover_program")


def _hover_utils_module():
    _ensure_artifact_on_path()
    return import_module("gepa_artifact.benchmarks.hover.hover_utils")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _hover_cache_dir() -> Path:
    """Local cache dir for the upstream HoVer JSON release. Co-located with
    the artifact's BM25 index so all HoVer-side data lives together.
    """
    return _artifact_root() / "gepa_artifact" / "benchmarks" / "hover"


def _download_hover_train_json() -> Path:
    """Download the HoVer official train JSON to the cache dir if absent.

    Returns the local cache path. Raises if both cache miss and download
    fail.
    """
    cache_dir = _hover_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _HOVER_TRAIN_CACHE_NAME
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    print(f"[hover] downloading {_HOVER_TRAIN_URL} -> {cache_path}")
    with urllib.request.urlopen(_HOVER_TRAIN_URL) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return cache_path


def _normalise_supporting_facts(raw: list) -> list[dict]:
    """Convert HoVer's ``[[title, sent_idx], ...]`` entries into the
    ``{"key": title, "value": sent_idx}`` dict form that the artifact's
    metric (``discrete_retrieval_eval``) and feedback callbacks
    (``provide_feedback_to_*``) consume via ``doc["key"]``.
    """
    out = []
    for fact in raw:
        if isinstance(fact, dict) and "key" in fact:
            out.append(fact)
        elif isinstance(fact, list | tuple) and len(fact) >= 1:
            title = fact[0]
            sent_idx = fact[1] if len(fact) > 1 else None
            out.append({"key": title, "value": sent_idx})
        else:
            raise ValueError(f"Unrecognised supporting_fact entry: {fact!r}")
    return out


def _count_unique_docs(example: dict) -> int:
    return len({fact["key"] for fact in example["supporting_facts"]})


def load_hover_dataset(
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
):
    """Load the artifact-aligned HoVer 3-hop split.

    Replicates ``gepa_artifact.benchmarks.hover.hoverBench`` (lite mode)
    preprocessing, but without going through ``datasets.load_dataset``
    (whose script-based ``hover`` loader is broken on modern HF
    ``datasets`` versions). Concretely:

    - Downloads ``hover_train_release_v1.1.json`` from the upstream
      GitHub repo to ``gepa-artifact/gepa_artifact/benchmarks/hover/``
      (cached after first call).
    - Converts ``supporting_facts`` from ``[[title, sent_idx], ...]`` to
      ``[{"key": title, "value": sent_idx}, ...]`` (matches the artifact
      metric / feedback contract).
    - Filters to examples with exactly 3 unique supporting-doc titles
      (3-hop subset).
    - Shuffles with ``random.Random(0)``.
    - Splits ``test = data[:0.4*N]``, ``val = data[0.4*N:0.8*N]``,
      ``train = data[0.8*N:]``.
    - Trims to 150 / 300 / 300 using ``random.Random(1).sample``
      (artifact "lite" trim).
    - Truncates further with head-slice if caller passes smaller sizes.

    Returns ``(trainset, valset, testset)`` as lists of ``dspy.Example``
    with ``claim`` as the input field and ``supporting_facts`` /
    ``label`` as gold fields.
    """
    cache_path = _download_hover_train_json()
    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    reformatted: list[dict] = []
    for example in raw:
        normalised = {
            "claim": example["claim"],
            "supporting_facts": _normalise_supporting_facts(example["supporting_facts"]),
            "label": example["label"],
        }
        if _count_unique_docs(normalised) == 3:
            reformatted.append(normalised)

    rng_shuffle = random.Random()
    rng_shuffle.seed(0)
    rng_shuffle.shuffle(reformatted)

    n = len(reformatted)
    test_split = reformatted[: int(0.4 * n)]
    val_split = reformatted[int(0.4 * n):int(0.8 * n)]
    train_split = reformatted[int(0.8 * n):]

    def _trim(dataset: list[dict], lite_cap: int) -> list[dict]:
        if lite_cap is None or lite_cap >= len(dataset):
            return dataset
        rng_trim = random.Random()
        rng_trim.seed(1)
        return rng_trim.sample(dataset, lite_cap)

    # Artifact "lite" caps from gepa_artifact.benchmarks.benchmark.Benchmark.
    train_split = _trim(train_split, 150)
    val_split = _trim(val_split, 300)
    test_split = _trim(test_split, 300)

    def _to_examples(records: list[dict]) -> list[dspy.Example]:
        return [dspy.Example(**r).with_inputs("claim") for r in records]

    train_set = _to_examples(train_split)
    val_set = _to_examples(val_split)
    test_set = _to_examples(test_split)

    if train_size is not None and train_size < len(train_set):
        train_set = train_set[:train_size]
    if val_size is not None and val_size < len(val_set):
        val_set = val_set[:val_size]
    if test_size is not None and test_size < len(test_set):
        test_set = test_set[:test_size]

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Program (4 predictors)
# ---------------------------------------------------------------------------


def build_hover_program(
    instructions: dict[str, str | None] | None = None,
):
    """Build an artifact-aligned ``HoverMultiHop`` with optional per-stage
    instructions.

    ``instructions`` is a dict mapping a subset of ``STAGE_NAMES`` to
    the system instruction string to install on that predictor's
    ``ChainOfThought``. Missing or ``None`` entries leave the artifact
    default instruction (DSPy's auto-generated signature instructions)
    untouched.

    This matches the GEPA optimization surface of optimizing the four
    predictor instructions independently.
    """
    program_mod = _hover_program_module()
    program = program_mod.HoverMultiHop()

    if not instructions:
        return program

    for stage in STAGE_NAMES:
        text = instructions.get(stage)
        if text is None:
            continue
        predictor = getattr(program, stage)
        sig = cast(Any, predictor.predict.signature)
        predictor.predict.signature = sig.with_instructions(text)
    return program


def build_hover_program_four_stage(
    summarize1: str | None,
    create_query_hop2: str | None,
    summarize2: str | None,
    create_query_hop3: str | None,
):
    """Convenience wrapper around :func:`build_hover_program` taking the
    four stage instructions positionally, matching the order used in the
    tagged output format produced by the generator LM.
    """
    return build_hover_program(
        {
            "summarize1": summarize1,
            "create_query_hop2": create_query_hop2,
            "summarize2": summarize2,
            "create_query_hop3": create_query_hop3,
        }
    )


def artifact_default_instructions() -> dict[str, str]:
    """Return the artifact-default DSPy ChainOfThought instructions for
    each of the four HoVer predictors. Useful for reporting / wandb
    config and for sanity-checking the optimization baseline.
    """
    program = _hover_program_module().HoverMultiHop()
    out: dict[str, str] = {}
    for stage in STAGE_NAMES:
        sig = cast(Any, getattr(program, stage).predict.signature)
        out[stage] = sig.instructions
    return out


# ---------------------------------------------------------------------------
# Four-stage metaprompt output format
# ---------------------------------------------------------------------------


def four_stage_output_format() -> str:
    """Exact tagged output format the generator LM must follow.

    The four sections appear in execution order
    (``summarize1`` → ``create_query_hop2`` → ``summarize2`` →
    ``create_query_hop3``) and each is consumed by the predictor of the
    same name.
    """
    return (
        f"{_STAGE_TAGS['summarize1']}\n"
        "...stage 1 (summarize first-hop passages) instruction...\n"
        f"{_STAGE_TAGS['create_query_hop2']}\n"
        "...stage 2 (write hop-2 retrieval query) instruction...\n"
        f"{_STAGE_TAGS['summarize2']}\n"
        "...stage 3 (summarize second-hop passages) instruction...\n"
        f"{_STAGE_TAGS['create_query_hop3']}\n"
        "...stage 4 (write hop-3 retrieval query) instruction..."
    )


def parse_four_stage_prompt(text: str) -> dict[str, str] | None:
    """Parse a generator LM output into the four stage instructions.

    Returns a dict ``{stage_name: instruction}`` covering every entry of
    :data:`STAGE_NAMES` if and only if all four tags appear in the
    expected order and each section is non-empty after stripping.
    Otherwise returns ``None``.
    """
    if not text:
        return None
    ordered_tags = [_STAGE_TAGS[s] for s in STAGE_NAMES]
    indices: list[int] = []
    for tag in ordered_tags:
        idx = text.find(tag)
        if idx == -1:
            return None
        indices.append(idx)
    # Tags must appear in the canonical execution order.
    if any(indices[i] >= indices[i + 1] for i in range(len(indices) - 1)):
        return None

    sections: dict[str, str] = {}
    for i, stage in enumerate(STAGE_NAMES):
        start = indices[i] + len(ordered_tags[i])
        end = indices[i + 1] if i + 1 < len(indices) else len(text)
        section = text[start:end].strip()
        if not section:
            return None
        sections[stage] = section
    return sections


# ---------------------------------------------------------------------------
# Metric (artifact's discrete retrieval eval)
# ---------------------------------------------------------------------------


def _artifact_discrete_retrieval_eval() -> Callable:
    return _hover_utils_module().discrete_retrieval_eval


def hover_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Artifact's ``discrete_retrieval_eval`` lifted to the GEPA-compatible
    ``(gold, pred, trace, pred_name, pred_trace)`` signature.

    Returns ``True`` iff the set of gold supporting-doc titles is a
    subset of the titles found in ``pred.retrieved_docs`` (titles are
    ``dspy.evaluate.normalize_text`` normalised).
    """
    return _artifact_discrete_retrieval_eval()(gold, pred, trace)


__all__ = [
    "STAGE_NAMES",
    "TRAIN_SIZE",
    "VAL_SIZE",
    "TEST_SIZE",
    "load_hover_dataset",
    "build_hover_program",
    "build_hover_program_four_stage",
    "artifact_default_instructions",
    "four_stage_output_format",
    "parse_four_stage_prompt",
    "hover_metric",
]
