"""Shared resume-checkpointing + per-phase token/cost tracking for the
LiveBench-Math dynamic few-shot baselines.

Both ``dynamic_fewshot.py`` (qwen3-8b) and ``dynamic_fewshot_gpt41mini.py``
import from here so the two variants behave identically:

  - Per-run JSONL checkpoint (keyed by the wandb run name) so an interrupted
    run resumes from the next iteration instead of restarting.
  - A litellm success-callback that buckets every LM call's token usage and
    USD cost by phase (prompt generation vs validation eval vs test eval),
    splitting input (``prompt_tokens``) from output (``completion_tokens``).

Mirrors the implementation in ``examples/hover/dynamic_fewshot.py``.
"""

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Checkpointing (resume support)
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"


def checkpoint_path(run_name: str) -> Path:
    """Per-run checkpoint file. Keyed by the wandb run name so the qwen3-8b
    and gpt-4.1-mini variants (and different iter/shot configs) never share
    a checkpoint.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
    return _CHECKPOINT_DIR / f"{safe}.jsonl"


def load_checkpoint(run_name: str) -> list[dict]:
    """Load completed iterations from the checkpoint file, if any.

    Each line is one iteration's result dict, returned ordered by
    ``iteration``. Tolerant of a truncated final line (e.g. process killed
    mid-write): malformed trailing JSON is skipped.
    """
    path = checkpoint_path(run_name)
    if not path.exists():
        return []
    results: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[checkpoint] skipping malformed line in {path.name}")
                continue
    results.sort(key=lambda r: r["iteration"])
    return results


def append_checkpoint(run_name: str, record: dict) -> None:
    """Append one completed-iteration record to the checkpoint file."""
    path = checkpoint_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Token / cost tracking (litellm success callback), bucketed by phase
# ---------------------------------------------------------------------------

# Phases attributed independently. Within one iteration these never overlap in
# wall-clock time (the generator call finishes before validation eval starts,
# and validation eval blocks until all worker threads complete), so a single
# process-global "current phase" label is read by the callback — even from
# dspy's evaluation worker threads — and attributes every call correctly.
PHASE_GENERATE = "optimize_generate"   # generator LM writes the candidate prompt
PHASE_OPT_EVAL = "optimize_eval"       # solver scored on the validation pool
PHASE_TEST_EVAL = "test_eval"          # solver scored on the held-out test set
PHASES = (PHASE_GENERATE, PHASE_OPT_EVAL, PHASE_TEST_EVAL)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "calls")


def empty_bucket() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0}


class UsageTracker:
    """Thread-safe accumulator for token usage and USD cost, split by phase.

    Wired into litellm via a global success callback so it captures *every*
    LM call — solver predictions across all ``num_threads`` evaluation
    workers and generator calls alike. Each call is attributed to the
    currently-active phase (set by :meth:`phase`). ``prompt_tokens`` is the
    input side, ``completion_tokens`` the output side. Cost comes from
    litellm's per-call ``response_cost`` (OpenRouter passes USD through),
    falling back to litellm's price table.

    Counters are cumulative for the process; :meth:`snapshot` reads them and
    :meth:`delta_since` computes a per-iteration slice without resetting
    global state (so a resumed run's totals stay monotonic).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_phase = None
        self._buckets = {p: empty_bucket() for p in PHASES}

    @contextmanager
    def phase(self, name: str):
        """Attribute all LM calls made within this block to ``name``."""
        prev = self._current_phase
        self._current_phase = name
        try:
            yield
        finally:
            self._current_phase = prev

    def record(self, kwargs, completion_response, start_time, end_time) -> None:
        phase = self._current_phase
        if phase is None:
            return  # call outside any tracked phase; ignore

        prompt = completion = total = 0
        usage = getattr(completion_response, "usage", None)
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            total = getattr(usage, "total_tokens", 0) or (prompt + completion)

        try:
            cost = kwargs.get("response_cost")
            if cost is None:
                cost = litellm.completion_cost(completion_response=completion_response)
        except Exception:
            cost = 0.0
        cost = float(cost or 0.0)

        with self._lock:
            b = self._buckets[phase]
            b["prompt_tokens"] += prompt
            b["completion_tokens"] += completion
            b["total_tokens"] += total
            b["cost_usd"] += cost
            b["calls"] += 1

    def snapshot(self) -> dict:
        """Deep copy of all per-phase buckets (cost rounded)."""
        with self._lock:
            return {
                p: {
                    "prompt_tokens": b["prompt_tokens"],
                    "completion_tokens": b["completion_tokens"],
                    "total_tokens": b["total_tokens"],
                    "cost_usd": round(b["cost_usd"], 6),
                    "calls": b["calls"],
                }
                for p, b in self._buckets.items()
            }

    @staticmethod
    def delta_since(before: dict, after: dict) -> dict:
        """Per-phase diff of two snapshots."""
        out: dict = {}
        for p in after:
            bb, ab = before.get(p, empty_bucket()), after[p]
            out[p] = {
                k: round(ab[k] - bb[k], 6) if isinstance(ab[k], float) else ab[k] - bb[k]
                for k in USAGE_FIELDS
            }
        return out


def sum_buckets(snap: dict, phases=PHASES) -> dict:
    """Collapse selected phase buckets of a snapshot into one total bucket."""
    out = empty_bucket()
    for p in phases:
        b = snap.get(p, empty_bucket())
        for k in USAGE_FIELDS:
            out[k] += b[k]
    out["cost_usd"] = round(out["cost_usd"], 6)
    return out


def add_snapshots(a: dict, b: dict) -> dict:
    """Element-wise add two full per-phase snapshots (for carry-over)."""
    out = {}
    for p in PHASES:
        ab, bb = a.get(p, empty_bucket()), b.get(p, empty_bucket())
        out[p] = {k: ab.get(k, 0) + bb.get(k, 0) for k in USAGE_FIELDS}
        out[p]["cost_usd"] = round(out[p]["cost_usd"], 6)
    return out


# Single process-wide tracker, registered once on litellm's success path.
USAGE = UsageTracker()
_TRACKER_REGISTERED = False


def ensure_usage_callback() -> None:
    """Register the usage tracker as a litellm success callback exactly once."""
    global _TRACKER_REGISTERED
    if _TRACKER_REGISTERED:
        return
    litellm.success_callback = [*list(litellm.success_callback or []), USAGE.record]
    _TRACKER_REGISTERED = True
