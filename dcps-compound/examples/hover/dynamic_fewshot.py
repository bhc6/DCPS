"""Dynamic few-shot prompt search baseline for HoVer (qwen3-8b).

Mirrors ``examples/ifbench/dynamic_fewshot.py`` but adapted to HoVer's
4-predictor optimisation surface: the generator LM emits **four** stage
instructions per iteration (one for each of ``summarize1``,
``create_query_hop2``, ``summarize2``, ``create_query_hop3``), which are
installed on a fresh ``HoverMultiHop`` and scored on a fixed validation
pool.

Dataset split, fixed validation-pool strategy, program, and metric are
artifact-aligned via ``examples.hover.artifact_aligned`` (see that
module's docstring for the exact split policy and metric).
"""

import argparse
import json
import os
import random
import threading
from contextlib import contextmanager
from pathlib import Path

import dspy
import litellm
from dotenv import load_dotenv
from dspy.adapters import ChatAdapter

import wandb
from examples.hover.artifact_aligned import (
    STAGE_NAMES,
    build_hover_program,
    four_stage_output_format,
    hover_metric,
    load_hover_dataset,
    parse_four_stage_prompt,
)

# ---------------------------------------------------------------------------
# Checkpointing (resume support)
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"


def _checkpoint_path(run_name: str) -> Path:
    """Per-run checkpoint file. Keyed by ``wandb_run_name`` so the qwen3-8b
    and gpt-4.1-mini variants (and different iter/shot configs) never share
    a checkpoint.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
    return _CHECKPOINT_DIR / f"{safe}.jsonl"


def _load_checkpoint(run_name: str) -> list[dict]:
    """Load completed iterations from the checkpoint file, if any.

    Each line is one iteration's result dict. Returns them ordered by
    ``iteration``. Tolerant of a truncated final line (e.g. process killed
    mid-write): malformed trailing JSON is skipped.
    """
    path = _checkpoint_path(run_name)
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


def _append_checkpoint(run_name: str, record: dict) -> None:
    """Append one completed-iteration record to the checkpoint file."""
    path = _checkpoint_path(run_name)
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
PHASE_GENERATE = "optimize_generate"   # generator LM writes the 4-stage prompt
PHASE_OPT_EVAL = "optimize_eval"       # solver scored on the validation pool
PHASE_TEST_EVAL = "test_eval"          # solver scored on the held-out test set
_PHASES = (PHASE_GENERATE, PHASE_OPT_EVAL, PHASE_TEST_EVAL)
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "calls")


def _empty_bucket() -> dict:
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
        self._buckets = {p: _empty_bucket() for p in _PHASES}

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
            bb, ab = before.get(p, _empty_bucket()), after[p]
            out[p] = {
                k: round(ab[k] - bb[k], 6) if isinstance(ab[k], float) else ab[k] - bb[k]
                for k in _USAGE_FIELDS
            }
        return out


def _sum_buckets(snap: dict, phases=_PHASES) -> dict:
    """Collapse selected phase buckets of a snapshot into one total bucket."""
    out = _empty_bucket()
    for p in phases:
        b = snap.get(p, _empty_bucket())
        for k in _USAGE_FIELDS:
            out[k] += b[k]
    out["cost_usd"] = round(out["cost_usd"], 6)
    return out


def _add_snapshots(a: dict, b: dict) -> dict:
    """Element-wise add two full per-phase snapshots (for carry-over)."""
    out = {}
    for p in _PHASES:
        ab, bb = a.get(p, _empty_bucket()), b.get(p, _empty_bucket())
        out[p] = {k: ab.get(k, 0) + bb.get(k, 0) for k in _USAGE_FIELDS}
        out[p]["cost_usd"] = round(out[p]["cost_usd"], 6)
    return out


# Single process-wide tracker, registered once on litellm's success path.
USAGE = UsageTracker()
_TRACKER_REGISTERED = False


def _ensure_usage_callback() -> None:
    """Register the usage tracker as a litellm success callback exactly once."""
    global _TRACKER_REGISTERED
    if _TRACKER_REGISTERED:
        return
    litellm.success_callback = [*list(litellm.success_callback or []), USAGE.record]
    _TRACKER_REGISTERED = True


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _build_solver(stage_instructions: dict[str, str]):
    """Return a fresh ``HoverMultiHop`` whose four predictors use the
    given stage instructions, matching the GEPA optimisation surface of
    optimising the four predictor instructions independently.
    """
    return build_hover_program(stage_instructions)


def evaluate_on_dataset(stage_instructions: dict[str, str], dataset, num_threads: int = 16) -> float:
    solver = _build_solver(stage_instructions)
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=hover_metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=999_999,
        failure_score=0.0,
        provide_traceback=False,
    )
    result = evaluator(solver)
    return result.score / 100.0


# ---------------------------------------------------------------------------
# Few-shot sampling & metaprompt
# ---------------------------------------------------------------------------


def sample_fewshot_examples(trainset, num_examples: int = 3) -> str:
    """Sample HoVer few-shot examples in the format: ``claim`` + gold
    supporting-doc titles. We never expose passages or labels — just the
    multi-hop *retrieval target*, which is what every stage's instruction
    is ultimately judged against.
    """
    sampled = random.sample(trainset, min(num_examples, len(trainset)))
    out = ""
    for i, ex in enumerate(sampled, 1):
        gold_titles = [doc["key"] for doc in ex.supporting_facts]
        out += f"Example {i}:\n"
        out += f"Claim: {ex.claim}\n"
        out += f"Gold supporting-doc titles ({len(gold_titles)} hops): {gold_titles}\n\n"
    return out.strip()


def create_metaprompt(fewshot_examples: str) -> str:
    return f"""You are an expert prompt engineer for a multi-hop retrieval AI system.

The system you are designing prompts for performs **3-hop retrieval** to verify a claim. Its architecture is fixed and consists of four LLM-driven stages, interleaved with BM25 retrieval over Wikipedia abstracts:

  HOP 1: BM25 retrieve k=7 docs using the raw claim
         -> [Stage 1 — summarize1] summarises hop-1 passages w.r.t. the claim
            (inputs: claim, passages; output: summary)
  HOP 2: [Stage 2 — create_query_hop2] writes a BM25 query for hop 2
            (inputs: claim, summary_1; output: query)
         BM25 retrieve k=7 docs
         -> [Stage 3 — summarize2] summarises hop-2 passages, conditioned on hop-1 summary
            (inputs: claim, context, passages; output: summary)
  HOP 3: [Stage 4 — create_query_hop3] writes a BM25 query for hop 3
            (inputs: claim, summary_1, summary_2; output: query)
         BM25 retrieve k=10 docs

The final score is whether the **set of gold supporting-doc titles is fully covered by the union of all retrieved docs across the three hops**.
Stages 1 and 3 do NOT directly retrieve — they steer the next stage's query. Stages 2 and 4 directly determine retrieval recall.

Based on the few-shot examples below, design FOUR system prompts — one per stage. Each prompt should:

1. Be tailored to that stage's specific input/output schema and role in the retrieval chain (summarisation vs. query-writing, with/without prior context).
2. Encourage strategies that maximise multi-hop retrieval *recall* of the gold supporting docs.
   - For query stages: prefer entity bridges and disambiguating context.
   - For summary stages: surface entities, relations, and unresolved references that the next query must chase.
3. Be general enough to apply to new claims drawn from the same distribution.
4. Avoid copying any specific example claim or title into the prompt.

Output format (exact, no extra commentary, no code fences). The four sections MUST appear in this order:

{four_stage_output_format()}

Here are the few-shot examples to analyse:

{fewshot_examples}

Now, generate the four stage prompts following the output format above:"""


def generate_prompt_set_with_llm(metaprompt: str) -> dict[str, str] | None:
    """Ask the generator LM to emit a four-stage instruction set.

    Returns ``None`` if generation fails or the output cannot be parsed
    into all four tagged sections.
    """
    class PromptGenerationSignature(dspy.Signature):
        metaprompt = dspy.InputField(
            desc="Metaprompt for generating a set of four HoVer multi-hop retrieval system prompts, one per predictor stage."
        )
        generated_prompt = dspy.OutputField(
            desc="Four system prompts in the required four-stage tagged format, to be installed as the instructions for summarize1 / create_query_hop2 / summarize2 / create_query_hop3."
        )

    generator = dspy.ChainOfThought(PromptGenerationSignature)
    try:
        result = generator(metaprompt=metaprompt)
    except Exception as exc:
        print(f"[generate_prompt_set_with_llm] generator failed: {exc!r}")
        return None
    return parse_four_stage_prompt(result.generated_prompt or "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_default_lms(api_key: str, api_base: str | None):
    """qwen3-8b default solver+generator (paper-aligned sampling)."""
    solver_lm = dspy.LM(
        "openrouter/qwen/qwen3-8b",
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        num_retries=0,
        cache=False,
        extra_body={"top_k": 20, "provider": {"only": ["alibaba"]}},
    )
    generator_lm = dspy.LM(
        "openrouter/qwen/qwen3-8b",
        api_key=api_key,
        api_base=api_base,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        num_retries=0,
        cache=False,
        extra_body={"top_k": 20, "provider": {"only": ["alibaba"]}},
    )
    return solver_lm, generator_lm


def run(
    *,
    solver_lm: dspy.LM,
    generator_lm: dspy.LM,
    num_iterations: int,
    num_fewshot: int,
    top_k: int,
    val_sample_size: int,
    num_threads: int,
    wandb_run_name: str,
    api_key_env_for_logging: str,
    resume: bool = True,
):
    """Shared dynamic-fewshot driver. The gpt-4.1-mini variant calls
    this with different LMs but identical pipeline / split logic.

    When ``resume`` is true (the default), iterations completed in a prior
    run are loaded from the per-run checkpoint file and skipped, so an
    interrupted run picks up where it left off. Pass ``resume=False`` to
    delete any existing checkpoint and start cleanly from iteration 1."""
    _ensure_usage_callback()
    dspy.configure(
        adapter=ChatAdapter(use_json_adapter_fallback=False),
        lm=solver_lm,
    )

    trainset, full_valset, testset = load_hover_dataset()
    valset = full_valset[:val_sample_size]
    print(f"[OK] Fixed validation pool: {len(valset)} / {len(full_valset)} (head-slice of paper valset)")

    wandb.init(
        project="hover-dynamic-fewshot",
        name=wandb_run_name,
        config={
            "task": "hover",
            "num_iterations": num_iterations,
            "num_fewshot_examples": num_fewshot,
            "top_k_prompts": top_k,
            "val_sample_size": val_sample_size,
            "solver_model": solver_lm.model,
            "generator_model": generator_lm.model,
            "metaprompt_style": "four_stage_multi_hop_retrieval",
            "optimization_surface": "four_predictor_instructions",
            "stage_names": list(STAGE_NAMES),
            "sampling_temperature": solver_lm.kwargs.get("temperature"),
            "top_p": solver_lm.kwargs.get("top_p"),
            "max_tokens": solver_lm.kwargs.get("max_tokens"),
            "extra_body": solver_lm.kwargs.get("extra_body"),
            "trainset_size": len(trainset),
            "full_valset_size": len(full_valset),
            "valset_size": len(valset),
            "testset_size": len(testset),
            "val_strategy": "fixed_head_slice_of_paper_valset",
            "split_alignment": "identical_to_gepa_artifact",
            "data_source": "gepa-artifact/gepa_artifact/benchmarks/hover",
            "metric": "gepa_artifact.benchmarks.hover.hover_utils.discrete_retrieval_eval",
            "api_key_env": api_key_env_for_logging,
        },
    )

    print(f"Iterations: {num_iterations}, Few-shot: {num_fewshot}, Top K: {top_k}")
    print(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    ckpt_path = _checkpoint_path(wandb_run_name)
    if not resume and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[checkpoint] resume=False; removed existing checkpoint {ckpt_path.name}")

    completed = _load_checkpoint(wandb_run_name) if resume else []
    # Only iterations strictly within the requested range count as done.
    completed = [r for r in completed if r["iteration"] <= num_iterations]
    done_iterations = {r["iteration"] for r in completed}
    start_iteration = len(done_iterations)
    if completed:
        print(
            f"[checkpoint] resuming from {ckpt_path.name}: "
            f"{len(completed)} iteration(s) already done, "
            f"continuing at iteration {start_iteration + 1}"
        )

    all_results: list[dict] = list(completed)
    best_val_score = max((r["val_score"] for r in completed), default=0.0)
    prompt_table_cols = ["iteration", "val_score", *STAGE_NAMES, "fewshot_examples"]
    prompt_table_rows: list[list] = [
        [r["iteration"], r["val_score"], *(r["stage_instructions"][s] for s in STAGE_NAMES), r["fewshot_examples"]]
        for r in completed
    ]

    # Per-phase usage carried over from already-completed iterations so the
    # running totals stay monotonic across a resumed run. The live tracker
    # (USAGE) only counts this process's calls; we add the prior sums on top.
    # Old checkpoint records (pre-tracking) lack "usage_by_phase" and simply
    # contribute zeros.
    carried_usage = {p: _empty_bucket() for p in _PHASES}
    for r in completed:
        u = r.get("usage_by_phase")
        if u:
            carried_usage = _add_snapshots(carried_usage, u)

    for iteration in range(num_iterations):
        if (iteration + 1) in done_iterations:
            continue
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        usage_before = USAGE.snapshot()
        fewshot_examples = sample_fewshot_examples(trainset, num_fewshot)
        metaprompt = create_metaprompt(fewshot_examples)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=generator_lm,
        )
        with USAGE.phase(PHASE_GENERATE):
            prompt_set = generate_prompt_set_with_llm(metaprompt)

        dspy.configure(
            adapter=ChatAdapter(use_json_adapter_fallback=False),
            lm=solver_lm,
        )

        if prompt_set is None:
            print("[iteration] generator produced unparseable output, scoring 0.")
            stage_instr = {s: "" for s in STAGE_NAMES}
            val_score = 0.0
        else:
            stage_instr = prompt_set
            for s in STAGE_NAMES:
                print(f"  {s}: {stage_instr[s][:160]}...")
            try:
                with USAGE.phase(PHASE_OPT_EVAL):
                    val_score = evaluate_on_dataset(stage_instr, valset, num_threads=num_threads)
            except Exception as exc:
                print(f"[iteration] evaluation failed: {exc!r}; scoring 0.")
                val_score = 0.0
        print(f"Validation score (on {len(valset)} fixed samples): {val_score:.2%}")

        best_val_score = max(best_val_score, val_score)

        usage_after = USAGE.snapshot()
        iter_by_phase = USAGE.delta_since(usage_before, usage_after)
        gen_b = iter_by_phase[PHASE_GENERATE]
        eval_b = iter_by_phase[PHASE_OPT_EVAL]
        iter_total = _sum_buckets(iter_by_phase, (PHASE_GENERATE, PHASE_OPT_EVAL))
        # Cumulative optimization-side total (generate + opt eval), all iters.
        cum_by_phase = _add_snapshots(carried_usage, usage_after)
        cum_total = _sum_buckets(cum_by_phase, (PHASE_GENERATE, PHASE_OPT_EVAL))
        print(
            f"  [generate] in {gen_b['prompt_tokens']:,} / out {gen_b['completion_tokens']:,} tok, "
            f"${gen_b['cost_usd']:.4f}  |  "
            f"[opt-eval] in {eval_b['prompt_tokens']:,} / out {eval_b['completion_tokens']:,} tok, "
            f"${eval_b['cost_usd']:.4f}  ({eval_b['calls']} calls)"
        )
        print(
            f"  iter total {iter_total['total_tokens']:,} tok ${iter_total['cost_usd']:.4f}  |  "
            f"cumulative opt {cum_total['total_tokens']:,} tok ${cum_total['cost_usd']:.4f}"
        )

        prompt_table_rows.append([
            iteration + 1, val_score,
            *(stage_instr[s] for s in STAGE_NAMES),
            fewshot_examples,
        ])
        # wandb logging is best-effort: a transient I/O error (e.g. a wiped
        # %TEMP% dir after a reboot) must never crash the run — the checkpoint
        # below is the source of truth and is written regardless.
        try:
            wandb.log({
                "iteration": iteration + 1,
                "val_score": val_score,
                "best_val_score": best_val_score,
                # optimize / generate phase (input vs output)
                "iter_generate_input_tokens": gen_b["prompt_tokens"],
                "iter_generate_output_tokens": gen_b["completion_tokens"],
                "iter_generate_cost_usd": gen_b["cost_usd"],
                "iter_generate_calls": gen_b["calls"],
                # optimize / validation-eval phase (input vs output)
                "iter_opt_eval_input_tokens": eval_b["prompt_tokens"],
                "iter_opt_eval_output_tokens": eval_b["completion_tokens"],
                "iter_opt_eval_cost_usd": eval_b["cost_usd"],
                "iter_opt_eval_calls": eval_b["calls"],
                # iteration + cumulative optimization totals
                "iter_total_tokens": iter_total["total_tokens"],
                "iter_cost_usd": iter_total["cost_usd"],
                "cumulative_opt_input_tokens": cum_total["prompt_tokens"],
                "cumulative_opt_output_tokens": cum_total["completion_tokens"],
                "cumulative_opt_total_tokens": cum_total["total_tokens"],
                "cumulative_opt_cost_usd": cum_total["cost_usd"],
                "cumulative_opt_calls": cum_total["calls"],
                "iteration_prompts": wandb.Table(columns=prompt_table_cols, data=list(prompt_table_rows)),
            })
        except Exception as exc:
            print(f"[wandb] log failed for iteration {iteration + 1} (continuing): {exc!r}")

        all_results.append({
            "iteration": iteration + 1,
            "stage_instructions": stage_instr,
            "val_score": val_score,
            "fewshot_examples": fewshot_examples,
            "usage_by_phase": iter_by_phase,
        })
        _append_checkpoint(wandb_run_name, all_results[-1])

    all_results.sort(key=lambda x: x["val_score"], reverse=True)
    top_results = all_results[:top_k]

    print(f"\n=== Top {top_k} Prompt Sets from Validation ===")
    for i, r in enumerate(top_results):
        print(f"\nRank {i + 1} (Iteration {r['iteration']}): Val={r['val_score']:.2%}")
        for s in STAGE_NAMES:
            print(f"  {s}: {r['stage_instructions'][s][:160]}...")

    top_k_val_table = wandb.Table(
        columns=["rank", "iteration", "val_score", *STAGE_NAMES],
        data=[
            [i + 1, r["iteration"], r["val_score"], *(r["stage_instructions"][s] for s in STAGE_NAMES)]
            for i, r in enumerate(top_results)
        ],
    )
    try:
        wandb.log({"top_k_validation": top_k_val_table})
    except Exception as exc:
        print(f"[wandb] top_k_validation log failed (continuing): {exc!r}")

    print("\n=== Final Testing on Test Set ===")
    final_results: list[dict] = []
    final_cols = ["rank", "iteration", "val_score", "test_score", "val_test_gap", *STAGE_NAMES]
    final_rows: list[list] = []

    for i, r in enumerate(top_results):
        print(f"\nTesting Rank {i + 1} on test set...")
        try:
            with USAGE.phase(PHASE_TEST_EVAL):
                test_score = evaluate_on_dataset(r["stage_instructions"], testset, num_threads=num_threads)
        except Exception as exc:
            print(f"  [test] evaluation failed: {exc!r}; scoring 0.")
            test_score = 0.0
        print(f"  Test Score: {test_score:.2%}")

        final_rows.append([
            i + 1, r["iteration"], r["val_score"],
            test_score, r["val_score"] - test_score,
            *(r["stage_instructions"][s] for s in STAGE_NAMES),
        ])
        try:
            wandb.log({
                "test_rank": i + 1,
                "test_score": test_score,
                "test_val_score": r["val_score"],
                "val_test_gap": r["val_score"] - test_score,
                "final_results": wandb.Table(columns=final_cols, data=list(final_rows)),
            })
        except Exception as exc:
            print(f"[wandb] test-rank log failed for rank {i + 1} (continuing): {exc!r}")
        final_results.append({
            "rank": i + 1,
            "iteration": r["iteration"],
            "stage_instructions": r["stage_instructions"],
            "val_score": r["val_score"],
            "test_score": test_score,
        })

    print("\n=== FINAL REPORT ===")
    print(f"Generated {num_iterations} four-stage prompt sets via dynamic few-shot sampling")
    print(f"Selected top {top_k} based on validation performance")
    print(f"Tested on {len(testset)} test examples\n")

    print("Rank | Iter | Val Score | Test Score")
    print("-" * 50)
    for r in final_results:
        print(f"{r['rank']:4} | {r['iteration']:4} | {r['val_score']:9.2%} | {r['test_score']:10.2%}")

    best = max(final_results, key=lambda x: x["test_score"])
    avg_val = sum(r["val_score"] for r in final_results) / len(final_results)
    avg_test = sum(r["test_score"] for r in final_results) / len(final_results)

    print(f"\nBest: Iter {best['iteration']}, Val={best['val_score']:.2%}, Test={best['test_score']:.2%}")
    for s in STAGE_NAMES:
        print(f"Best {s}: {best['stage_instructions'][s]}")
    print(f"\nAvg Val (Top {top_k}): {avg_val:.2%}, Avg Test: {avg_test:.2%}, Gap: {avg_val - avg_test:.2%}")

    # Full per-phase usage = carried (prior iterations) + this process's calls.
    final_snap = _add_snapshots(carried_usage, USAGE.snapshot())
    gen = final_snap[PHASE_GENERATE]
    opt_eval = final_snap[PHASE_OPT_EVAL]
    test_eval = final_snap[PHASE_TEST_EVAL]
    optimize_total = _sum_buckets(final_snap, (PHASE_GENERATE, PHASE_OPT_EVAL))
    grand_total = _sum_buckets(final_snap, _PHASES)

    def _fmt(label: str, b: dict) -> str:
        return (
            f"  {label:<22} input {b['prompt_tokens']:>12,}  output {b['completion_tokens']:>12,}  "
            f"total {b['total_tokens']:>12,}  calls {b['calls']:>7,}  ${b['cost_usd']:.4f}"
        )

    print("\n=== TOKEN / COST USAGE (whole run, incl. resumed iterations) ===")
    print("OPTIMIZATION:")
    print(_fmt("generate (prompts)", gen))
    print(_fmt("eval (validation)", opt_eval))
    print(_fmt("optimize subtotal", optimize_total))
    print("TESTING:")
    print(_fmt("eval (test set)", test_eval))
    print("GRAND TOTAL:")
    print(_fmt("all phases", grand_total))

    if wandb.run is not None:
        summary = {
            "best_test_score": best["test_score"],
            "best_val_score": best["val_score"],
            "best_iteration": best["iteration"],
            "avg_val_score_top_k": avg_val,
            "avg_test_score_top_k": avg_test,
            "val_test_gap": avg_val - avg_test,
            "total_iterations": num_iterations,
            "optimization_surface": "four_predictor_instructions",
        }
        # Flatten every phase bucket + the two roll-ups into summary keys.
        for label, b in (
            ("generate", gen),
            ("opt_eval", opt_eval),
            ("optimize_total", optimize_total),
            ("test_eval", test_eval),
            ("grand_total", grand_total),
        ):
            summary[f"usage_{label}_input_tokens"] = b["prompt_tokens"]
            summary[f"usage_{label}_output_tokens"] = b["completion_tokens"]
            summary[f"usage_{label}_total_tokens"] = b["total_tokens"]
            summary[f"usage_{label}_cost_usd"] = b["cost_usd"]
            summary[f"usage_{label}_calls"] = b["calls"]
        for s in STAGE_NAMES:
            summary[f"best_{s}"] = best["stage_instructions"][s]
        try:
            wandb.run.summary.update(summary)
        except Exception as exc:
            print(f"[wandb] summary update failed (continuing): {exc!r}")

    try:
        wandb.finish()
    except Exception as exc:
        print(f"[wandb] finish failed (ignored): {exc!r}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic few-shot prompt search for HoVer (qwen3-8b)")
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

    solver_lm, generator_lm = _build_default_lms(api_key, api_base)

    run(
        solver_lm=solver_lm,
        generator_lm=generator_lm,
        num_iterations=args.num_iterations,
        num_fewshot=args.num_fewshot,
        top_k=args.top_k,
        val_sample_size=args.val_sample_size,
        num_threads=args.num_threads,
        wandb_run_name=f"dynamic_fewshot_qwen3_8b_{args.num_iterations}iter_{args.num_fewshot}shot",
        api_key_env_for_logging=api_key_env,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
