from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ACTIVITY = ROOT.parent / "openrouter_activity_2026-05-19.csv"
WINDOWS = ROOT / "fspo_run_time_windows.csv"
OUT_RUN = ROOT / "fspo_openrouter_match_quality_per_run.csv"
OUT_CELL = ROOT / "fspo_openrouter_match_quality_per_cell.csv"

MODEL_ALIASES = {
    "openrouter/openai/gpt-4.1-mini": ["gpt-4.1-mini", "openai/gpt-4.1-mini", "openrouter/openai/gpt-4.1-mini"],
    "openrouter/openai/gpt-4o-mini": ["gpt-4o-mini", "openai/gpt-4o-mini", "openrouter/openai/gpt-4o-mini"],
    "openrouter/qwen/qwen3-8b": ["qwen3-8b", "qwen/qwen3-8b", "openrouter/qwen/qwen3-8b"],
}

TASK_HINTS = [
    ("hotpot", "HotpotQA"),
    ("ifbench", "IFBench"),
    ("hover", "Hover"),
    ("pupa", "PUPA"),
    ("papillon", "PUPA"),
    ("aime", "AIME-2025"),
    ("livebench", "LiveBench-Math"),
]

EXPECTED_KEY_HINTS = {
    "HotpotQA": ["hotpot"],
    "IFBench": ["ifbench"],
    "Hover": ["hover"],
    "PUPA": ["pupa", "project"],
    "AIME-2025": ["aime"],
    "LiveBench-Math": ["livebench", "mathbench"],
}


def model_short(model):
    s = str(model).lower()
    if "gpt-4.1" in s:
        return "GPT-4.1-mini"
    if "gpt-4o" in s:
        return "GPT-4o-mini"
    if "qwen3-8b" in s:
        return "Qwen3-8B"
    return str(model)


def model_match(series, solver_model):
    aliases = MODEL_ALIASES.get(str(solver_model), [str(solver_model)])
    text = series.astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for alias in aliases:
        mask = mask | text.str.contains(alias.lower(), regex=False, na=False)
    return mask


def task_name(row):
    text = f"{row.get('project', '')} {row.get('run_name', '')}".lower()
    for key, task in TASK_HINTS:
        if key in text:
            return task
    return ""


def key_expected(task, key):
    key = str(key).lower()
    hints = EXPECTED_KEY_HINTS.get(task, [])
    return any(h in key for h in hints)


def main():
    activity = pd.read_csv(ACTIVITY, low_memory=False)
    windows = pd.read_csv(WINDOWS)
    activity["created_at"] = pd.to_datetime(activity["created_at"], utc=True, errors="coerce")
    activity["tokens_prompt"] = pd.to_numeric(activity["tokens_prompt"], errors="coerce").fillna(0)
    activity["tokens_completion"] = pd.to_numeric(activity["tokens_completion"], errors="coerce").fillna(0)
    activity["cost_total"] = pd.to_numeric(activity["cost_total"], errors="coerce").fillna(0)

    rows = []
    for _, run in windows.iterrows():
        start = pd.to_datetime(run["created_at"], utc=True, errors="coerce")
        runtime = pd.to_numeric(run["runtime_mins"], errors="coerce")
        if pd.isna(start) or pd.isna(runtime):
            continue
        end = start + pd.Timedelta(minutes=float(runtime))
        task = task_name(run)
        hit = activity[activity["created_at"].between(start, end) & model_match(activity["model_permaslug"], run["config_solver_model"])]
        if hit.empty:
            rows.append({
                "project": run["project"],
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "task": task,
                "model_short": model_short(run["config_solver_model"]),
                "matched_requests": 0,
            })
            continue
        by_key = hit.groupby("api_key_name", dropna=False).agg(
            requests=("generation_id", "size"),
            input_tokens=("tokens_prompt", "sum"),
            output_tokens=("tokens_completion", "sum"),
            cost_usd=("cost_total", "sum"),
        ).reset_index().sort_values("requests", ascending=False)
        top_key = str(by_key.iloc[0]["api_key_name"])
        expected = by_key[by_key["api_key_name"].map(lambda x: key_expected(task, x))]
        expected_requests = int(expected["requests"].sum()) if not expected.empty else 0
        total_requests = int(by_key["requests"].sum())
        expected_frac = expected_requests / total_requests if total_requests else 0
        key_breakdown = "; ".join(
            f"{r.api_key_name}:{int(r.requests)} req,{int(r.input_tokens)} in,{int(r.output_tokens)} out,${float(r.cost_usd):.4f}"
            for _, r in by_key.head(8).iterrows()
        )
        rows.append({
            "project": run["project"],
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "task": task,
            "model_short": model_short(run["config_solver_model"]),
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "matched_requests": total_requests,
            "input_tokens": hit["tokens_prompt"].sum(),
            "output_tokens": hit["tokens_completion"].sum(),
            "cost_usd": hit["cost_total"].sum(),
            "n_api_keys": by_key.shape[0],
            "top_api_key": top_key,
            "top_key_request_frac": float(by_key.iloc[0]["requests"] / total_requests),
            "expected_key_request_frac": expected_frac,
            "key_breakdown": key_breakdown,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_RUN, index=False)
    cell = out.groupby(["model_short", "task"], dropna=False).agg(
        n_runs=("run_id", "size"),
        matched_requests=("matched_requests", "sum"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
        mean_api_keys=("n_api_keys", "mean"),
        min_expected_key_request_frac=("expected_key_request_frac", "min"),
    ).reset_index()
    cell.to_csv(OUT_CELL, index=False)
    print(out[["task", "model_short", "run_id", "matched_requests", "n_api_keys", "top_api_key", "top_key_request_frac", "expected_key_request_frac"]].to_string(index=False, max_rows=100))
    print(f"wrote {OUT_RUN}")
    print(f"wrote {OUT_CELL}")


if __name__ == "__main__":
    main()
