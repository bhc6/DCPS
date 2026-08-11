import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WINDOWS = ROOT / "fspo_run_time_windows.csv"
OUT_RUN = ROOT / "fspo_openrouter_matched_per_run.csv"
OUT_CELL = ROOT / "fspo_openrouter_matched_per_cell.csv"

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


def find_col(df, candidates, contains=None):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if contains:
        for c in df.columns:
            cl = c.lower()
            if all(x in cl for x in contains):
                return c
    return None


def parse_time(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def norm_task(row):
    text = f"{row.get('project', '')} {row.get('run_name', '')}".lower()
    for key, task in TASK_HINTS:
        if key in text:
            return task
    return ""


def norm_model_short(model):
    m = str(model).lower()
    if "gpt-4.1" in m:
        return "GPT-4.1-mini"
    if "gpt-4o" in m:
        return "GPT-4o-mini"
    if "qwen3-8b" in m:
        return "Qwen3-8B"
    return str(model)


def model_match(series, solver_model):
    aliases = MODEL_ALIASES.get(str(solver_model), [str(solver_model)])
    text = series.astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for alias in aliases:
        mask = mask | text.str.contains(alias.lower(), regex=False, na=False)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("activity_csv", help="CSV exported from OpenRouter Activity/Usage page")
    parser.add_argument("--pad-mins", type=float, default=5.0, help="minutes added before start and after end of each WandB run")
    args = parser.parse_args()

    activity = pd.read_csv(args.activity_csv, low_memory=False)
    windows = pd.read_csv(WINDOWS, low_memory=False)

    time_col = find_col(activity, ["created_at", "created", "timestamp", "time", "date"], contains=["time"])
    model_col = find_col(activity, ["model", "model_name", "model_slug"], contains=["model"])
    prompt_col = find_col(activity, ["prompt_tokens", "input_tokens", "tokens_prompt", "native_tokens_prompt"], contains=["token"])
    completion_col = find_col(activity, ["completion_tokens", "output_tokens", "tokens_completion", "native_tokens_completion"], contains=["completion"])
    cost_col = find_col(activity, ["cost", "total_cost", "usage", "credits"], contains=["cost"])
    id_col = find_col(activity, ["id", "generation_id", "request_id"])

    missing = [name for name, col in [("time", time_col), ("model", model_col), ("prompt/input tokens", prompt_col), ("completion/output tokens", completion_col)] if col is None]
    if missing:
        raise SystemExit(f"Cannot identify required OpenRouter columns: {missing}. Columns are: {list(activity.columns)}")

    activity = activity.copy()
    activity["_time"] = parse_time(activity[time_col])
    activity["_input_tokens"] = pd.to_numeric(activity[prompt_col], errors="coerce").fillna(0)
    activity["_output_tokens"] = pd.to_numeric(activity[completion_col], errors="coerce").fillna(0)
    activity["_cost"] = pd.to_numeric(activity[cost_col], errors="coerce") if cost_col else pd.NA

    out = []
    pad = pd.Timedelta(minutes=args.pad_mins)
    for _, run in windows.iterrows():
        start = parse_time(run.get("created_at"))
        runtime = pd.to_numeric(run.get("runtime_mins"), errors="coerce")
        if pd.isna(start) or pd.isna(runtime):
            continue
        end = start + pd.Timedelta(minutes=float(runtime))
        solver_model = run.get("config_solver_model", "")
        mask = activity["_time"].between(start - pad, end + pad) & model_match(activity[model_col], solver_model)
        hit = activity.loc[mask]
        item = {
            "project": run.get("project", ""),
            "run_id": run.get("run_id", ""),
            "run_name": run.get("run_name", ""),
            "task": norm_task(run),
            "solver_model": solver_model,
            "model_short": norm_model_short(solver_model),
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "pad_mins": args.pad_mins,
            "matched_requests": len(hit),
            "input_tokens": hit["_input_tokens"].sum(),
            "output_tokens": hit["_output_tokens"].sum(),
            "cost_usd": hit["_cost"].sum() if cost_col else pd.NA,
        }
        if id_col:
            item["matched_ids"] = " ".join(hit[id_col].dropna().astype(str).head(20).tolist())
        out.append(item)

    run_df = pd.DataFrame(out)
    run_df.to_csv(OUT_RUN, index=False)

    valid = run_df[run_df["matched_requests"] > 0].copy() if not run_df.empty else run_df
    if not valid.empty:
        cell = valid.groupby(["model_short", "task"], dropna=False).agg(
            n_runs=("run_id", "size"),
            matched_requests=("matched_requests", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
        ).reset_index()
    else:
        cell = pd.DataFrame(columns=["model_short", "task", "n_runs", "matched_requests", "input_tokens", "output_tokens", "cost_usd"])
    cell.to_csv(OUT_CELL, index=False)

    print(f"OpenRouter columns: time={time_col}, model={model_col}, input={prompt_col}, output={completion_col}, cost={cost_col}, id={id_col}")
    print(f"wrote {OUT_RUN}")
    print(f"wrote {OUT_CELL}")
    print(cell.to_string(index=False))


if __name__ == "__main__":
    main()
