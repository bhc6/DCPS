import math
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "awesome-prompt"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_wandb_data.csv"
OUT_RUN = ROOT / "fspo_wandb_history_tokens_per_run.csv"
OUT_CELL = ROOT / "fspo_wandb_history_tokens_per_cell.csv"

TOKEN_KEYS = [
    "input_tokens_cumulative",
    "output_tokens_cumulative",
    "total_tokens_cumulative",
    "optimization_input_tokens_cumulative",
    "optimization_output_tokens_cumulative",
    "optimization_total_tokens_cumulative",
    "summary_input_tokens_cumulative",
    "summary_output_tokens_cumulative",
    "summary_total_tokens_cumulative",
]

PRICE = {
    "openrouter/openai/gpt-4.1-mini": (0.40, 1.60),
    "openrouter/openai/gpt-4o-mini": (0.15, 0.60),
    "openrouter/qwen/qwen3-8b": (0.117, 0.455),
}

BENCH_MAP = {
    "HotpotQABench": "HotpotQA",
    "IFBench": "IFBench",
    "hoverBench": "Hover",
    "Papillon": "PUPA",
    "AIMEBench": "AIME-2025",
    "LiveBenchMathBench": "LiveBench-Math",
}


def norm_model(row):
    for col in ["config_solver_model", "config_model", "config_generator_model", "config_target_model", "config_agent_model"]:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            return v.strip()
    name = str(row.get("run_name", "")).lower()
    if "gpt41" in name or "gpt-4.1" in name:
        return "openrouter/openai/gpt-4.1-mini"
    if "gpt" in name:
        return "openrouter/openai/gpt-4o-mini"
    if "qwen" in name:
        return "openrouter/qwen/qwen3-8b"
    return ""


def norm_model_short(model):
    m = str(model).lower()
    if "gpt-4.1" in m:
        return "GPT-4.1-mini"
    if "gpt-4o" in m:
        return "GPT-4o-mini"
    if "qwen3-8b" in m:
        return "Qwen3-8B"
    return model


def norm_task(row):
    b = row.get("config_benchmark_name")
    if isinstance(b, str) and b in BENCH_MAP:
        return BENCH_MAP[b]
    s = f"{row.get('project', '')} {row.get('run_name', '')}".lower()
    if "hotpot" in s:
        return "HotpotQA"
    if "ifbench" in s:
        return "IFBench"
    if "hover" in s:
        return "Hover"
    if "pupa" in s or "papillon" in s:
        return "PUPA"
    if "aime" in s:
        return "AIME-2025"
    if "livebench" in s:
        return "LiveBench-Math"
    return ""


def max_numeric_from_frame(frame, keys):
    best = math.nan
    best_key = ""
    for key in keys:
        if key in frame.columns:
            vals = pd.to_numeric(frame[key], errors="coerce").dropna()
            vals = vals[vals > 0]
            if not vals.empty:
                val = float(vals.max())
                if math.isnan(best) or val > best:
                    best = val
                    best_key = key
    return best, best_key


def main():
    df = pd.read_csv(RAW, low_memory=False)
    text = df.get("project", pd.Series("", index=df.index)).astype(str) + " " + df.get("run_name", pd.Series("", index=df.index)).astype(str)
    mask = text.str.contains("dynamic|fewshot|fspo", case=False, regex=True, na=False)
    rows = df.loc[mask].copy()
    api = wandb.Api()
    out = []
    for _, row in rows.iterrows():
        project = row["project"]
        run_id = row["run_id"]
        path = f"{ENTITY}/{project}/{run_id}"
        item = {
            "project": project,
            "run_id": run_id,
            "run_name": row.get("run_name", ""),
            "task": norm_task(row),
            "solver_model": norm_model(row),
        }
        try:
            run = api.run(path)
            summary = dict(run.summary)
            hist = run.scan_history(keys=TOKEN_KEYS, page_size=500)
            hist_df = pd.DataFrame(list(hist))
            source_df = pd.concat([pd.DataFrame([summary]), hist_df], ignore_index=True, sort=False)
            in_val, in_key = max_numeric_from_frame(source_df, [k for k in TOKEN_KEYS if "input" in k])
            out_val, out_key = max_numeric_from_frame(source_df, [k for k in TOKEN_KEYS if "output" in k])
            total_val, total_key = max_numeric_from_frame(source_df, [k for k in TOKEN_KEYS if "total" in k])
            item.update({"input_tokens": in_val, "output_tokens": out_val, "total_tokens": total_val, "input_key": in_key, "output_key": out_key, "total_key": total_key, "error": ""})
        except Exception as exc:
            item.update({"input_tokens": math.nan, "output_tokens": math.nan, "total_tokens": math.nan, "input_key": "", "output_key": "", "total_key": "", "error": repr(exc)})
        model = item["solver_model"]
        pin, pout = PRICE.get(model, (math.nan, math.nan))
        if pd.notna(item["input_tokens"]) and pd.notna(item["output_tokens"]) and pd.notna(pin):
            item["estimated_cost_usd"] = item["input_tokens"] / 1e6 * pin + item["output_tokens"] / 1e6 * pout
        else:
            item["estimated_cost_usd"] = math.nan
        item["model_short"] = norm_model_short(model)
        out.append(item)
        print(f"{path}: in={item['input_tokens']} out={item['output_tokens']} cost={item['estimated_cost_usd']} err={item['error']}")
    out_df = pd.DataFrame(out)
    out_df.to_csv(OUT_RUN, index=False)
    valid = out_df[out_df["input_tokens"].notna() | out_df["output_tokens"].notna()].copy()
    if not valid.empty:
        cell = valid.groupby(["model_short", "task"], dropna=False).agg(
            n=("run_id", "size"),
            input_tokens=("input_tokens", "mean"),
            output_tokens=("output_tokens", "mean"),
            estimated_cost_usd=("estimated_cost_usd", "mean"),
        ).reset_index()
    else:
        cell = pd.DataFrame(columns=["model_short", "task", "n", "input_tokens", "output_tokens", "estimated_cost_usd"])
    cell.to_csv(OUT_CELL, index=False)
    print(f"wrote {OUT_RUN}")
    print(f"wrote {OUT_CELL}")


if __name__ == "__main__":
    main()
