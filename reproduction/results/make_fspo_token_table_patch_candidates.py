from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "fspo_openrouter_recovery_confidence_v3.csv"
WANDB = ROOT / "fspo_wandb_history_tokens_per_run.csv"
OUT = ROOT / "fspo_token_table_patch_candidates_no_gpt4o.csv"

TABLE_ROWS = [
    ("GPT-4.1-mini", "IFBench", "FSPO", "IFBench"),
    ("GPT-4.1-mini", "Hover", "FSPO", "Hover"),
    ("GPT-4.1-mini", "PUPA / Papillon", "FSPO", "PUPA"),
    ("GPT-4.1-mini", "AIME-2025", "FSPO-Fixed15", "AIME-2025"),
    ("GPT-4.1-mini", "LiveBench-Math", "FSPO", "LiveBench-Math"),
    ("GPT-4.1-mini", "HotpotQA", "FSPO", "HotpotQA"),
    ("Qwen3-8B", "AIME-2025", "FSPO-base", "AIME-2025"),
    ("Qwen3-8B", "AIME-2025", "FSPO-Fixed15", "AIME-2025"),
    ("Qwen3-8B", "AIME-2025", "FSPO-Fixed45", "AIME-2025"),
    ("Qwen3-8B", "LiveBench-Math", "FSPO", "LiveBench-Math"),
    ("Qwen3-8B", "HotpotQA", "FSPO", "HotpotQA"),
    ("Qwen3-8B", "PUPA / Papillon", "FSPO", "PUPA"),
]

METHOD_HINTS = {
    "FSPO-base": ["dynamic_fewshot_20iter_3shot"],
    "FSPO-Fixed15": ["fixed15"],
    "FSPO-Fixed45": ["fixed45"],
}


def pick_candidate(conf, model, task_key, method):
    d = conf[(conf["model_short"] == model) & (conf["task"] == task_key)].copy()
    if d.empty:
        return None, "no_run"
    hints = METHOD_HINTS.get(method)
    if hints:
        mask = d["run_name"].astype(str).str.lower().map(lambda s: any(h.lower() in s for h in hints))
        d = d[mask].copy()
        if d.empty:
            return None, "no_method_matched_run"
    base_usable = d["confidence"].isin(["A_wandb_direct", "A_clean_time_model", "B_same_task_key_mix"])
    usable = d[base_usable].copy()
    if usable.empty:
        best = d.iloc[0]
        return best, "not_usable_" + str(best["confidence"])
    priority = {"A_wandb_direct": 0, "A_clean_time_model": 1, "B_same_task_key_mix": 2}
    usable["_priority"] = usable["confidence"].map(priority)
    usable = usable.sort_values(["_priority", "matched_requests"], ascending=[True, False])
    return usable.iloc[0], "usable"


def main():
    conf = pd.read_csv(CONF)
    conf = conf[conf["model_short"] != "GPT-4o-mini"].copy()
    wandb = pd.read_csv(WANDB)
    wandb = wandb.rename(columns={
        "input_tokens": "wandb_input_tokens",
        "output_tokens": "wandb_output_tokens",
        "estimated_cost_usd": "wandb_cost_usd",
    })
    conf = conf.merge(wandb[["run_id", "wandb_input_tokens", "wandb_output_tokens", "wandb_cost_usd"]], on="run_id", how="left")

    rows = []
    for model, table_task, method, task_key in TABLE_ROWS:
        cand, status = pick_candidate(conf, model, task_key, method)
        item = {
            "model": model,
            "table_task": table_task,
            "method": method,
            "status": status,
            "run_id": "",
            "run_name": "",
            "confidence": "",
            "source": "",
            "n_tok_over_n": "0 / 1",
            "input_m": "--",
            "output_m": "--",
            "cost_usd": "--",
            "raw_input_tokens": "",
            "raw_output_tokens": "",
            "raw_cost_usd": "",
            "reason": "",
        }
        if cand is None:
            item["reason"] = status
            rows.append(item)
            continue
        item["run_id"] = cand["run_id"]
        item["run_name"] = cand["run_name"]
        item["confidence"] = cand["confidence"]
        if status != "usable":
            item["reason"] = status
            rows.append(item)
            continue
        if cand["confidence"] == "A_wandb_direct":
            raw_in = cand["wandb_input_tokens"]
            raw_out = cand["wandb_output_tokens"]
            raw_cost = cand["wandb_cost_usd"]
            item["source"] = "WandB history cumulative"
        else:
            raw_in = cand["input_tokens"]
            raw_out = cand["output_tokens"]
            raw_cost = cand["cost_usd"]
            item["source"] = "OpenRouter clean time-window + model"
        item["n_tok_over_n"] = "1 / 1"
        item["raw_input_tokens"] = int(round(float(raw_in)))
        item["raw_output_tokens"] = int(round(float(raw_out)))
        item["raw_cost_usd"] = float(raw_cost)
        item["input_m"] = f"{float(raw_in) / 1e6:.2f}"
        item["output_m"] = f"{float(raw_out) / 1e6:.2f}"
        item["cost_usd"] = f"\\${float(raw_cost):.2f}"
        rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
