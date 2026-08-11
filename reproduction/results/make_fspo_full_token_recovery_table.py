from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "fspo_openrouter_recovery_confidence_v3.csv"
WANDB = ROOT / "fspo_wandb_history_tokens_per_run.csv"
CSV_OUT = ROOT / "fspo_full_token_recovery_all_non_gpt4o.csv"
TEX_OUT = ROOT / "tables" / "tab_app_fspo_full_token_recovery.tex"


def esc(s):
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("$", "\\$")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def fmt_m(x):
    if pd.isna(x):
        return "--"
    return f"{float(x) / 1e6:.2f}"


def fmt_cost(x):
    if pd.isna(x):
        return "--"
    return f"\\${float(x):.2f}"


def compact_key_breakdown(s, max_len=80):
    s = str(s) if pd.notna(s) else ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def main():
    conf = pd.read_csv(CONF)
    conf = conf[conf["model_short"] != "GPT-4o-mini"].copy()

    wandb = pd.read_csv(WANDB)
    wandb = wandb.rename(
        columns={
            "input_tokens": "wandb_input_tokens",
            "output_tokens": "wandb_output_tokens",
            "estimated_cost_usd": "wandb_cost_usd",
        }
    )
    conf = conf.merge(
        wandb[["run_id", "wandb_input_tokens", "wandb_output_tokens", "wandb_cost_usd"]],
        on="run_id",
        how="left",
    )

    conf["openrouter_input_m"] = conf["input_tokens"].map(lambda x: float(x) / 1e6 if pd.notna(x) else pd.NA)
    conf["openrouter_output_m"] = conf["output_tokens"].map(lambda x: float(x) / 1e6 if pd.notna(x) else pd.NA)
    conf["wandb_input_m"] = conf["wandb_input_tokens"].map(lambda x: float(x) / 1e6 if pd.notna(x) else pd.NA)
    conf["wandb_output_m"] = conf["wandb_output_tokens"].map(lambda x: float(x) / 1e6 if pd.notna(x) else pd.NA)
    conf["key_breakdown_short"] = conf["key_breakdown"].map(compact_key_breakdown)

    csv_cols = [
        "confidence",
        "project",
        "task",
        "model_short",
        "run_id",
        "run_name",
        "matched_requests",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wandb_input_tokens",
        "wandb_output_tokens",
        "wandb_cost_usd",
        "n_api_keys",
        "top_api_key",
        "top_key_request_frac",
        "expected_key_request_frac",
        "n_fspo_overlaps",
        "n_non_fspo_overlaps",
        "overlap_run_ids",
        "non_fspo_overlap_ids",
        "non_fspo_overlap_names",
        "key_breakdown",
    ]
    conf[csv_cols].sort_values(["model_short", "task", "confidence", "run_id"]).to_csv(CSV_OUT, index=False)

    tex_df = conf.sort_values(["model_short", "task", "confidence", "run_id"])
    lines = []
    lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full FSPO token-recovery audit table for all non-GPT-4o-mini runs. "
        r"This table is intentionally inclusive: it lists clean, mixed-key, and overlapping "
        r"OpenRouter/WandB matches so that the reader can judge which rows to trust. "
        r"OpenRouter columns aggregate all requests in the run's model/time window; WandB "
        r"columns are populated only when cumulative-token fields were found directly in "
        r"run history.}"
    )
    lines.append(r"\label{tab:app_fspo_full_token_recovery}")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llllllllrrrrrlll@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Task} & \textbf{Run} & \textbf{Name} & \textbf{Conf.} & "
        r"\textbf{Top key} & \textbf{FSPO ov.} & \textbf{Non-FSPO ov.} & \textbf{Req.} & "
        r"\textbf{OR In} & \textbf{OR Out} & \textbf{OR Cost} & \textbf{WB In} & "
        r"\textbf{WB Out} & \textbf{WB Cost} & \textbf{Keys} \\ \midrule"
    )
    for _, r in tex_df.iterrows():
        lines.append(
            f"{esc(r['model_short'])} & {esc(r['task'])} & {esc(r['run_id'])} & "
            f"{esc(r['run_name'])} & {esc(r['confidence'])} & {esc(r.get('top_api_key', ''))} & "
            f"{int(r.get('n_fspo_overlaps', 0))} & {int(r.get('n_non_fspo_overlaps', 0))} & "
            f"{int(r.get('matched_requests', 0))} & {fmt_m(r.get('input_tokens'))} & "
            f"{fmt_m(r.get('output_tokens'))} & {fmt_cost(r.get('cost_usd'))} & "
            f"{fmt_m(r.get('wandb_input_tokens'))} & {fmt_m(r.get('wandb_output_tokens'))} & "
            f"{fmt_cost(r.get('wandb_cost_usd'))} & {esc(r.get('key_breakdown_short', ''))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\end{landscape}")
    TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {CSV_OUT}")
    print(f"wrote {TEX_OUT}")
    print(tex_df[["model_short", "task", "run_id", "confidence", "matched_requests", "input_tokens", "output_tokens", "cost_usd", "n_fspo_overlaps", "n_non_fspo_overlaps"]].to_string(index=False))


if __name__ == "__main__":
    main()
