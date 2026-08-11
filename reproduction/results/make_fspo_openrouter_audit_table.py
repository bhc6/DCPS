from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "fspo_openrouter_recovery_confidence_v3.csv"
OUT = ROOT / "tables" / "tab_app_fspo_openrouter_audit.tex"


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&")


def fmt_m(x):
    return f"{float(x) / 1e6:.2f}"


def fmt_cost(x):
    return f"\\${float(x):.2f}"


def main():
    df = pd.read_csv(CONF)
    df = df[df["model_short"] != "GPT-4o-mini"].copy()
    df = df[df["confidence"].isin(["A_clean_time_model", "A_wandb_direct", "B_same_task_key_mix"])].copy()
    df = df.sort_values(["model_short", "task", "confidence", "run_id"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Audited FSPO OpenRouter/WandB token-recovery runs. "
        r"Rows shown here have either direct WandB cumulative-token fields "
        r"or a clean OpenRouter model/time-window match with no same-model "
        r"WandB overlap. GPT-4o-mini runs are excluded from the paper token "
        r"recovery.}"
    )
    lines.append(r"\label{tab:app_fspo_openrouter_audit}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llllrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Task} & \textbf{Run} & \textbf{Confidence} & \textbf{Req.} & \textbf{Input (M)} & \textbf{Output (M)} & \textbf{Cost} \\ \midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{esc(r['model_short'])} & {esc(r['task'])} & {esc(r['run_id'])} & {esc(r['confidence'])} & "
            f"{int(r['matched_requests'])} & {fmt_m(r['input_tokens'])} & {fmt_m(r['output_tokens'])} & {fmt_cost(r['cost_usd'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(df[["model_short", "task", "run_id", "confidence", "matched_requests", "input_tokens", "output_tokens", "cost_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
