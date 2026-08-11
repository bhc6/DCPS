from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "phase1_runtime_appendix_runs.csv"
OUT = ROOT / "results" / "tables" / "tab_app_phase1_runtime.tex"


def esc(value):
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.2f}"


def main():
    df = pd.read_csv(RUNS)
    df["runtime_mins"] = pd.to_numeric(df["runtime_mins"], errors="coerce")
    df["epochs"] = pd.to_numeric(df["epochs"], errors="coerce")
    df = df.dropna(subset=["runtime_mins"])

    grouped = (
        df.groupby(["dataset", "paper_method", "gpu", "epochs"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            seeds=("seed", lambda s: ",".join(sorted({str(int(float(x))) for x in s.dropna() if str(x) != ""}))),
            mean_runtime_mins=("runtime_mins", "mean"),
            median_runtime_mins=("runtime_mins", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "paper_method", "epochs"])
    )

    lines = []
    lines.append(r"\begin{longtable}{@{}llclrrr@{}}")
    caption = (
        r"\caption{Phase-1 GPU runtime audit by dataset on Gemma1.1-7B. "
        r"Rows include runs whose names explicitly contain the Gemma-to-Gemma Phase-1 pattern "
        r"(\texttt{gemma-1.1-7b-it\_google/gemma-1.1-7b-it} or "
        r"\texttt{gemma-1.1-7b-it\_TO\_gemma-1.1-7b-it}). "
        r"All listed runs use local GPU execution only; no API token or USD cost is reported. "
        r"GPU is recorded as A40 according to the Phase-1 experimental setup. "
        r"GFB/AlgPrompt runs are mapped to StablePrompt-RAE, while PPO-enabled runs are mapped "
        r"to StablePrompt-PPO.}\\"
    )
    lines.append(caption)
    lines.append(r"\label{tab:app_phase1_runtime}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Method} & \textbf{GPU} & \textbf{Epochs} & \textbf{$n$} & \textbf{Mean min} & \textbf{Median min} \\ \midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Method} & \textbf{GPU} & \textbf{Epochs} & \textbf{$n$} & \textbf{Mean min} & \textbf{Median min} \\ \midrule")
    lines.append(r"\endhead")
    for _, row in grouped.iterrows():
        epochs = "--" if pd.isna(row["epochs"]) else str(int(row["epochs"]))
        lines.append(
            f"{esc(row['dataset'])} & {esc(row['paper_method'])} & {esc(row['gpu'])} & {epochs} & "
            f"{int(row['n_runs'])} & {fmt(row['mean_runtime_mins'])} & {fmt(row['median_runtime_mins'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
