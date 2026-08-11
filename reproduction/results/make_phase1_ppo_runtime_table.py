from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "results" / "clean_paper_data.csv"
OUT = ROOT / "results" / "tables" / "tab_app_phase1_ppo_runtime.tex"

GLUE = {"MNLI", "MRPC", "QNLI", "RTE", "SNLI", "SST2"}
BBII_MC = {
    "CAUSAL_JUDGMENT", "DISAMBIGUATION_QA", "EPISTEMIC_REASONING", "HYPERBATON",
    "IMPLICATURES", "LOGICAL_FALLACY_DETECTION", "MOVIE_RECOMMENDATION", "NAVIGATE",
    "PRESUPPOSITIONS_AS_NLI", "RUIN_NAMES", "SNARKS", "SPORTS_UNDERSTANDING",
    "WINOWHY",
}
BBII_GEN = {
    "DYCK_LANGUAGES", "GENDER_INCLUSIVE_SENTENCES_GERMAN", "OBJECT_COUNTING",
    "OPERATORS", "TENSE", "WORD_SORTING",
}
II = {
    "ACTIVE_TO_PASSIVE", "ANTONYMS", "CAUSE_AND_EFFECT", "COMMON_CONCEPT", "DIFF",
    "FIRST_WORD_LETTER", "INFORMAL_TO_FORMAL", "LARGER_ANIMAL", "LETTERS_LIST",
    "NEGATION", "NUM_TO_VERBAL", "ORTHOGRAPHY_STARTS_WITH", "RHYMES",
    "SECOND_WORD_LETTER", "SENTENCE_SIMILARITY", "SENTIMENT", "SINGULAR_TO_PLURAL",
    "SUM", "SYNONYMS", "TAXONOMY_ANIMAL", "TRANSLATION_EN-DE",
    "TRANSLATION_EN-ES", "TRANSLATION_EN-FR", "WORD_IN_CONTEXT",
}
ORDER = ["GLUE/SuperGLUE", "BBII-MC", "BBII-Gen", "II", "MMLU"]


def family(dataset):
    d = str(dataset)
    if d in GLUE:
        return "GLUE/SuperGLUE"
    if d in BBII_MC:
        return "BBII-MC"
    if d in BBII_GEN:
        return "BBII-Gen"
    if d in II:
        return "II"
    return "MMLU"


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.2f}"


def main():
    df = pd.read_csv(CLEAN, low_memory=False)
    df = df[
        df["agent_model"].eq("google/gemma-1.1-7b-it")
        & df["runtime_mins"].notna()
    ].copy()
    is_gfb = (
        df["project"].astype(str).str.contains("GFB", regex=False)
        | df["run_name"].astype(str).str.contains("GFB", regex=False)
    )
    is_ppo = (
        df["method"].astype(str).str.contains("PPO", regex=False)
        | df["ppo_used"].astype(str).eq("True")
        | df["method"].eq("AlgPrompt")
    )
    df = df[is_ppo & ~is_gfb].copy()
    df["runtime_mins"] = pd.to_numeric(df["runtime_mins"], errors="coerce")
    df["epochs"] = pd.to_numeric(df["epochs"], errors="coerce")
    df = df.dropna(subset=["runtime_mins"])
    df["task_family"] = df["dataset"].map(family)

    per_subset = (
        df.groupby(["task_family", "epochs", "dataset"], dropna=False)
        .agg(subset_runtime_mins=("runtime_mins", "mean"))
        .reset_index()
    )
    grouped = (
        per_subset.groupby(["task_family", "epochs"], dropna=False)
        .agg(
            sweep_runtime_mins=("subset_runtime_mins", "sum"),
            mean_subset_runtime_mins=("subset_runtime_mins", "mean"),
        )
        .reset_index()
    )
    grouped["order"] = grouped["task_family"].map({name: i for i, name in enumerate(ORDER)})
    grouped = grouped.sort_values(["order", "epochs"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Supplemental StablePrompt-PPO A100 runtime by Main Results task-family "
        r"granularity. Rows include non-GFB Gemma1.1-7B A100 rows mapped to "
        r"StablePrompt-PPO, including raw AlgPrompt-labelled rows; GFB is never "
        r"treated as PPO in the paper mapping. Parallel repeated runs are first "
        r"averaged within each subset, then summed across subsets to estimate one family sweep.}"
    )
    lines.append(r"\label{tab:app_phase1_ppo_runtime}")
    lines.append(r"\begin{tabular}{@{}llccrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{Task Family} & \textbf{GPU} & \textbf{Epochs} & "
        r"\textbf{Family sweep (h)} & \textbf{Mean min/subset} \\ \midrule"
    )
    for _, row in grouped.iterrows():
        epochs = "--" if pd.isna(row["epochs"]) else str(int(row["epochs"]))
        lines.append(
            f"StablePrompt-PPO & {row['task_family']} & A100 & {epochs} & "
            f"{fmt(row['sweep_runtime_mins'] / 60.0)} & {fmt(row['mean_subset_runtime_mins'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
