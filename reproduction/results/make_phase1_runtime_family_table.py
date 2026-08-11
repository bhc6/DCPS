from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "results" / "clean_paper_data.csv"
OUT = ROOT / "results" / "tables" / "tab_app_phase1_runtime.tex"

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

ORDER = [
    "GLUE/SuperGLUE",
    "BBII-MC",
    "BBII-Gen",
    "II",
    "MMLU",
]


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
    source = pd.read_csv(CLEAN, low_memory=False)
    source = source[
        source["agent_model"].eq("google/gemma-1.1-7b-it")
        & source["runtime_mins"].notna()
        & (
            source["run_name"].astype(str).str.contains("gemma-1.1-7b-it_google/gemma-1.1-7b-it", regex=False)
            | source["run_name"].astype(str).str.contains("gemma-1.1-7b-it_TO_gemma-1.1-7b-it", regex=False)
        )
    ].copy()

    is_gfb = (
        source["project"].astype(str).str.contains("GFB", regex=False)
        | source["run_name"].astype(str).str.contains("GFB", regex=False)
    )
    is_ppo = (
        source["method"].astype(str).str.contains("PPO", regex=False)
        | source["ppo_used"].astype(str).eq("True")
    )
    is_algprompt_without_ppo = source["method"].eq("AlgPrompt") & ~is_ppo

    rae = source[is_gfb | is_algprompt_without_ppo].copy()
    rae["paper_method"] = "StablePrompt-RAE"
    rae["gpu"] = "A40"

    df = rae
    df["runtime_mins"] = pd.to_numeric(df["runtime_mins"], errors="coerce")
    df["epochs"] = pd.to_numeric(df["epochs"], errors="coerce")
    df = df.dropna(subset=["runtime_mins"])
    df["task_family"] = df["dataset"].map(family)

    per_subset = (
        df.groupby(["paper_method", "task_family", "gpu", "epochs", "dataset"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            subset_runtime_mins=("runtime_mins", "mean"),
        )
        .reset_index()
    )
    grouped = (
        per_subset.groupby(["paper_method", "task_family", "gpu", "epochs"], dropna=False)
        .agg(
            n_runs=("n_runs", "sum"),
            n_datasets=("dataset", "nunique"),
            sweep_runtime_mins=("subset_runtime_mins", "sum"),
            mean_subset_runtime_mins=("subset_runtime_mins", "mean"),
        )
        .reset_index()
    )
    grouped["order"] = grouped["task_family"].map({name: i for i, name in enumerate(ORDER)})
    grouped["method_order"] = grouped["paper_method"].map({"StablePrompt-RAE": 0})
    grouped = grouped.sort_values(["method_order", "order", "epochs", "gpu"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Phase-1 GPU runtime by Main Results task-family granularity. "
        r"StablePrompt-RAE rows preserve the original rule that GFB is always RAE and "
        r"AlgPrompt without PPO is RAE; supplemental StablePrompt-PPO A100 rows are "
        r"reported separately in Table~\ref{tab:app_phase1_ppo_runtime}. "
        r"Parallel repeated runs are not "
        r"summed: runtime is first averaged within each subset, then summed across subsets to "
        r"estimate one family sweep; for MMLU the subsets are subjects. "
        r"Because the two methods use different GPU classes, these rows are hardware-qualified "
        r"runtime audits rather than a same-hardware speed comparison. Phase~1 has no API token "
        r"or USD cost.}"
    )
    lines.append(r"\label{tab:app_phase1_runtime}")
    lines.append(r"\begin{tabular}{@{}llccrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{Task Family} & \textbf{GPU} & \textbf{Epochs} & "
        r"\textbf{Family sweep (h)} & \textbf{Mean min/subset} \\ \midrule"
    )
    for _, row in grouped.iterrows():
        epochs = "--" if pd.isna(row["epochs"]) else str(int(row["epochs"]))
        lines.append(
            f"{row['paper_method']} & {row['task_family']} & {row['gpu']} & {epochs} & "
            f"{fmt(row['sweep_runtime_mins'] / 60.0)} & {fmt(row['mean_subset_runtime_mins'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
