"""按任务分类导出有效实验结果。
- 不修改任何已生成产物。
- 过滤失败实验：final_score 为 NaN 或 <= 0。
- 为每条 run 增加 task_family 标签（GLUE / BBH-II / MMLU / Math / QA / GEPA-Bench / Other）。
- 输出两个 CSV：
    1) by_task_runs.csv     —— 每条 run 一行
    2) by_task_summary.csv  —— 按 (task_family, dataset, agent_model, method) 聚合
"""
import os
import pandas as pd

RESULTS_DIR = os.environ.get(
    "DCPS_RESULTS_DIR", os.path.dirname(os.path.abspath(__file__))
)
SRC = os.path.join(RESULTS_DIR, "clean_paper_data.csv")
OUT_RUNS = os.path.join(RESULTS_DIR, "by_task_runs.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "by_task_summary.csv")

GLUE = {"MNLI", "MRPC", "QNLI", "RTE", "SNLI", "SST2", "HOVER"}
BBH_II = {
    "ANTONYMS", "WORD_IN_CONTEXT", "RHYMES", "NUM_TO_VERBAL",
    "ACTIVE_TO_PASSIVE", "CAUSE_AND_EFFECT", "FIRST_WORD_LETTER",
    "LARGER_ANIMAL", "LETTERS_LIST", "ORTHOGRAPHY_STARTS_WITH",
    "SECOND_WORD_LETTER", "SENTIMENT", "SUM", "SYNONYMS",
    "TAXONOMY_ANIMAL", "TRANSLATION_EN-DE", "TRANSLATION_EN-ES",
    "TRANSLATION_EN-FR", "SINGULAR_TO_PLURAL", "SENTENCE_SIMILARITY",
    "NEGATION", "INFORMAL_TO_FORMAL", "DIFF", "COMMON_CONCEPT",
    "DYCK_LANGUAGES", "GENDER_INCLUSIVE_SENTENCES_GERMAN",
    "OBJECT_COUNTING", "WORD_SORTING", "OPERATORS", "TENSE",
    "PRESUPPOSITIONS_AS_NLI", "LINGUISTICS_PUZZLES", "HYPERBATON",
    "DISAMBIGUATION_QA", "EPISTEMIC_REASONING", "MOVIE_RECOMMENDATION",
    "SNARKS", "NAVIGATE", "SPORTS_UNDERSTANDING", "RUIN_NAMES",
    "IMPLICATURES", "WINOWHY", "LOGICAL_FALLACY_DETECTION",
    "BBH-MC", "BBH-GEN",
}
MATH = {"AIME-2025", "LIVEBENCH-MATH"}
QA = {"HOTPOTQA", "PUPA", "IFBENCH"}

MMLU_KEYWORDS = [
    "HISTORY", "LAW", "MMLU", "BIOLOGY", "CHEMISTRY", "PHYSICS",
    "MATHEMATICS", "COMPUTER", "PSYCHOLOGY", "ECONOMICS", "MEDICINE",
    "PHILOSOPHY", "GEOGRAPHY", "STATISTICS", "ALGEBRA", "ANATOMY",
    "ASTRONOMY", "ETHICS", "KNOWLEDGE", "JURISPRUDENCE", "MARKETING",
    "MANAGEMENT", "NUTRITION", "POLICY", "RELATIONS", "SECURITY",
    "SOCIOLOGY", "VIROLOGY", "GENETICS", "AGING", "SEXUALITY",
    "PREHISTORY", "ELECTRICAL", "ECONOMETRICS", "FACTS", "LOGIC",
    "CONCEPTUAL", "ELEMENTARY", "CLINICAL", "PROFESSIONAL", "FORMAL",
    "GLOBAL", "INTERNATIONAL", "MISCELLANEOUS", "MORAL",
    "MACHINE_LEARNING", "PUBLIC",
]


def assign_task_family(ds):
    if pd.isna(ds):
        return "Unknown"
    s = str(ds).upper()
    if s in GLUE:
        return "GLUE"
    if s in BBH_II:
        return "BBH-II"
    if s in MATH:
        return "Math"
    if s in QA:
        return "QA"
    if any(k in s for k in MMLU_KEYWORDS):
        return "MMLU"
    return "Other"


def main():
    df = pd.read_csv(SRC, low_memory=False)
    n_total = len(df)

    # 过滤失败实验：final_score 必须为正
    df = df[df["final_score"].notna() & (df["final_score"] > 0)].copy()
    n_valid = len(df)
    print(f"Filtered failed runs: {n_total} -> {n_valid}")

    # 按任务分类
    df["task_family"] = df["dataset"].apply(assign_task_family)

    # ---- 1) per-run 详情 ----
    cols_priority = [
        "task_family", "dataset", "agent_model", "method",
        "ppo_used", "cs", "ca", "metric", "optimizer_name",
        "metaprompt_style", "num_fewshot", "top_k", "top_k_prompts",
        "valset_size", "num_iterations", "total_iterations",
        "epochs", "train_size", "batch_size", "prompt_per_example",
        "global_step",
        "final_score", "rollouts", "rollouts_budget", "test_size",
        "input_tokens", "output_tokens",
        "real_cost_usd", "estimated_cost_usd",
        "runtime_mins", "seed",
        "project", "run_name", "run_id",
    ]
    cols = [c for c in cols_priority if c in df.columns]
    df_runs = df[cols].sort_values(
        by=["task_family", "dataset", "agent_model", "method", "seed"],
        kind="stable",
    )
    df_runs.to_csv(OUT_RUNS, index=False)
    print(f"[saved] {OUT_RUNS}  ({len(df_runs)} rows, {len(cols)} cols)")

    # ---- 2) 按 (task_family, dataset, agent_model, method) 聚合 ----
    agg_funcs = {
        "final_score": ["count", "mean", "std"],
        "rollouts": ["mean"],
        "rollouts_budget": ["mean"],
        "test_size": ["mean"],
        "real_cost_usd": ["mean"],
        "estimated_cost_usd": ["mean"],
        "runtime_mins": ["mean"],
        "input_tokens": ["mean"],
        "output_tokens": ["mean"],
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    grp = (
        df.groupby(["task_family", "dataset", "agent_model", "method"], dropna=False)
        .agg(agg_funcs)
    )
    # flatten columns
    grp.columns = [
        f"{a}_{b}" if b not in ("mean", "") else (f"n_runs" if (a, b) == ("final_score", "count") else f"{a}_{b}")
        for a, b in grp.columns
    ]
    grp = grp.reset_index()
    # rename count -> n_runs
    if "final_score_count" in grp.columns:
        grp = grp.rename(columns={"final_score_count": "n_runs"})
    grp = grp.sort_values(
        by=["task_family", "dataset", "agent_model", "method"], kind="stable"
    )
    grp.to_csv(OUT_SUMMARY, index=False)
    print(f"[saved] {OUT_SUMMARY}  ({len(grp)} rows)")

    # 简明分布报告
    print()
    print("=== valid runs by task_family ===")
    print(df["task_family"].value_counts().to_string())
    print()
    print("=== valid runs by (task_family, method) ===")
    print(df.groupby(["task_family", "method"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
