import math
from pathlib import Path

import pandas as pd

W = 4
ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "results" / "tables"


def wt(t_in, t_out):
    return t_in + W * t_out


B_SCORE = {
    ("Qwen3", "HotpotQA"): 40.33,
    ("Qwen3", "IFBench"): 38.61,
    ("Qwen3", "Hover"): 33.67,
    ("Qwen3", "PUPA"): 81.55,
    ("Qwen3", "AIME-2025"): 47.33,
    ("Qwen3", "LiveBench-Math"): 65.01,
    ("GPT", "HotpotQA"): 36.10,
    ("GPT", "IFBench"): 48.13,
    ("GPT", "Hover"): 40.33,
    ("GPT", "PUPA"): 80.81,
    ("GPT", "AIME-2025"): 40.00,
    ("GPT", "LiveBench-Math"): 55.80,
}

M_SCORE = {
    ("Qwen3", "AIME-2025"): {"MIPRO": 58.00, "GEPAM": 62.67, "GEPA": 59.33, "FSPO": 55.33},
    ("Qwen3", "LiveBench-Math"): {"MIPRO": 67.46, "GEPAM": 66.28, "GEPA": 70.57, "FSPO": 65.08},
    ("GPT", "HotpotQA"): {"MIPRO": 55.00, "GEPAM": 63.33, "GEPA": 65.00, "FSPO": 59.00},
    ("GPT", "IFBench"): {"MIPRO": 51.19, "GEPAM": 49.49, "GEPA": 49.83, "FSPO": 51.53},
    ("GPT", "Hover"): {"MIPRO": 47.33, "GEPAM": 49.00, "GEPA": 49.67, "FSPO": 45.67},
    ("GPT", "PUPA"): {"MIPRO": 83.59, "GEPAM": 93.52, "GEPA": 90.10, "FSPO": 94.09},
    ("GPT", "AIME-2025"): {"MIPRO": 46.67, "GEPAM": 46.67, "GEPA": 48.00, "FSPO": 48.00},
    ("GPT", "LiveBench-Math"): {"MIPRO": 57.30, "GEPAM": 61.14, "GEPA": 64.21, "FSPO": 59.52},
}

TOK = {
    ("GPT", "HotpotQA"): {
        "B_test": (0.771, 0.162),
        "MIPRO_opt": (39.873, 3.254), "MIPRO_test": (2.006, 0.162),
        "GEPAM_opt": (13.801, 1.675), "GEPAM_test": (1.477, 0.234),
        "GEPA_opt": (21.434, 3.306), "GEPA_test": (1.477, 0.228),
        "FSPO_opt": (2.940, 0.650),
    },
    ("GPT", "IFBench"): {
        "B_test": (0.217, 0.245),
        "MIPRO_opt": (8.880, 2.503), "MIPRO_test": (0.448, 0.303),
        "GEPAM_opt": (4.095, 1.894), "GEPAM_test": (0.570, 0.275),
        "GEPA_opt": (4.988, 2.191), "GEPA_test": (0.550, 0.265),
        "FSPO_opt": (1.045, 0.725),
    },
    ("GPT", "Hover"): {
        "B_test": (0.825, 0.236),
        "MIPRO_opt": (19.746, 2.714), "MIPRO_test": (2.234, 0.262),
        "GEPAM_opt": (19.665, 4.141), "GEPAM_test": (1.408, 0.313),
        "GEPA_opt": (21.846, 4.863), "GEPA_test": (1.599, 0.321),
        "FSPO_opt": (3.089, 1.245),
    },
    ("GPT", "PUPA"): {
        "B_test": (0.294, 0.132),
        "MIPRO_opt": (7.717, 1.384), "MIPRO_test": (0.418, 0.115),
        "GEPAM_opt": (5.659, 1.812), "GEPAM_test": (0.524, 0.126),
        "GEPA_opt": (6.195, 1.987), "GEPA_test": (0.552, 0.132),
        "FSPO_opt": (6.326, 2.019), "FSPO_test": (0.498, 0.124),
    },
    ("GPT", "AIME-2025"): {
        "B_test": (0.006, 0.086),
        "MIPRO_opt": (6.056, 3.885), "MIPRO_test": (0.296, 0.182),
        "GEPAM_opt": (4.377, 9.075), "GEPAM_test": (0.044, 0.145),
        "GEPA_opt": (4.850, 9.469), "GEPA_test": (0.051, 0.123),
        "FSPO_opt": (2.209, 3.649),
    },
    ("GPT", "LiveBench-Math"): {
        "B_test": (0.060, 0.190),
        "MIPRO_opt": (7.196, 1.878), "MIPRO_test": (0.427, 0.193),
        "GEPAM_opt": (2.597, 2.601), "GEPAM_test": (0.111, 0.226),
        "GEPA_opt": (2.650, 2.810), "GEPA_test": (0.212, 0.247),
        "FSPO_opt": (0.540, 0.860), "FSPO_test": (0.520, 1.030),
    },
    ("Qwen3", "AIME-2025"): {
        "B_test": (0.026, 1.375),
        "MIPRO_opt": (2.223, 16.735), "MIPRO_test": (0.108, 1.181),
        "FSPO_opt": (1.230, 2.450),
    },
    ("Qwen3", "LiveBench-Math"): {
        "B_test": (0.058, 0.453),
        "MIPRO_opt": (4.788, 6.840), "MIPRO_test": (0.378, 0.670),
        "GEPAM_opt": (2.159, 9.631), "GEPAM_test": (0.129, 0.796),
        "GEPA_opt": (2.750, 9.368), "GEPA_test": (0.136, 0.733),
        "FSPO_opt": (1.111, 10.110),
    },
}

CELLS = [
    ("GPT", "HotpotQA"),
    ("GPT", "IFBench"),
    ("GPT", "Hover"),
    ("GPT", "PUPA"),
    ("GPT", "AIME-2025"),
    ("GPT", "LiveBench-Math"),
]
METHODS = ["MIPRO", "GEPAM", "GEPA", "FSPO"]
DISPLAY = {"MIPRO": "MIPROv2-Heavy", "GEPAM": "GEPA-MERGE", "GEPA": "GEPA", "FSPO": "FSPO"}


def fspo_test(cell):
    if "FSPO_test" in TOK.get(cell, {}):
        return TOK[cell]["FSPO_test"]
    tests = [TOK[cell][k] for k in ["MIPRO_test", "GEPAM_test", "GEPA_test"] if k in TOK.get(cell, {})]
    if not tests:
        return None
    return (sum(t[0] for t in tests) / len(tests), sum(t[1] for t in tests) / len(tests))


def baseline_test(cell):
    bt = TOK.get(cell, {}).get("B_test")
    return fspo_test(cell) if bt is None else bt


def method_tokens(cell, method):
    if method == "FSPO":
        return TOK.get(cell, {}).get("FSPO_opt"), fspo_test(cell)
    return TOK.get(cell, {}).get(method + "_opt"), TOK.get(cell, {}).get(method + "_test")


def compute(cell, method):
    if cell not in TOK or cell not in M_SCORE or method not in M_SCORE[cell]:
        return None
    opt, tst = method_tokens(cell, method)
    bt = baseline_test(cell)
    if opt is None or tst is None or bt is None:
        return None
    tm_w = wt(*opt) + wt(*tst)
    tb_w = wt(*bt)
    log2 = math.log2(tm_w / tb_w)
    rer = (M_SCORE[cell][method] - B_SCORE[cell]) / (100 - B_SCORE[cell]) * 100
    return {"tm_w": tm_w, "tb_w": tb_w, "log2": log2, "rer": rer, "rcei": rer / log2}


def fmt_signed(x, digits=2):
    return f"{x:+.{digits}f}"


def generate():
    rows = []
    for cell in CELLS:
        for method in METHODS:
            r = compute(cell, method)
            if r:
                rows.append({"model": cell[0], "task": cell[1], "method": method, **r})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "rcei_weighted_audited_recomputed.csv", index=False)

    common = [c for c in CELLS if all(compute(c, m) is not None for m in METHODS)]
    macro = []
    for method in METHODS:
        vals = [compute(c, method) for c in common]
        macro.append({
            "method": method,
            "score": sum(M_SCORE[c][method] for c in common) / len(common),
            "rer": sum(v["rer"] for v in vals) / len(vals),
            "tm_w": sum(v["tm_w"] for v in vals) / len(vals),
            "log2": sum(v["log2"] for v in vals) / len(vals),
            "rcei": sum(v["rcei"] for v in vals) / len(vals),
        })
    macro_df = pd.DataFrame(macro).sort_values("rcei")
    macro_df.to_csv(ROOT / "results" / "rcei_weighted_audited_macro.csv", index=False)

    baseline_score = sum(B_SCORE[c] for c in common) / len(common)
    baseline_tb = sum(wt(*baseline_test(c)) for c in common) / len(common)
    caption = (
        r"Compute-Anchored RCEI of prompt-optimization methods relative to the "
        r"unoptimized baseline ($\mathcal{B}$). "
        r"Cost axis: $T=T_{\text{in}}+4T_{\text{out}}$ (optimizer + test tokens). "
        r"Macro values are means across the GPT-4.1-mini 6-task subset: "
        r"\{HotpotQA, IFBench, Hover, PUPA, AIME-2025, LiveBench-Math\}."
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:rcei}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\textbf{Method ($\mathcal{M}$)} & \textbf{Macro Score} & \textbf{Macro RER} & \textbf{$T_\mathcal{M}$ (M-eq)} & \textbf{$\overline{\log_2(T_\mathcal{M}/T_\mathcal{B})}$} & \textbf{Macro RCEI} \\ \midrule",
        rf"Baseline ($\mathcal{{B}}$, unoptimized) & ${baseline_score:.2f}$ & $0$ & ${baseline_tb:.2f}$ & $0$ & -- \\",
    ]

    # Find column-wise bests among optimized methods
    best_score = macro_df["score"].max()
    best_rer = macro_df["rer"].max()
    best_tm = macro_df["tm_w"].min()
    best_log2 = macro_df["log2"].min()
    best_rcei = macro_df["rcei"].max()

    for _, r in macro_df.iterrows():
        # format values
        m_name = DISPLAY[r["method"]]
        s_val = f"{r['score']:.2f}"
        rer_val = f"{fmt_signed(r['rer'])}\\%"
        tm_val = f"{r['tm_w']:.2f}"
        log2_val = f"{r['log2']:.2f}"
        rcei_val = fmt_signed(r["rcei"])

        # bold if best among optimized
        s_str = rf"\mathbf{{{s_val}}}" if r["score"] == best_score else s_val
        rer_str = rf"\mathbf{{{rer_val}}}" if r["rer"] == best_rer else rer_val
        tm_str = rf"\mathbf{{{tm_val}}}" if r["tm_w"] == best_tm else tm_val
        log2_str = rf"\mathbf{{{log2_val}}}" if r["log2"] == best_log2 else log2_val
        rcei_str = rf"\mathbf{{{rcei_val}}}" if r["rcei"] == best_rcei else rcei_val

        lines.append(f"{m_name} & ${s_str}$ & ${rer_str}$ & ${tm_str}$ & ${log2_str}$ & ${rcei_str}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{@{}l}{\footnotesize Per-task breakdown in Table~\ref{tab:app_rcei_pertask}.}\\",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    (TABLE_DIR / "tab7_rcei.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    task_order = [
        ("GPT", "HotpotQA"), ("GPT", "IFBench"), ("GPT", "Hover"), ("GPT", "PUPA"),
        ("GPT", "AIME-2025"), ("GPT", "LiveBench-Math"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-task RCEI breakdown. Cells marked ``--'' lack the required token data for that method/task.}",
        r"\label{tab:app_rcei_pertask}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{@{}llccrrrr@{}}",
        r"\toprule",
        r"& & & & \multicolumn{4}{c}{\textbf{Per-method RER\% / RCEI}} \\",
        r"\cmidrule(lr){5-8}",
        r"\textbf{Model} & \textbf{Task} & $\mathcal{B}$ & $T_\mathcal{B}$ (M-eq) & \textbf{MIPROv2-Heavy} & \textbf{GEPA-MERGE} & \textbf{GEPA} & \textbf{FSPO} \\ \midrule",
    ]
    for i, cell in enumerate(task_order):
        if i == 0:
            prefix = r"\multirow{6}{*}{GPT-4.1-mini}"
        else:
            prefix = ""
        bt = baseline_test(cell)
        tb = f"{wt(*bt):.3f}" if bt else "--"
        row_results = {method: compute(cell, method) for method in METHODS}
        max_rer = max([r["rer"] for r in row_results.values() if r is not None], default=None)
        max_rcei = max([r["rcei"] for r in row_results.values() if r is not None], default=None)
        vals = []
        for method in METHODS:
            r = row_results[method]
            if r is None:
                vals.append("-- / --")
            else:
                rer_val = fmt_signed(r['rer'])
                rcei_val = fmt_signed(r['rcei'])
                rer_str = rf"\mathbf{{{rer_val}}}" if r["rer"] == max_rer else rer_val
                rcei_str = rf"\mathbf{{{rcei_val}}}" if r["rcei"] == max_rcei else rcei_val
                vals.append(rf"${rer_str}$ / ${rcei_str}$")
        lines.append(rf"{prefix} & {cell[1]} & {B_SCORE[cell]:.2f} & {tb} & " + " & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{@{}p{1.05\textwidth}}{\footnotesize Macro RCEI ranking on the common subset: "
        + " $>$ ".join(f"{DISPLAY[r['method']]} ${fmt_signed(r['rcei'])}$" for _, r in macro_df.sort_values('rcei', ascending=False).iterrows())
        + r".}\\",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    (TABLE_DIR / "tab_app_rcei_pertask.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("common", common)
    print(macro_df.sort_values("rcei", ascending=False).to_string(index=False))


if __name__ == "__main__":
    generate()
