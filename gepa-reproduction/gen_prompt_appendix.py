"""Generate a verbatim full-text appendix of every optimized prompt.

Sources:
- extracted_all_prompts.json  (GEPA/GEPA-MERGE/Abl-SelectBestCandidate/GRPO/MIPRO)
- ../gepa/case_study/dcps_prompts.json + prompts_readable_*.txt (DCPS-Compound)

For MIPROv2-Heavy the pkl stores the whole candidate-instruction pool; we print
only the compiled program's module instructions (the first block, before the
default-template signatures), and say so.
"""
import json
import re

d = json.load(open("extracted_all_prompts.json", encoding="utf-8"))

BENCH_ORDER = ["AIMEBench", "LiveBenchMathBench", "hoverBench",
               "HotpotQABench", "IFBench", "Papillon"]
BENCH_TITLE = {
    "AIMEBench": "AIME-2025", "LiveBenchMathBench": "LiveBench-Math",
    "hoverBench": "HoVer (4-hop)", "HotpotQABench": "HotpotQA (multi-hop)",
    "IFBench": "IFBench", "Papillon": "PUPA / Papillon",
}
# number of pipeline modules per benchmark (compiled program size)
NMOD = {"AIMEBench": 1, "LiveBenchMathBench": 1, "hoverBench": 4,
        "HotpotQABench": 4, "IFBench": 2, "Papillon": 1}
METHODS = ["GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate", "MIPROv2-Heavy", "GRPO"]
MODELS = ["gpt-41-mini", "qwen3-8b"]


def find(bench, method, model):
    for fn in d:
        if fn.startswith(bench) and f"_{method}_" in fn and model in fn:
            return fn
    return None


def stages(fn, bench, method):
    sigs = d[fn]["signatures"]
    if method == "MIPROv2-Heavy":
        # compiled modules = first NMOD signatures (before default-template block)
        return sigs[: NMOD[bench]]
    return [s for s in sigs if s["len"] > 90] or sigs


out = ["# Appendix: Full Optimized Prompts (verbatim)\n",
       "Every optimized instruction below is reproduced in full from the run "
       "artifacts. GEPA / GEPA-MERGE / Abl-SelectBestCandidate / GRPO show all "
       "compiled module instructions. MIPROv2-Heavy shows only the compiled "
       "program's module instructions (its full candidate pool in the pkl is "
       "omitted). DCPS-Compound prompts (HoVer, LiveBench) are in the second "
       "half.\n"]

for bench in BENCH_ORDER:
    out.append(f"\n---\n\n## {BENCH_TITLE[bench]}\n")
    for model in MODELS:
        for method in METHODS:
            fn = find(bench, method, model)
            if not fn:
                continue
            st = stages(fn, bench, method)
            st = [s for s in st if s["len"] > 40]
            if not st:
                continue
            out.append(f"\n### {method} — {model}\n")
            out.append(f"*(file: `{fn}`)*\n")
            for i, s in enumerate(st):
                label = f"stage {i+1}/{len(st)}"
                flds = ", ".join(s["fields"]) if s["fields"] else "?"
                out.append(f"\n**{label}** — outputs `{flds}` — {s['len']} chars\n")
                out.append("```text")
                out.append(s["instruction"].rstrip())
                out.append("```")

open("../gepa/case_study/APPENDIX_PROMPTS_FULL.md", "w", encoding="utf-8").write(
    "\n".join(out))
print("wrote APPENDIX_PROMPTS_FULL.md; sections:", len(out))
