"""Compute char + real-token lengths for every optimizer × benchmark × backbone.

GPT cells use tiktoken cl100k_base; Qwen cells use the Qwen3-8B tokenizer.
DCPS from dcps_paper_prompts.json (paper-Table-2b runs) + PUPA/HotpotQA verbatim
from CASE_STUDY_PROMPTS.md; GEPA/MERGE/Abl/MIPRO from ../../gepa-rp extraction.
"""
import json
import re
import os

import tiktoken
from transformers import AutoTokenizer

ENC_GPT = tiktoken.get_encoding("cl100k_base")
TOK_QWEN = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def toks(s, model):
    return len(ENC_GPT.encode(s)) if model == "gpt" else len(TOK_QWEN.encode(s))


HERE = os.path.dirname(os.path.abspath(__file__))
gepa = json.load(open(os.path.join(HERE, "..", "..", "gepa-rp",
                                    "extracted_all_prompts.json"), encoding="utf-8"))
dcps = json.load(open(os.path.join(HERE, "dcps_paper_prompts.json"), encoding="utf-8"))


def gepa_text(fn):
    if fn not in gepa or "signatures" not in gepa[fn]:
        return None
    sigs = [s for s in gepa[fn]["signatures"] if s["len"] > 90]
    return "\n".join(s["instruction"] for s in sigs) if sigs else None


# PUPA (2-pred) and HotpotQA (4-stage) DCPS text is documented verbatim in
# CASE_STUDY_PROMPTS.md; reconstruct concatenated text lengths from annotations.
# We store their concatenated prompt text if available; else fall back to char totals.
DCPS_EXTRA_CHARS = {"pupa__gpt-4.1-mini": 1458, "hotpotqa__gpt-4.1-mini": 3006}

BENCH = [
    ("AIME-2025", "AIMEBench_CoT", "aime"),
    ("LiveBench-Math", "LiveBenchMathBench_CoT", "livebench"),
    ("HotpotQA", "HotpotQABench_HotpotMultiHop", "hotpotqa"),
    ("IFBench", "IFBench_IFBenchCoT2StageProgram", "ifbench"),
    ("PUPA", "Papillon_PAPILLON", "pupa"),
    ("HoVer", "hoverBench_HoverMultiHop", "hover"),
]
METHODS = ["GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate", "MIPROv2-Heavy"]


def row(model_tag, model_suffix):
    print(f"\n### {model_tag} ###")
    hdr = f"{'benchmark':<16}{'DCPS c/t':>14}" + "".join(f"{m[:10]+' c/t':>16}" for m in METHODS)
    print(hdr)
    for disp, prefix, dk in BENCH:
        cells = []
        # DCPS
        key = f"{dk}__{model_tag}"
        if key in dcps:
            p = dcps[key]["prompt"]
            cells.append(f"{len(p)}/{toks(p, model_suffix)}")
        elif f"{dk}__{model_suffix if False else model_tag}" in DCPS_EXTRA_CHARS:
            cells.append(f"{DCPS_EXTRA_CHARS[key]}/~")
        else:
            cells.append("—")
        for m in METHODS:
            fn = f"{prefix}_{m}_{'gpt-41-mini' if model_suffix=='gpt' else 'qwen3-8b'}.pkl"
            t = gepa_text(fn)
            cells.append(f"{len(t)}/{toks(t, model_suffix)}" if t else "—")
        print(f"{disp:<16}{cells[0]:>14}" + "".join(f"{c:>16}" for c in cells[1:]))


row("gpt-4.1-mini", "gpt")
row("qwen3-8b", "qwen")
