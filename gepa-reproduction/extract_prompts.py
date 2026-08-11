"""Extract optimized instructions from DSPy program pkls WITHOUT importing dspy
or executing any pickled code. Uses pickletools.genops to read the opcode
stream; the signature instruction is the string immediately following the
`__doc__` key in each pydantic Signature class namespace.
"""
import io
import os
import re
import sys
import json
import pickletools

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DUNDER_OR_ID = re.compile(r"^(__.*__|[A-Za-z_][A-Za-z0-9_]*)$")


def read_strings(path):
    with open(path, "rb") as f:
        data = f.read()
    seq = []
    for op, arg, pos in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE") and isinstance(arg, str):
            seq.append(arg)
    return seq


def extract(path):
    seq = read_strings(path)
    sigs = []
    for i, s in enumerate(seq):
        if s != "__doc__":
            continue
        instr = seq[i + 1] if i + 1 < len(seq) else ""
        # skip when the next token is another dunder/attr (e.g. a function's __closure__)
        if not instr or DUNDER_OR_ID.match(instr):
            continue
        # recover field names: identifiers appearing before the nearest preceding
        # '__class_vars__' marker, walking back a small window
        fields = []
        j = i - 1
        # find the class_vars / annotations marker just before __doc__
        while j > 0 and seq[j] not in ("__class_vars__", "__annotations__"):
            j -= 1
        k = j - 1
        while k > 0 and seq[k] not in ("str", "__abstractmethods__", "__parameters__"):
            if IDENT.match(seq[k]) and not seq[k].startswith("__"):
                fields.append(seq[k])
            k -= 1
        fields = list(dict.fromkeys(reversed(fields)))
        sigs.append({"fields": fields, "len": len(instr), "instruction": instr})
    # predictor names present in the stream
    names = [s for s in seq if s.endswith(".predict") or s in
             ("create_query_hop2", "create_query_hop3", "summarize1", "summarize2",
              "predict", "craft_redacted_request", "respond_to_query", "generate_query")]
    return {"predictor_names": list(dict.fromkeys(names)), "signatures": sigs}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    pkl_dir = "final_pkls"
    out = {}
    for fn in sorted(os.listdir(pkl_dir)):
        if not fn.endswith(".pkl"):
            continue
        if target and target.lower() not in fn.lower():
            continue
        try:
            out[fn] = extract(os.path.join(pkl_dir, fn))
        except Exception as e:
            out[fn] = {"error": f"{type(e).__name__}: {e}"}
    with open("extracted_all_prompts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for fn, info in out.items():
        if "error" in info:
            print(f"{fn}: ERROR {info['error']}")
        else:
            lens = [s["len"] for s in info["signatures"]]
            print(f"{fn}: {len(info['signatures'])} sigs lens={lens}")
