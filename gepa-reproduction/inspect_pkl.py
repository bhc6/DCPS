import pickle
import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Mock missing arbor client
class MockArbor:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return []
    class ArborProvider:
        def __init__(self, *args, **kwargs): pass
        def __getstate__(self): return {}
        def __setstate__(self, state): pass

sys.modules["dspy.clients.lm_local_arbor"] = MockArbor

# Aliasing
import gepa_artifact
import gepa_artifact.benchmarks as benchmarks
import gepa_artifact.benchmarks.dspy_program as dspy_program
import gepa_artifact.benchmarks.papillon as papillon

bench_map = {"papillon": papillon}
for i in range(1, 6):
    prefix = ".".join(["langProBe"] * i)
    sys.modules[prefix] = gepa_artifact
    sys.modules["benchmarks." + prefix] = gepa_artifact
    for name, mod in bench_map.items():
        sys.modules[f"{prefix}.{name}"] = mod

pkl_path = "final_pkls/Papillon_PAPILLON_GEPA_qwen3-8b.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        prog = pickle.load(f)
    
    print(f"Program type: {type(prog)}")
    print(f"Attributes: {dir(prog)}")
    if hasattr(prog, 'untrusted_model'):
        print(f"untrusted_model: {prog.untrusted_model}")
        if hasattr(prog.untrusted_model, 'model'):
            print(f"untrusted_model model: {prog.untrusted_model.model}")
        if hasattr(prog.untrusted_model, 'api_base'):
            print(f"untrusted_model api_base: {prog.untrusted_model.api_base}")
else:
    print(f"File not found: {pkl_path}")
