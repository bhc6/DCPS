import sys
import os
from unittest.mock import MagicMock

# Monkeypatch
try:
    import pkg_resources
except ImportError:
    sys.modules['pkg_resources'] = MagicMock()

sys.path.append(os.getcwd())

from gepa_artifact.benchmarks.hover import benchmark as hover_metas
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotQA_metas
from gepa_artifact.benchmarks.papillon import benchmark as papillon_metas
from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas
from gepa_artifact.benchmarks.livebench_math import benchmark as math_metas
from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

all_metas = [
    ("Hover", hover_metas),
    ("HotpotQA", hotpotQA_metas),
    ("Papillon", papillon_metas),
    ("IFBench", ifbench_metas),
    ("LiveBenchMath", math_metas),
    ("AIME", aime_metas)
]

for name, metas in all_metas:
    try:
        bm = metas[0].benchmark()
        print(f"{name}: {len(bm.test_set)} examples")
    except Exception as e:
        print(f"{name}: Failed to load ({e})")
