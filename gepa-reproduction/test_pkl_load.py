import sys
import os
import pickle
from unittest.mock import MagicMock

# Monkeypatch pkg_resources for syllapy
try:
    import pkg_resources
except ImportError:
    sys.modules['pkg_resources'] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

import gepa_artifact
# Alias langProBe to gepa_artifact
sys.modules['langProBe'] = gepa_artifact
import gepa_artifact.benchmarks as benchmarks
sys.modules['langProBe.benchmarks'] = benchmarks
import gepa_artifact.benchmarks.dspy_program as dspy_program
sys.modules['langProBe.benchmarks.dspy_program'] = dspy_program
sys.modules['langProBe.dspy_program'] = dspy_program

def test_load(pkl_path):
    print(f"Testing load of {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            p = pickle.load(f)
        print(f"Successfully loaded {type(p)}")
        return True
    except Exception as e:
        print(f"Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_load('final_pkls/AIMEBench_CoT_GEPA_gpt-41-mini.pkl')
