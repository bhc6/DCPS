import os
import sys
import dspy
import pickle
from pathlib import Path
from unittest.mock import MagicMock

# Environment setup
try:
    import pkg_resources
except ImportError:
    sys.modules['pkg_resources'] = MagicMock()
sys.modules['syllapy'] = MagicMock()

sys.path.append(os.getcwd())

import gepa_artifact
sys.modules['langProBe'] = gepa_artifact
import gepa_artifact.benchmarks as benchmarks
sys.modules['langProBe.benchmarks'] = benchmarks
import gepa_artifact.benchmarks.dspy_program as dspy_program
sys.modules['langProBe.benchmarks.dspy_program'] = dspy_program
sys.modules['langProBe.dspy_program'] = dspy_program

from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

def sample_check():
    # Setup LM
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
    model_name = "openai/gpt-4o-mini" if api_base else "openai/gpt-4.1-mini-2025-04-14"
    
    lm = dspy.LM(model_name, api_key=api_key, api_base=api_base, max_tokens=2048)
    dspy.configure(lm=lm)
    
    # Load Program
    pkl_path = "final_pkls/AIMEBench_CoT_GEPA_gpt-41-mini.pkl"
    with open(pkl_path, 'rb') as f:
        program = pickle.load(f)
    program.set_lm(lm)

    # Check Program State
    print(f"\n--- [Checking Program State] ---")
    for name, predictor in program.named_predictors():
        print(f"Predictor: {name}")
        # Print first 200 chars of instructions
        instr = getattr(predictor.signature, 'instructions', 'No instructions found')
        print(f"Instructions (preview): {instr[:300]}...")
        
        demos = getattr(predictor, 'demos', [])
        print(f"Number of Demos: {len(demos)}")

    # Get Data
    benchmark = aime_metas[0].benchmark()
    test_set = benchmark.test_set
    sample_example = test_set[0]
    
    print(f"\n--- [Sample Check] ---")
    print(f"Problem: {sample_example.problem}")
    print(f"Expected Answer: {sample_example.answer}")
    print(f"\nRunning Model...")
    
    prediction = program(problem=sample_example.problem)
    
    print(f"\n--- [Model Output] ---")
    # Print all fields in prediction to be sure
    for key, value in prediction.items():
        print(f"{key.capitalize()}: {value}")
    
    # Inspect history to see the full prompt
    print(f"\n--- [Full Prompt History (Last Call)] ---")
    lm.inspect_history(n=1)

if __name__ == "__main__":
    sample_check()
