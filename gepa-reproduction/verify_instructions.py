import os
import sys
import pickle
from unittest.mock import MagicMock

# 1. Mock dependencies
sys.modules['syllapy'] = MagicMock()
sys.path.append(os.getcwd())

# 2. Alias namespaces
import gepa_artifact
sys.modules['langProBe'] = gepa_artifact
import gepa_artifact.benchmarks as benchmarks
sys.modules['langProBe.benchmarks'] = benchmarks
import gepa_artifact.benchmarks.dspy_program as dspy_program
sys.modules['langProBe.benchmarks.dspy_program'] = dspy_program
sys.modules['langProBe.dspy_program'] = dspy_program

def verify():
    pkl_path = "final_pkls/AIMEBench_CoT_GEPA_gpt-41-mini.pkl"
    print(f"Loading: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        program = pickle.load(f)
    
    print("\n" + "="*50)
    print("PROGRAM STATE VERIFICATION")
    print("="*50)
    
    # Check instructions
    instr = program.predict.signature.instructions
    print(f"\n[INSTRUCTIONS FOUND]:\n{instr}")
    
    if len(instr) > 200:
        print("\n✅ SUCCESS: Found long optimized instructions!")
    else:
        print("\n❌ WARNING: Instructions seem too short. Possible loading issue.")

    # Check demos
    demos = program.predict.demos
    print(f"\n[NUMBER OF DEMOS]: {len(demos)}")
    if len(demos) > 0:
        print("✅ SUCCESS: Demos are loaded!")
    else:
        print("❌ WARNING: No demos found.")

if __name__ == "__main__":
    verify()
