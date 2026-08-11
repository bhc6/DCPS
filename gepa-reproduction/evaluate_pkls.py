import os
import sys
import json
import dspy
import argparse
import pickle
from pathlib import Path
from unittest.mock import MagicMock

# Monkeypatch pkg_resources for syllapy
try:
    import pkg_resources
except ImportError:
    sys.modules['pkg_resources'] = MagicMock()

# Mock syllapy to avoid "ValueError: Cannot open console output buffer for reading" on Windows
sys.modules['syllapy'] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

# Aliasing langProBe and its benchmarks (handling recursive paths in pkls)
import gepa_artifact
import gepa_artifact.benchmarks as benchmarks
import gepa_artifact.benchmarks.dspy_program as dspy_program
import gepa_artifact.benchmarks.benchmark as benchmark_mod
import gepa_artifact.benchmarks.IFBench as ifbench
import gepa_artifact.benchmarks.AIME as aime
import gepa_artifact.benchmarks.hotpotQA as hotpot
import gepa_artifact.benchmarks.hover as hover
import gepa_artifact.benchmarks.papillon as papillon
import gepa_artifact.benchmarks.livebench_math as livebench

bench_map = {
    "IFBench": ifbench,
    "AIME": aime,
    "hotpotQA": hotpot,
    "hover": hover,
    "papillon": papillon,
    "livebench_math": livebench
}

gepa_artifact.dspy_program = dspy_program
benchmarks.dspy_program = dspy_program

# Add benchmark-specific aliases recursively to cover deep nesting like benchmarks.langProBe.langProBe.langProBe.papillon
for i in range(1, 6):
    prefix = ".".join(["langProBe"] * i)
    sys.modules[prefix] = gepa_artifact
    sys.modules["benchmarks." + prefix] = gepa_artifact
    sys.modules[prefix + ".benchmarks"] = benchmarks
    
    for name, mod in {**bench_map, "dspy_program": dspy_program, "benchmark": benchmark_mod}.items():
        # Handle all combinations
        sys.modules[f"{prefix}.{name}"] = mod
        sys.modules[f"benchmarks.{prefix}.{name}"] = mod
        sys.modules[f"{prefix}.benchmarks.{name}"] = mod
        sys.modules[f"benchmarks.{prefix}.benchmarks.{name}"] = mod
        # Attach to benchmarks object for attribute access
        if name in bench_map:
            setattr(benchmarks, name, mod)

sys.modules['benchmarks'] = benchmarks
sys.modules['benchmarks.dspy_program'] = dspy_program
sys.modules['dspy_program'] = dspy_program

class MockArborMeta(type):
    def __getattr__(cls, name):
        class MockArborInner:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return []
            def __getstate__(self): return {}
            def __setstate__(self, state): pass
        MockArborInner.__name__ = name
        MockArborInner.__qualname__ = f"MockArbor.{name}"
        setattr(cls, name, MockArborInner)
        return MockArborInner

# Mock missing arbor client for GRPO pkls
class MockArbor(metaclass=MockArborMeta):
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return []
    class ArborProvider:
        def __init__(self, *args, **kwargs): pass
        def __getstate__(self): return {}
        def __setstate__(self, state): pass
    class ArborTrainingJob:
        def __init__(self, *args, **kwargs): pass
        def __getstate__(self): return {}
        def __setstate__(self, state): pass
    class ArborReinforceJob:
        def __init__(self, *args, **kwargs): pass
        def __getstate__(self): return {}
        def __setstate__(self, state): pass

sys.modules["dspy.clients.lm_local_arbor"] = MockArbor
dspy.clients.lm_local_arbor = MockArbor

for name, mod in bench_map.items():
    sys.modules[f"benchmarks.{name}"] = mod

from scripts.experiment_configs import get_benchmarks, LM_CONFIGS, MAX_CONTEXT_LENGTH

def create_lm(lm_config):
    config = lm_config.copy()
    
    # Use OpenRouter if key is available, otherwise fall back to OpenAI
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    extra_kwargs = {}
    
    if openrouter_key:
        api_key = openrouter_key
        api_base = "https://openrouter.ai/api/v1"
        
        orig_model = config.get("model", "")
        # Allow metric LMs (like gpt-4o-mini) or specified judges to bypass the Qwen3-8B override
        if "judge" in config.get("name", "").lower() or "gpt" in orig_model.lower() or "gpt" in config.get("name", "").lower():
            model_name = orig_model if orig_model else "openai/gpt-4o-mini"
            if not model_name.startswith("openrouter/"):
                model_name = "openrouter/" + model_name.replace("openai/", "")
        else:
            # Force target models to Qwen3 8B on Alibaba as requested
            model_name = "openrouter/qwen/qwen3-8b"
            extra_kwargs["extra_body"] = {
                "provider": {
                    "allow_fallbacks": False,
                    "order": ["Alibaba"]
                }
            }
        
        # Disable thinking/reasoning to avoid conflicts with JSON mode
        if "extra_body" not in extra_kwargs:
            extra_kwargs["extra_body"] = {}
        extra_kwargs["extra_body"].update({
            "include_reasoning": False,
            "enable_thinking": False
        })
    elif openai_key:
        api_key = openai_key
        api_base = None
        model_name = config.get("model", "openai/gpt-4.1-mini-2025-04-14")
    else:
        raise ValueError("Neither OPENROUTER_API_KEY nor OPENAI_API_KEY found.")

    # Align max_tokens with MAX_CONTEXT_LENGTH from experiment_configs.py
    if "qwen" in model_name.lower():
        max_tokens = 8192
    else:
        max_tokens = 16384 # GPT-4.1-mini supports much larger contexts

    # Original fixed_config from run_experiments.py
    fixed_config = {
        "max_tokens": max_tokens,
        "num_retries": 0, # Aligned with original script
        "api_key": api_key,
        "model": model_name,
        **extra_kwargs
    }
    if api_base:
        fixed_config["api_base"] = api_base
    
    # Remove 'name' and 'model' from config to avoid conflicts with fixed_config
    # but KEEP other hyperparameters like temperature, top_p, etc.
    filtered_config = {k: v for k, v in config.items() if k not in ["name", "model", "api_key", "api_base", "launch_kwargs", "train_kwargs"]}
    
    return dspy.LM(**filtered_config, **fixed_config)

def patch_program(module, lm):
    """Recursively patch the program to use the correct LM for all submodules and predictors."""
    if hasattr(module, 'set_lm'):
        try:
            module.set_lm(lm)
        except:
            pass
    
    # Specifically for PAPILLON where untrusted_model is used as a callable
    if hasattr(module, 'untrusted_model'):
        module.untrusted_model = lm
    
    # Handle DSPy internal predictors/submodules
    if hasattr(module, 'predictors'):
        for predictor in module.predictors():
            predictor.lm = lm
            
    # Recurse into named submodules if available (DSPy 2.5+)
    if hasattr(module, 'named_submodules'):
        try:
            for _, submodule in module.named_submodules():
                patch_program(submodule, lm)
        except:
            pass
    
    # Fallback for generic objects
    if hasattr(module, '__dict__'):
        for attr_name, attr_value in module.__dict__.items():
            if isinstance(attr_value, dspy.Module) and attr_value is not module:
                patch_program(attr_value, lm)

def run_evaluation(benchmark_filter=None):
    pkl_dir = Path("final_pkls")
    seed_0_dir = Path("seed_0")
    
    results_path = Path("final_evaluation_results.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            try:
                results = json.load(f)
                # Ensure it's a dict
                if not isinstance(results, dict):
                    results = {}
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    benchmark_metas = get_benchmarks()

    for pkl_path in pkl_dir.glob("*.pkl"):
        exp_name = pkl_path.stem
        
        # Flexible filter: all keywords in benchmark_filter must be present in exp_name
        if benchmark_filter:
            keywords = benchmark_filter.replace('_', ' ').lower().split()
            if not all(kw in exp_name.lower() for kw in keywords):
                continue
                
        # Skip if already evaluated to save cost/time
        if exp_name in results and "score" in results[exp_name]:
            print(f"Skipping {exp_name}: already evaluated.")
            continue

        config_path = seed_0_dir / exp_name / "config.json"
        config = {}
        if not config_path.exists():
            # Attempt to infer from filename
            # Pattern: <Benchmark>_<Program>_<Optimizer>_<Model>
            parts = exp_name.split('_')
            if len(parts) >= 4:
                benchmark_name = parts[0]
                program_name = parts[1]
                optimizer_name = parts[2]
                model_name = parts[3]
                
                # Find LM config
                lm_config = next((c for c in LM_CONFIGS if c['name'] == model_name), None)
                if not lm_config:
                    # Fallback to gpt-41-mini if unknown
                    lm_config = next((c for c in LM_CONFIGS if c['name'] == "gpt-41-mini"), LM_CONFIGS[0])
                
                print(f"Inferred config for {exp_name}: benchmark={benchmark_name}, program={program_name}, model={model_name}")
            else:
                print(f"Skipping {exp_name}: config.json not found and filename pattern mismatch")
                continue
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            benchmark_name = config['benchmark_name']
            program_name = config['program_name']
            lm_config = config['lm_config']
        
        print(f"\n>>> Evaluating {exp_name} ...")
        
        # Find matching benchmark
        benchmark_meta = None
        for meta in benchmark_metas:
            # Check by meta.name if available
            if meta.name == benchmark_name:
                benchmark_meta = meta
                break
        
        if not benchmark_meta:
            # Check by class name without initializing
            for meta in benchmark_metas:
                # meta.benchmark is the class
                if meta.benchmark.__name__ == benchmark_name:
                    benchmark_meta = meta
                    break
        
        if not benchmark_meta:
            # Fallback for name mismatches (partial match)
            for meta in benchmark_metas:
                m_name = meta.name or meta.benchmark.__name__
                if benchmark_name.startswith(m_name) or m_name.startswith(benchmark_name):
                    benchmark_meta = meta
                    break
        
        if not benchmark_meta:
            print(f"Error: Could not find benchmark {benchmark_name}")
            continue

        # Load program
        program = None
        for p in benchmark_meta.program:
            p_name = getattr(p, "_name", p.__class__.__name__)
            if p_name == program_name:
                program = p
                break
        
        if not program:
            program = benchmark_meta.program[0]

        # Initialize benchmark ONLY when needed
        benchmark = benchmark_meta.benchmark()
        testset = benchmark.test_set
        
        print(f"Benchmark: {benchmark_name}, Program: {program_name}, Testset size: {len(testset)}")

        # Configure LM and disable cache to ensure fresh results
        lm = create_lm(lm_config)
        dspy.settings.configure(lm=lm, cache=False)
        
        # Load pkl state
        try:
            with open(pkl_path, 'rb') as f:
                optimized_program = pickle.load(f)
            # Patch the program recursively with the new LM
            patch_program(optimized_program, lm)
            # Extra safety
            if hasattr(optimized_program, 'set_lm'):
                optimized_program.set_lm(lm)
        except Exception as e:
            print(f"Error loading pkl {pkl_path}: {e}")
            continue

        # Run evaluation
        evaluate = dspy.Evaluate(
            devset=testset,
            metric=benchmark_meta.metric,
            num_threads=10, # Reduced for stability without cache
            display_progress=True,
            provide_traceback=True
        )
        
        # Patch Papillon judge if necessary (it hardcodes OpenAI)
        if "papillon" in benchmark_name.lower():
            try:
                from gepa_artifact.benchmarks.papillon import papillon_utils
                
                # Always use a strong judge for Papillon evaluation
                metric_lm_name = "openai/gpt-4o-mini"
                print(f"Patching Papillon judge to use {metric_lm_name}...")
                judge_lm = create_lm({"model": metric_lm_name, "name": "judge"})
                
                if hasattr(papillon_utils.llm_judge, 'set_lm'):
                    papillon_utils.llm_judge.set_lm(judge_lm)
                else:
                    patch_program(papillon_utils.llm_judge, judge_lm)
            except Exception as e:
                print(f"Warning: Could not patch Papillon judge: {e}")
        
        try:
            score = evaluate(optimized_program)
            print(f"Score for {exp_name}: {score}")
            
            # Calculate metrics from history
            total_input_tokens = 0
            total_output_tokens = 0
            total_cost = 0.0
            
            # DSPy 2.5+ uses history as a list of data objects
            for call in lm.history:
                usage = getattr(call, 'usage', {})
                if not isinstance(usage, dict):
                    usage = getattr(call, 'get', lambda x, y: {})( 'usage', {})
                
                total_input_tokens += usage.get('prompt_tokens', 0)
                total_output_tokens += usage.get('completion_tokens', 0)
                total_cost += getattr(call, 'cost', 0.0)

            # Safely get the score value for JSON serialization
            final_score = getattr(score, 'score', float(score))
            
            results[exp_name] = {
                "score": final_score,
                "benchmark": benchmark_name,
                "program": program_name,
                "model": lm.model if hasattr(lm, 'model') else str(lm),
                "lm_config": lm_config,
                "metrics": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_cost": total_cost
                },
                "timestamp": str(Path(pkl_path).stat().st_mtime) 
            }
            
            # Real-time save after each experiment
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Progress saved to {results_path}")
            
        except Exception as e:
            print(f"Error during evaluation of {exp_name}: {e}")
            results[exp_name] = {"error": str(e)}
            # Save error state too
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

    print(f"\nAll tasks processed. Final summary in {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name (e.g. AIME)")
    args = parser.parse_args()

    if "OPENROUTER_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    run_evaluation(benchmark_filter=args.filter)
