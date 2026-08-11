#!/usr/bin/env python3
import os, re, json, glob
from pathlib import Path
from collections import defaultdict

def extract_results_from_log(filepath):
    """Extract score and key metrics from log file"""
    results = {'filename': os.path.basename(filepath)}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for final score patterns
        final_score_match = re.search(r'(?:Final|Average|Max|Best).*?[Ss]core[:\s]+([\d.]+)', content, re.IGNORECASE)
        if final_score_match:
            results['final_score'] = float(final_score_match.group(1))
        
        # Look for accuracy/score in different formats
        acc_match = re.search(r'[Aa]ccuracy[:\s]+([\d.]+)', content, re.IGNORECASE)
        if acc_match:
            results['accuracy'] = float(acc_match.group(1))
            
        # Extract iteration info
        iter_matches = re.findall(r'Iteration\s+(\d+)', content)
        if iter_matches:
            results['iterations'] = max([int(x) for x in iter_matches])
        
        # Extract dataset size
        size_match = re.search(r'dataset_size[:\s]+(\d+)', content, re.IGNORECASE)
        if size_match:
            results['dataset_size'] = int(size_match.group(1))
            
        return results
    except Exception as e:
        results['error'] = str(e)
        return results

def parse_filename(filename):
    """Parse experiment info from filename"""
    info = {}
    fn = filename.lower()
    
    # Benchmark
    if 'hover' in fn:
        info['benchmark'] = 'HoVer'
    elif 'if_' in fn or 'ifbench' in fn:
        info['benchmark'] = 'IFBench'
    elif 'aime' in fn:
        info['benchmark'] = 'AIME'
    elif 'lb_' in fn:
        info['benchmark'] = 'LiveBench-Math'
    else:
        info['benchmark'] = 'Other'
    
    # Model
    if 'gpt41mini' in fn or 'gpt-4.1-mini' in fn:
        info['model'] = 'GPT-4.1-mini'
    elif 'qwen' in fn:
        info['model'] = 'Qwen3-8B'
    else:
        info['model'] = 'Unknown'
    
    # Experiment type
    if 'baseline' in fn and 'artifact' not in fn:
        info['type'] = 'Baseline'
    elif 'fewshot' in fn or 'dynamic' in fn:
        info['type'] = 'Dynamic Few-shot'
    elif 'artifact_base' in fn:
        info['type'] = 'Artifact Base'
    elif 'artifact' in fn:
        info['type'] = 'Artifact Optimized'
    elif 'v2' in fn:
        info['type'] = 'V2'
    else:
        info['type'] = 'Standard'
    
    return info

def main():
    # Process all log files
    logs = sorted(glob.glob('logs_*.txt'))
    
    # Group by benchmark
    by_benchmark = defaultdict(list)
    
    for log_file in logs:
        results = extract_results_from_log(log_file)
        info = parse_filename(os.path.basename(log_file))
        info.update(results)
        by_benchmark[info['benchmark']].append(info)
    
    # Print summary
    print("=" * 120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 120)
    print()
    
    for benchmark in sorted(by_benchmark.keys()):
        entries = by_benchmark[benchmark]
        print(f"\n## {benchmark}")
        print("-" * 120)
        print(f"{'Filename':<45} {'Model':<15} {'Type':<20} {'Score/Accuracy':<15} {'Iterations':<10}")
        print("-" * 120)
        
        for e in sorted(entries, key=lambda x: (x.get('model', ''), x.get('type', ''))):
            score = e.get('final_score', e.get('accuracy', 'N/A'))
            iters = str(e.get('iterations', 'N/A'))
            print(f"{e['filename']:<45} {e.get('model', '?'):<15} {e.get('type', '?'):<20} {str(score):<15} {iters:<10}")
    
    # Print detailed config for key experiments
    print("\n\n" + "=" * 120)
    print("DETAILED CONFIGURATION & RESULTS")
    print("=" * 120)
    
    # AIME experiments
    print("\n### AIME Math Experiments")
    print("- Dataset: AI-MO/aimo-validation-aime (45 train / 45 val / 150 test)")
    print("- Metric: Exact match on final answer")
    print("- Validation: Fixed head-slice of 15 from 45 val examples")
    
    # IFBench experiments
    print("\n### IFBench Experiments")
    print("- Dataset: IFBench train/test from gepa-artifact")
    print("- Metric: Task-specific (accuracy, edit distance, etc.)")
    print("- Train: 150 examples (from 300 val pool, seed 1)")
    print("- Val: 300 examples (head slice)")
    print("- Test: 294 examples")
    
    # LiveBench experiments
    print("\n### LiveBench-Math Experiments")
    print("- Dataset: livebench/math (368 total)")
    print("- Split: 121 train / 121 val / 126 test (seed 0)")
    print("- Tasks: AMPS_Hard, math_comp, olympiad")
    print("- Validation: Fixed head-slice of 30 from full val")
    
    # HoVer experiments
    print("\n### HoVer Experiments")
    print("- Dataset: HuggingFace 'hover' (3-hop only)")
    print("- Split: 150 train / 300 val / test (shuffled seed 0/1)")
    print("- Metric: discrete_retrieval_eval (gold_titles ⊆ found_titles)")
    print("- Program: 4-predictor multi-hop (summarize1, query_hop2, summarize2, query_hop3)")
    print("- Validation: Fixed head-slice of 30 from 300 val")

if __name__ == '__main__':
    main()
