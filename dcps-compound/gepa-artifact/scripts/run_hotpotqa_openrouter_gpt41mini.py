import argparse
import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LM_CONFIG = {
    "name": "gpt-41-mini-openrouter",
    "model": "openrouter/openai/gpt-4.1-mini",
    "api_key": "env:OPENROUTER_API_KEY_HOTPOTQA",
    "temperature": 1.0,
}

OPTIMIZER_INDEX_BY_NAME = {
    "Baseline": 0,
    "MIPROv2-Heavy": 1,
    "GEPA-MERGE": 2,
    "GEPA": 3,
}


def _openrouter_lm_config() -> dict:
    config = dict(LM_CONFIG)
    api_key = os.getenv("OPENROUTER_API_KEY_HOTPOTQA") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY_HOTPOTQA or OPENROUTER_API_KEY in your environment or .env file.")
    config["api_key"] = api_key
    api_base = os.getenv("OPENROUTER_API_BASE")
    if api_base:
        config["api_base"] = api_base
    return config


def _patched_get_optimizers(original_get_optimizers):
    patched = []
    for name, optimizer_config in original_get_optimizers():
        optimizer_config = copy.deepcopy(optimizer_config)
        optimizer_config.langProBe_configs["launch_arbor"] = False
        patched.append((name, optimizer_config))
    return patched


def _hotpotqa_only_benchmarks():
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotqa_metas

    # The artifact runner's dry-run path references ``benchmark.dev_set``,
    # which does not exist on ``HotpotQABench``. Wrap the benchmark factory so
    # every fresh benchmark instance gets ``dev_set`` set to ``test_set``.
    patched_metas = []
    for meta in hotpotqa_metas:
        original_factory = meta.benchmark

        def benchmark_factory(*args, _factory=original_factory, **kwargs):
            instance = _factory(*args, **kwargs)
            if not hasattr(instance, "dev_set"):
                instance.dev_set = instance.test_set
            return instance

        meta.benchmark = benchmark_factory
        patched_metas.append(meta)
    return patched_metas


def main():
    parser = argparse.ArgumentParser(description="Run artifact HotpotQA with GPT-4.1-mini through OpenRouter.")
    parser.add_argument(
        "--optimizer",
        choices=["Baseline", "MIPROv2-Heavy", "GEPA", "GEPA-MERGE"],
        required=True,
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY in your environment or .env file.")
    run_experiments = importlib.import_module("scripts.run_experiments")
    experiment_configs = importlib.import_module("scripts.experiment_configs")
    run_experiments_typed = cast(Any, run_experiments)
    experiment_configs_typed = cast(Any, experiment_configs)
    run_experiments_typed.wandb_api_key = os.environ["WANDB_API_KEY"]
    run_experiments_typed.get_benchmarks = _hotpotqa_only_benchmarks
    run_experiments_typed.get_optimizers = lambda: _patched_get_optimizers(experiment_configs_typed.get_optimizers)

    opt_idx = OPTIMIZER_INDEX_BY_NAME[args.optimizer]
    use_cache_from_opt = None
    if args.optimizer == "MIPROv2-Heavy":
        use_cache_from_opt = "Baseline"
    elif args.optimizer in {"GEPA", "GEPA-MERGE"}:
        use_cache_from_opt = "MIPROv2-Heavy"

    run_experiments_typed.run_experiment_and_write_results(
        bm_idx=0,
        benchmark_name="HotpotQABench",
        num_threads=args.num_threads,
        program_idx=0,
        prog_name="HotpotMultiHop",
        opt_idx=opt_idx,
        optim_name=args.optimizer,
        lm_config=_openrouter_lm_config(),
        dry_run=args.dry_run,
        use_cache_from_opt=use_cache_from_opt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
