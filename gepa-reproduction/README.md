# GEPA (Genetic-Pareto) Reproduction

> **These numbers are not the DCPS paper's GEPA baseline.** This harness evaluates on the
> full 150-sample test set with `gpt-4o-mini`; the paper's baselines are `gpt-4.1-mini` and
> Qwen3-8B. Treat everything below as exploratory, and read the paper's GEPA column from
> `reproduction/results/clean_paper_data.csv` instead. This directory also vendors its own
> copy of `gepa_artifact/`, which will drift from `dcps-compound/gepa-artifact/`.

GEPA is **Genetic-Pareto** (Agrawal et al., 2025). Earlier revisions of this README expanded
the acronym as "Generative Evidence-based Prompt Adherence", which was never the paper's name.

This repository contains the reproduction and evaluation logic for the GEPA project. It focuses on evaluating optimized program artifacts (`.pkl` files) across various benchmarks, specifically testing their performance in reasoning-heavy tasks like AIME.

## Overview

The goal of this project is to verify the effectiveness of GEPA-optimized prompts by running them through a standardized evaluation pipeline. We provide scripts to gather results from experimental runs, recover the optimized DSPy programs, and run final evaluations against the original benchmarks.

## Key Features

- **Automated Result Gathering**: Extract `program.pkl` files from experimental directories.
- **Robust Evaluation Pipeline**: Supports evaluation via OpenAI or OpenRouter (GPT-4o-mini).
- **Environment Compatibility**: Optimized for Windows and CPU-only environments.
- **CPU-Only / API-Driven**: No local GPU required; leverages `jax[cpu]` and LLM APIs for optimization and evaluation.
- **Verified Results**: Pre-evaluated results for benchmarks like AIME.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast and reproducible dependency management.

1. **Install uv** (if not already installed):
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Initialize Environment**:
   ```powershell
   uv sync
   ```
   *Note: This will automatically create a `.venv` and install all dependencies, including CPU-optimized JAX.*

3. **API Keys**: Create a `.env` file or set your environment variables:
   ```powershell
   $env:OPENROUTER_API_KEY = "your_key_here"
   # or
   $env:OPENAI_API_KEY = "your_key_here"
   ```

## Usage

### 1. Gather Optimized Programs
If you have new experimental runs in `seed_0`, gather the pkls first:
```powershell
python gather_pkls.py
```

### 2. Run Evaluation
Run the evaluation script with a filter for the specific benchmark (e.g., AIME):
```powershell
python evaluate_pkls.py --filter "AIME"
```

### 3. Sample Check
Verify if the model is reasoning correctly by running a single sample:
```powershell
python sample_check_aime.py
```

## Reproduction Results (AIME)

Evaluated on the full test set (150 samples) using `gpt-4o-mini` via OpenRouter.

| Method | Accuracy (%) | Note |
| :--- | :--- | :--- |
| **GEPA (Optimized)** | **10.0%** | Reproduced result |
| **MIPROv2-Heavy** | **3.33%** | Baseline comparison |

*Note: AIME is a highly challenging mathematical competition benchmark. A 10.0% accuracy represents a significant improvement over standard prompt optimization methods.*

## Project Structure

- `gepa_artifact/`: Core benchmark and program logic.
- `final_pkls/`: Gathered optimized program artifacts. **37 of 41 ship here.** The four
  `*_GRPO_qwen3-8b.pkl` checkpoints (375–765 MB each) are trained model weights from local
  GPU training, every one over GitHub's 100 MB per-file limit, and this evaluation path never
  loads them. The 37 that ship are the `GEPA`, `GEPA-MERGE`, `MIPROv2-Heavy` and
  `Abl-SelectBestCandidate` artifacts, which hold prompts and demonstrations.
  **`pickle.load` executes arbitrary code from the file it reads.** Loading a `program.pkl` is
  ordinary practice in the DSPy ecosystem and these were produced locally by us, but a `.pkl`
  is not inert data.
- `seed_0/`: Original experimental data directories.
- `evaluate_pkls.py`: Main evaluation runner.
- `gather_pkls.py`: Utility to collect `.pkl` files.
- `pyproject.toml`: Dependency configuration (CPU-focused).

## CPU Mode & API Integration

This reproduction is designed to be **lightweight** and **cross-platform**:
- **No Local GPU Required**: All inference and optimization are performed via remote APIs (OpenRouter/OpenAI).
- **JAX CPU**: The environment uses `jax[cpu]` to avoid the complex installation requirements of CUDA on Windows.
- **Dependency Isolation**: Uses `uv` to manage a clean virtual environment without interfering with system-wide packages.

## Credits

Original GEPA project by [gepa-ai](https://github.com/gepa-ai/gepa-artifact).
