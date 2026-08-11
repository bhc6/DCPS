# Recover FSPO token/cost from OpenRouter Activity

This directory now contains two recovery scripts.

## 1. What has already been checked

`audit_fspo_wandb_history.py` scans WandB summaries and history for FSPO/DynamicFewshot token counters.

Observed output:

- `fspo_wandb_history_tokens_per_run.csv`
- `fspo_wandb_history_tokens_per_cell.csv`

Current result: only two PUPA runs expose cumulative token counters in WandB history:

- GPT-4o-mini x PUPA: 5,096,511 input / 1,402,620 output / $1.6060
- GPT-4.1-mini x PUPA: 6,325,719 input / 2,018,796 output / $5.7604

Other FSPO/DynamicFewshot runs do not expose token counters in WandB summary/history.

## 2. What to export from OpenRouter

Export request-level Activity/Usage CSV from OpenRouter for the date ranges in:

`fspo_run_time_windows.csv`

The important columns are:

- request/generation timestamp
- model
- input/prompt tokens
- output/completion tokens
- cost/total cost, if available
- request id or generation id, if available

The script tries to auto-detect common column names.

## 3. Match OpenRouter Activity to FSPO runs

Run:

```powershell
python results\match_openrouter_activity_to_fspo.py path\to\openrouter_activity.csv
```

Optional: increase/decrease the matching time padding around each WandB run:

```powershell
python results\match_openrouter_activity_to_fspo.py path\to\openrouter_activity.csv --pad-mins 10
```

Outputs:

- `results/fspo_openrouter_matched_per_run.csv`
- `results/fspo_openrouter_matched_per_cell.csv`

Matching rule:

- WandB `created_at` to `created_at + runtime_mins`
- plus configurable time padding
- model alias must match the run's `config_solver_model`

## 4. Caveat

If multiple FSPO runs with the same model overlap in time, the time-window match may double-count requests. Inspect `fspo_openrouter_matched_per_run.csv` for overlaps before using numbers in the paper.
