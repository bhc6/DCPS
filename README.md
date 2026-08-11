# Simplicity Goes Far: Auditing Prompt Optimizers with Demonstration-Conditioned Prompt Search

Code, data, and logs for the paper. **DCPS = Demonstration-Conditioned Prompt
Search** — a deliberately minimal control: propose a prompt conditioned on a fresh
draw of demonstrations, evaluate it, keep the best. No reflection, no mutation
operators, no Pareto front, no learned update.

The claim is not that DCPS is a better optimizer. It is that a control this simple
recovers most of the reported gains of much heavier methods, which means those gains
are not by themselves evidence for the machinery those methods introduce.

Two audits, one per family:

| Audit | Compared against | Tree |
|---|---|---|
| **DCPS-Compound** | GEPA, GEPA-MERGE, MIPROv2-Heavy on compound AI systems | [dcps-compound/](dcps-compound/) |
| **StablePrompt-DCPS** | StablePrompt-PPO (RL prompt optimization) | [stableprompt-dcps/](stableprompt-dcps/) |

## Experiment logs are public

Every run behind every number is on Weights & Biases:
**https://wandb.ai/awesome-prompt/projects**

Those 39 projects are the raw source. This repo also ships a **frozen snapshot**
(`results/raw_wandb_data.csv`) so the analysis reproduces with no wandb account and
no network:

- `results/fetch_wandb_data.py` — regenerate-only provenance. Reads `WANDB_ENTITY`
  (default `awesome-prompt`) and the 39-project list. `wandb` is an optional extra;
  the import is guarded, so reproduction never needs it.
- `results/clean_paper_data.csv` — the exact rows the paper's tables are computed from.
- `results/dcps_naming.py` — **single source of truth** for historical run name →
  canonical paper name.

## Layout

```
reproduction/       analysis, frozen wandb snapshot, naming crosswalk, the paper
                    source (main_v3.tex). Root LICENSE / THIRD_PARTY_NOTICES.md are
                    copies of the ones here.
dcps-compound/      fork of GEPA (Agrawal et al., 2025). DCPS-Compound entrypoints live
                    in examples/ next to the upstream optimizer, so both run through
                    the same harness on the same data.
  gepa-artifact/    benchmark suite: HotpotQABench, PAPILLON, metrics. Includes the
                    vendored DSPy gepa_study fork under gepa_artifact/utils/dspy/.
  dcps_supplement/  per-run summaries for the DCPS-Compound cells + dcps_results.csv
  case_study/       qualitative prompt analyses of what DCPS vs GEPA discover. All
                    scores are aligned to the paper's Table 2(b); RECONCILE_PLUS_VS_PAPER.md
                    is the run-id crosswalk behind them. APPENDIX_PROMPTS_FULL.md and
                    gepa_prompts_extracted.md are verbatim prompt dumps.
stableprompt-dcps/  StablePrompt-PPO vs StablePrompt-DCPS. The *_gfb.py file names keep
                    the historical "GFB" tag so they still line up with the WandB run
                    history; that directory's README.md documents the full mapping.
gepa-reproduction/  standalone GEPA reproduction harness. Its README reports gpt-4o-mini
                    on the full 150-sample test set, which is NOT the paper's baseline
                    configuration (gpt-4.1-mini / Qwen3-8B) — do not read its numbers as
                    the paper's GEPA column.
```

This repository is **self-contained**: cloning it is enough. Nothing is fetched from
another repository at install time, DSPy included — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for why the vendored `gepa_study` fork
ships here instead of being cloned.

Commands below are written with the development layout (`project/`, `gepa/`, `GFB/`),
since that is where the code is authored. In this repository read `project/` as
`reproduction/`, `gepa/` as `dcps-compound/`, and `GFB/` as `stableprompt-dcps/`.

One thing is deliberately **not** published: `dcps-artifact/`, the original run archive
the HotpotQA and PUPA entrypoints were ported from. The ports are in
`dcps-compound/examples/`, so shipping the archive too would mean two near-identical
copies of the same scripts, the older one carrying hardcoded per-task API key names.
What the ports changed deliberately: keys come from environment variables, and wandb and
dataset paths are not machine-pinned.

Also absent, and regenerated on first run rather than withheld: HoVer's Wikipedia
abstracts corpus and the BM25 index built from it. Both are far over GitHub's per-file
limit; `examples/hover/hover_program.py` downloads and rebuilds them.

`reproduction/` holds the paper's **source** — `main_v3.tex`, `references.bib`, the figures,
`llncs.cls`/`splncs04.bst`, the compiled PDF — but not the workspace around it: superseded
drafts, LaTeX build products, proof renders, and internal review notes are left out as clutter.
Two deliberate omissions there are worth naming rather than leaving as gaps. Our full-text
extractions of the papers we compare against are **not** here: reading them was research,
redistributing them would be republishing text we hold no rights to, so this repo cites those
papers and ships none of their content. Nor is the audit log of this repo's own preparation,
which narrates a since-rotated credential in detail.

Naming: the paper's canonical names are `DCPS-Compound`, `StablePrompt-DCPS`, and
`StablePrompt-PPO`. Historical logs and directories use `GFB`, `G&F`, and `APPO`.
`results/dcps_naming.py` maps every historical name to its canonical one, porting the
original `determine_method` / `_ppo_ran` logic verbatim so the crosswalk cannot drift
from the pipeline that produced the tables.

## Headline result — DCPS-Compound

Cost–performance, GPT-4.1-mini, macro-averaged over the 6 benchmarks. `RER` is
relative error reduction over the unoptimized baseline `B`; `T_M` is total token cost
in millions; `RCEI` is relative cost-efficiency improvement.

| Method | RER | T_M | mean log2(T_M/T_B) | RCEI | Rollouts | Cost |
|---|---|---|---|---|---|---|
| Baseline (unoptimized, B) | 0 | 1.06 | 0 | — | — | $0.43 |
| MIPROv2-Heavy | +12.70% | 27.11 | 4.67 | +2.66 | 3,936 | $10.84 |
| GEPA | +24.15% | 28.36 | 4.73 | +5.30 | 3,936 | $11.34 |
| GEPA-MERGE | +24.87% | 24.07 | 4.50 | +5.90 | 3,936 | $9.63 |
| **DCPS-Compound** | **+26.23%** | **19.26** | **3.85** | **+7.20** | **1,761** | **$7.70** |

DCPS-Compound reaches the highest error reduction on **45% of the rollout budget** and
two thirds of the token cost. Task-level accuracy (best per column starred):

| | Hotpot | IFBench | HoVer | PUPA | AIME | LB-Math | Macro |
|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | | | | | | | |
| Baseline | 40.33 | 38.61 | 33.67 | 81.55 | 47.33 | 65.01 | 51.08 |
| MIPROv2-Heavy | 58.67 | 38.61 | 44.00 | 78.67 | 58.00 | 67.46 | 57.57 |
| GEPA | *63.33* | 40.65 | 53.33 | 87.57 | 59.33 | *70.57* | *62.46* |
| GEPA-MERGE | 61.33 | 29.08 | 52.00 | 84.81 | *62.67* | 66.28 | 59.36 |
| DCPS-Compound | 61.33 | *43.88* | *56.67* | *90.02* | 55.33 | 65.08 | 62.05 |
| **GPT-4.1-mini** | | | | | | | |
| Baseline | 36.10 | 48.13 | 40.33 | 80.81 | 40.00 | 55.80 | 50.20 |
| MIPROv2-Heavy | 55.00 | 51.19 | 47.33 | 83.59 | 46.67 | 57.30 | 56.85 |
| GEPA | *65.00* | 49.83 | 49.67 | 90.10 | *48.00* | *64.21* | *61.14* |
| GEPA-MERGE | 63.33 | 49.49 | 49.00 | 93.52 | 46.67 | 61.14 | 60.53 |
| DCPS-Compound | 59.00 | *51.53* | *54.67* | *94.09* | *48.00* | 59.52 | *61.14* |

Rollout budgets, shared across backbones:

| | Hotpot | IFBench | HoVer | PUPA | AIME | LB-Math | Macro |
|---|---|---|---|---|---|---|---|
| GEPA / GEPA-MERGE / MIPROv2-Heavy | 6,871 | 3,593 | 7,051 | 2,426 | 1,839 | 1,839 | 3,936 |
| DCPS-Compound | 1,020 | 620 | 7,068 | 920 | 320 | 620 | 1,761 |

Rollout convention throughout: `(n_v + 1) x num_iterations`, where `n_v` is the
validation subset size.

Two caveats the paper states and this repo keeps visible:

- **HoVer is not budget-matched.** The small-budget DCPS run was genuinely weak, so the
  table reports the GEPA-aligned extended run (7,068 rollouts). It is the one cell where
  DCPS-Compound spends a comparable budget.
- **DCPS-Compound does not win everywhere.** GEPA leads on HotpotQA and LiveBench-Math
  with both backbones and on Qwen macro-average; GEPA-MERGE leads Qwen AIME. The result
  is "a trivial control is competitive", not "the control dominates".

## Headline result — StablePrompt-DCPS

Single-variable ablation: keep StablePrompt's protocol and swap only the PPO update for
DCPS's sample-and-select. Full table, CIs, Cohen's *d*, and per-run h/$ are in
[GFB/README.md](../GFB/README.md). Summary: PPO's learned update has limited marginal
value on template-like classification, **still leads** on BBII-Gen (+2.3) and MMLU
(+1.8), and the gap at II@100 closes — which shows budget can substitute for the
update, not that DCPS is better. The concrete win is that DCPS needs inference only,
so an A100 becomes an A40.

## Reproducing the tables (no API keys, no GPU, no wandb)

Everything in the paper's tables and figures is recomputed from the frozen snapshot.

```bash
cd project
pip install -e .                    # wandb is an optional extra, not installed here
python results/analyze_paper_data.py     # paper tables from clean_paper_data.csv
python results/export_by_task.py          # per-task breakdowns
python results/dcps_naming.py             # verify the naming crosswalk
```

Output location defaults to the script's own directory; override with
`DCPS_RESULTS_DIR`. No absolute paths are baked in.

To re-pull from wandb instead of using the snapshot (needs the optional extra and a
wandb login — the public projects are readable):

```bash
pip install -e ".[wandb]"
WANDB_ENTITY=awesome-prompt python results/fetch_wandb_data.py
```

## Re-running the searches (needs an OpenRouter key)

Copy the template and fill in one key; every per-benchmark variable falls back to it.

```bash
cp gepa/.env.example gepa/.env      # then edit — placeholders are inert
```

The entrypoints import DSPy and the benchmark classes from `gepa-artifact`. **DSPy is
vendored in this repo** at `gepa-artifact/gepa_artifact/utils/dspy/` — the `gepa_study`
branch of `gepa-ai/dspy`, which is not a released PyPI version, and which
`gepa-artifact/pyproject.toml` depends on *by path*. So the install needs no clone:

```bash
cd gepa/gepa-artifact
uv sync                    # or: pip install -e .
cd ../..
```

Do **not** run `setup_gepa_repo.sh`. It is kept for provenance — it records where the
fork came from — but it would re-clone over the vendored copy and replace the exact code
the paper's numbers came from with whatever the branch holds today.

DCPS-Compound has one entrypoint per benchmark × backbone, all under `gepa/examples/`.
For the four benchmarks that keep the upstream layout the convention is
`dynamic_fewshot.py` = Qwen3-8B and `dynamic_fewshot_gpt41mini.py` = GPT-4.1-mini:

```bash
cd gepa
python examples/hover/dynamic_fewshot.py               # HoVer,     Qwen3-8B
python examples/hover/dynamic_fewshot_gpt41mini.py     # HoVer,     GPT-4.1-mini
python examples/ifbench/dynamic_fewshot.py             # IFBench,   Qwen3-8B
python examples/ifbench/dynamic_fewshot_gpt41mini.py   # IFBench,   GPT-4.1-mini
python examples/aime_math/dynamic_fewshot.py           # AIME-2025, Qwen3-8B
python examples/aime_math/dynamic_fewshot_gpt41mini.py # AIME-2025, GPT-4.1-mini
python examples/livebench_math/dynamic_fewshot.py               # LB-Math, Qwen3-8B
python examples/livebench_math/dynamic_fewshot_gpt41mini.py     # LB-Math, GPT-4.1-mini
```

`examples/aime_math/` also holds seven exploratory siblings (`_agnostic`,
`_agnostic_nb`, `_litellm*`, `_test`) that log to the separate `dynamic-fewshot-agnostic`,
`dynamic-fewshot-agnostic-nb`, and `aime-math-litellm-agnostic-nb` projects. Those are
real runs and visible in the public logs, but the paper's AIME cells come from the two
files above — the pair that logs to `aime-math-dynamic-fewshot`.

HotpotQA and PUPA keep the file names they had in the original run archive (not published,
see above), which is why they read differently — PUPA switches backbone by env var rather
than by file:

```bash
python examples/hotpotqa/dynamic_fewshot_hotpotQA.py        # HotpotQA, GPT-4.1-mini
python examples/hotpotqa/dynamic_fewshot_hotpotQA_qwen.py   # HotpotQA, Qwen3-8B
EXP_MODE=gpt  python examples/papillon/dynamic_fewshot_nobase_subset.py   # PUPA, GPT
EXP_MODE=qwen python examples/papillon/dynamic_fewshot_nobase_subset.py   # PUPA, Qwen
```

Do not use `examples/pupa/` or `examples/hotpotqa/dcps_search.py`,
`dynamic_fewshot.py`, `dynamic_fewshot_gpt41mini.py`. Those are superseded
reconstructions kept only until they are deleted; each carries a `DISCARDED` banner and
exits on import. They optimize the wrong stage count, select Top-K instead of top-1
argmax, and do not reproduce the table.

For PUPA, `EXP_MODE` sets **all four** roles — solver, prompt generator, LLM judge, and
the untrusted external LLM. In `qwen` mode all four are `qwen/qwen3-8b`; the judge is
not pinned to GPT.

Defaults reproduce the paper: `ITERATIONS` / `NUM_ITERATIONS` 20, `NUM_THREADS` 32 (8
for the Qwen HotpotQA script, which otherwise OOMs). Runs write
`checkpoint_{gpt,qwen}.json` for resume and `best_prompts_{gpt,qwen}.json` for the
winner; both are gitignored. `gepa/dcps_supplement/generate_dcps_runs.py` regenerates
the per-run summary directories and `dcps_results.csv` from the recorded results — it
reports runs, it does not perform the search.

Selection is **top-1 argmax on the validation subset**, then a single test evaluation of
that winner. Qwen3-8B is pinned to the `alibaba` provider via
`extra_body={"provider": {"only": ["alibaba"]}}` — OpenRouter otherwise routes across
providers whose outputs differ enough to move scores.

StablePrompt-PPO vs StablePrompt-DCPS (PPO side needs a GPU; DCPS side is
inference-only):

```bash
cd GFB
python tc.py            # StablePrompt-PPO,  classification
python tc_gfb.py        # StablePrompt-DCPS, classification
# PPO:  tc.py, qa.py (via utils.py), origin_ii.py
# DCPS: tc_gfb.py, qa_gfb.py, ii_gfb.py, bbii_tc_gfb.py, bbii_tg_gfb.py
```

## License and attribution

MIT — see [LICENSE](LICENSE). Every upstream this repo vendors is also MIT; each keeps
its own `LICENSE` file in its own directory, and those must travel with the code.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) maps each vendored upstream (GEPA,
gepa-artifact, the SigOpt evalset, APE, and the DSPy `gepa_study` fork) to its in-tree
license path. DSPy's is at
`dcps-compound/gepa-artifact/gepa_artifact/utils/dspy/LICENSE` — it ships because the fork
is vendored here rather than cloned at setup time, and MIT requires the notice to travel
with the code. Do not prune it.
The notices also record StablePrompt as
derived-but-not-vendored along with the two reproduction fixes applied to it (a
train/test overlap in Instruction Induction, and missing batch weights in the
Softmax-difference metric). Cite StablePrompt (Kwon et al., 2024) and GEPA (Agrawal et
al., 2025) alongside this work.


