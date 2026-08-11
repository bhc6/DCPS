# Third-party notices

This repository vendors and adapts code from the projects below. Every upstream is MIT
licensed, which is why the monorepo is also MIT (see [LICENSE](LICENSE)). MIT requires
the original copyright and permission notices to travel with the code, so each
**vendored** upstream's `LICENSE` file is kept in its own directory — do not delete
them when assembling or pruning the monorepo.

This monorepo is **self-contained**: it clones and installs without fetching any other
repository. That includes DSPy, which is vendored here rather than cloned at setup time —
see the DSPy row below for why the upstream `setup_gepa_repo.sh` is therefore optional.

Paths below are the **published monorepo** paths. Before assembly the same trees are
`gepa/`, `GFB/` and `project/` on disk; the assembly script renames them to
`dcps-compound/`, `stableprompt-dcps/` and `reproduction/`.

| Upstream | Copyright | License file in tree | How it is used |
|---|---|---|---|
| GEPA | © 2025 Lakshya A Agrawal | `dcps-compound/LICENSE` | Forked. The DCPS-Compound entrypoints live in `examples/` alongside the upstream optimizer. |
| gepa-artifact | © 2025 Lakshya A Agrawal | `dcps-compound/gepa-artifact/LICENSE` | Vendored benchmark suite (`HotpotQABench`, `PAPILLON`, metrics). Our entrypoints import from it unmodified. |
| SigOpt evalset | © 2016 SigOpt | `dcps-compound/examples/blackbox/evalset/LICENSE` | Vendored blackbox-optimization test functions, inherited from the GEPA fork. No DCPS result depends on it. |
| Automatic Prompt Engineer (APE) | © 2022 keirp | `stableprompt-dcps/automatic_prompt_engineer/LICENSE.md` | Vendored under the StablePrompt audit for Instruction-Induction / BBII task data and evaluation helpers. |
| DSPy (`gepa_study` fork) | © 2023 Stanford Future Data Systems | `dcps-compound/gepa-artifact/gepa_artifact/utils/dspy/LICENSE` | Vendored. Supplies `dspy.Evaluate`, `dspy.Predict`, `dspy.ChainOfThought`, `dspy.LM` and `dspy.evaluate.answer_exact_match` to every DCPS-Compound entrypoint. See the note below. |

### Why DSPy is vendored rather than cloned

Upstream `gepa-artifact` fetches it at setup time — `setup_gepa_repo.sh:2` runs

```bash
git clone https://github.com/gepa-ai/dspy.git ./gepa_artifact/utils/dspy
cd ./gepa_artifact/utils/dspy && git checkout gepa_study
```

and `gepa-artifact/.gitignore:3` excludes that path so it is never committed. This
monorepo vendors the checkout instead, for three reasons:

1. **It is a dependency in name only if it is absent.**
   `dcps-compound/gepa-artifact/pyproject.toml` declares
   `dspy = { path = "gepa_artifact/utils/dspy", editable = true }` — a *path* dependency.
   Without the directory the install fails to resolve; it does not fall back to PyPI.
2. **PyPI DSPy is not a substitute.** The entrypoints need the `gepa_study` branch, which
   is not a released version.
3. **A clone is a dependency on another repository**, which this monorepo deliberately
   does not have. Vendoring pins the exact code the paper's numbers came from, so the
   reproduction cannot drift when the fork moves.

Because MIT requires the notice to travel with the code, the vendored
`.../utils/dspy/LICENSE` ships and must not be pruned. `setup_gepa_repo.sh` is kept for
provenance and still works, but it is **not** a required install step here — running it
would re-clone over the vendored copy.

**Arbor is not vendored, and is not needed.** `setup_gepa_repo.sh:3` also clones
`Ziems/arbor@gepa_study`, and `pyproject.toml:61` names it under `[tool.uv.sources]`. It is
nonetheless absent here without breaking anything: `arbor-ai` does not appear in that file's
`dependencies` array, so uv never resolves the source entry. Arbor is a real dependency only
of `pyproject_grpo.toml`, for local GRPO training on GPUs. DCPS-Compound is inference-only
through OpenRouter and never imports it, so no Arbor code and no Arbor license ships.

**One DSPy fork, not two.** `gepa-reproduction/` carries its own copy of `gepa_artifact/`, and
upstream that copy has its own setup-time clones of DSPy and Arbor under
`gepa_artifact/utils/`. Those are **not** published: shipping a second 414-file DSPy tree would
duplicate the fork and let the two drift, and it is drift that would quietly invalidate a
reproduction. So `gepa-reproduction/pyproject.toml`'s `dspy` path dependency is repointed at the
single vendored fork under `dcps-compound/`, and its inert `arbor-ai` source entry is commented
out for the same reason it is absent there. The MIT notice travels with the one copy that ships.

## Also derived from, not vendored

**StablePrompt** (Kwon et al., 2024) — the StablePrompt-PPO / StablePrompt-DCPS
scripts in `stableprompt-dcps/` follow its configuration and scoring protocol. Two reproduction
fixes were applied: a train/test overlap in Instruction Induction, and missing batch
weights in the Softmax-difference metric. Cite the original paper alongside ours.

## Benchmark data

Benchmark datasets (HotpotQA, PUPA, HoVer, IFBench, LiveBench-Math, AIME-2025, GLUE,
SuperGLUE, MMLU, Instruction Induction, BigBench-II) keep their own upstream licenses
and terms. They are downloaded at runtime rather than redistributed here.

Two HoVer artifacts are named explicitly because they are the largest and the only ones a
run writes back into the source tree: the Wikipedia abstracts corpus
`wiki.abstracts.2017.jsonl` (with its `.tar.gz`) and the `bm25s_retriever/` index built
from it. `dcps-compound/examples/hover/hover_program.py` downloads and builds both on
first run. They are excluded from this repository — each is far over GitHub's 100 MB
per-file limit, and both are regenerable. This is the one place where "self-contained"
stops at code and configuration: third-party corpora are still fetched on demand.
