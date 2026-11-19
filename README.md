# Ouroboros

**Auditable local model adaptation, weight editing, and bounded self-improvement experiments.**

Ouroboros gives an AI agent or a developer a controlled way to propose training data or tensor edits, train a candidate, measure its behavior, and keep or roll back the actual weights. It is an experimental research toolkit—not a demonstration of unlimited autonomous intelligence growth.

**What changed in 0.2:** the former random-gradient/random-score demonstration has been replaced with real optimization and measured evaluation. Checkpoints now contain tensors, rejected candidates really roll back, upstream sources have a verifiable inventory, and the research notes distinguish published claims from implemented and tested capabilities.

## Start here

Requires Python 3.10+ and Git. Install from this checkout as shown below, not an unrelated package with a similar name on PyPI. No model download, API key, GPU, or upstream framework installation is needed for the demo.

```bash
git clone https://github.com/Mr-Nobody-Anonymous/ouroboros.git
cd ouroboros
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
pip install -e '.[dev]'
ouroboros doctor
ouroboros demo --generations 3
python -m pytest -q
```

The demo is a **tiny NumPy LoRA color-label classifier, not an LLM or a math benchmark**. It computes actual cross-entropy gradients and scores held-out prompts. A plateau or regression is rejected rather than assigned an invented improvement. Its small fixture is for testing orchestration, not measuring generalization at scale.

Inspect `outputs/<run-id>/run.json` for baseline/candidate predictions, domain scores, decisions, data fingerprints, configuration, and the latest accepted checkpoint. `checkpoints/<run-id>/generation_*/` contains real adapter weights and a SHA-256 manifest.

The legacy command `python -m ouroboros.cli --generations 3 --domain math_logic` now runs this explicitly labeled measured demo; it does not claim to benchmark mathematics.

## Real local LLM fine-tuning

```bash
# Optional, substantially larger ML dependencies. Use the appropriate PyTorch build for your hardware.
pip install -e '.[train]'
# Edit examples/local_sft.json: point model.model_name_or_path at YOUR local model directory.
ouroboros run --config examples/local_sft.json --check-only
ouroboros run --config examples/local_sft.json
```

The Transformers/PEFT backend performs **completion-masked causal-LM SFT with LoRA**, frozen base weights, gradient accumulation/clipping, greedy evaluation, and safetensors adapter export. It only loads pre-downloaded local models, refuses remote model code and pickle weight files, and never downloads a model implicitly. Model and dataset licenses still apply. Optional 4-bit QLoRA requires CUDA and the `quantization` extra; that GPU path has not been validated here.

See [local training and data formats](docs/TRAINING.md). The tests also construct a tiny random GPT-2 locally and verify real CPU LoRA updates, frozen base tensors, label masking, rollback, and export/reload—without downloading pretrained weights. Those tests are **not** a pretrained-model capability evaluation.

## Capability status

| Capability | Actual status |
| --- | --- |
| Bounded multi-generation adaptation | Implemented: measured baseline, candidate training, promotion gate, rollback, run records |
| CPU reference training | Implemented and tested: analytical LoRA classifier gradients |
| Local causal-LM training | Implemented and CPU integration-tested: Transformers + installed PEFT; optional QLoRA untested on GPU |
| Replay / continual-learning checks | Configurable verified anchors; compares against both original and latest accepted scores; no guarantee against collapse |
| Self-generated curricula | Explicit proposer + independent verifier callbacks, deduplication, provenance, split-leakage rejection; no built-in frontier API calls |
| Direct weight edits | Shape/finite checks, contrastive directions, norm bounds, exact top-k budgets, reversible PyTorch tensor edits; no automatic semantic diagnosis |
| Model merging | Local task arithmetic, DARE, TIES, SLERP; mergekit recipe generation and explicit execution bridge |
| Factual editing | EasyEdit bridge calls the actual installed upstream editor or raises; separate setup and method hyperparameters required |
| Architecture experiments | Validated blueprints, array-level head scaling/steering/pruning, explicit copied Llama/Qwen-style layer-stack candidates; not automatic architecture search |
| Test-time compute / preferences | Real generator/verifier/judge callback interfaces, Best-of-N, voting, correctness-filtered exports and preference curation; not default DPO/GRPO training |
| Adversarial rounds | Verified challenge/learner interface plus small deterministic fixtures; not a trained minimax adversary |
| Code/loss changes | AST-targeted patch proposals and a bounded arithmetic objective interpreter; generated Python is never auto-executed or merged |
| Storage | Atomic tensor checkpoints, checksums, actual-size Pareto retention; only owned checkpoints are eligible for pruning |

## Sources: twelve, not twelve magically integrated engines

Seven pinned submodules and five preserved vendored snapshots are inventoried in [`external/sources.lock.json`](external/sources.lock.json). Existing vendored files were not discarded or silently attributed to invented commits.

- **Existing:** EasyEdit, mergekit, SEAL, PEFT, TRL, repeng, self-instruct.
- **Added:** original weight-space experiment logs, WSP-Scope, WeightBench, lm-evaluation-harness, Darwin Gödel Machine.
- **Important:** WSP-Scope currently contains only a paper PDF and no runnable code/license grant. Research logs and reference source are not installed training engines. Dependency environments and licenses differ.

```bash
# Fetch source only, at locked revisions; no package installation or upstream code execution.
ouroboros repos sync
ouroboros repos verify
# Or initialize all seven Git submodules directly:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1
```

See [the source inventory](external/README.md), [comparative analysis](analysis/REPOSITORY_DEEP_DIVE.md), and [integration boundaries](docs/INTEGRATIONS.md). Run the core without initializing any submodule if you do not need those references.

## Research, checked against primary sources

- [Paper index](papers/PAPERS_INDEX.md): 33 references, machine-readable [catalogue](papers/research.json), and [BibTeX](papers/references.bib).
- [Recent research review](papers/RECENT_RESEARCH_2026.md): **11 sources first released between August 19 and November 19, 2026**, with review depth and limitations recorded. This is a curated selection, not an exhaustive census.
- [The Astra/Qwen work](papers/Sparks_of_Weight_Space_Programming.md): MIT CSAIL's November 2 research **blog post**, not Harvard or a verified peer-reviewed paper. The authors report MATH-500 **7.9% → 48.1%** (+40.2 percentage points) for a particular repetition repair; Ouroboros has not reproduced it.
- [Self-evolving architecture](docs/SELF_EVOLVING_ARCHITECTURE.md): how data, weights, evaluation, proposals and storage actually connect.

Research notes are original summaries and links. Papers and model checkpoints are not automatically redistributed; upstream rights and licenses are retained. [Corrections to earlier claims](docs/AUDIT_AND_MIGRATION.md) include a previously miscited topology paper, overstated guarantees, and the incorrect assertion that the old release contained a 600 MB bundle.

## Useful commands

```bash
ouroboros objective 'loss * beta / (1 + step)' --loss 2 --beta 0.1 --step 1
ouroboros storage audit --root .
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001
# Add --apply only after inspecting the pruning preview.
ouroboros evaluate --checkpoint checkpoints/RUN_ID/generation_0001 --data examples/data/validation.json
python scripts/build_release.py
```

A NumPy demo checkpoint must be evaluated on the demo fixture, not the arithmetic example above; use `examples/data/demo_validation.json` for that case. Paths shown with `RUN_ID` are placeholders to replace with a real run's identifier.

## Storage, Git LFS, and releases

Code, configuration, small datasets, and research notes stay ordinary readable Git files. Only explicitly approved files under `artifacts/objects/` are assigned LFS attributes. Checkpoints, virtual environments, caches and release archives are ignored by default.

**No historical 600 MB cache/checkpoint payload was found in the starting checkout; the existing v0.1.0 release had zero uploaded assets.** LFS configuration cannot recover deleted files. The [artifact inventory](artifacts/manifest.json) is deliberately empty until real files are registered. With Git LFS installed, a normal clone ordinarily downloads tracked LFS objects; use `GIT_LFS_SKIP_SMUDGE=1` for pointers only, then `git lfs pull` when ready.

[Storage/LFS guide](docs/STORAGE_AND_RELEASES.md) explains registration, retrieval, costs/limits, safe retention, deterministic release bundles, and what is deliberately not uploaded. There is no automatic cleanup of system caches or unrelated files.

## Development and boundaries

```bash
python -m pytest -q
python -m unittest discover -s tests -v  # compatibility smoke tests; pytest runs the full suite
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
```

See [testing](docs/TESTING.md), [contributing](CONTRIBUTING.md), and [security](SECURITY.md). The reusable controller has cooperative step/time budgets, not a hardened execution boundary. Never run generated Python or an upstream autonomous agent with host secrets, write access to evaluation data, or the Docker socket.

Licensed under [MIT](LICENSE) for original Ouroboros code. Third-party sources and datasets retain their own licenses.
