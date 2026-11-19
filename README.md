# Ouroboros

**Auditable local model adaptation, weight editing, and bounded self-improvement experiments.**

Ouroboros gives an AI agent or developer a controlled way to propose training data or tensor edits, measure a candidate's behavior, and keep or roll back actual weights. It is an experimental research toolkit—not a demonstration of unlimited autonomous intelligence growth.

**New in 0.3:** read-only safetensors inspection, measured single-tensor edit trials, accepted-only reversible sparse patches, and an optional final-test split evaluated once after training selection. Trials always restore the live model; applying a patch is a separate explicit choice. See the [weight-edit guide](docs/WEIGHT_EDITING.md) and [release notes](docs/RELEASE_NOTES_0_3_0.md).

Version 0.2 replaced the former random-gradient/random-score demo with real optimization, measured evaluation, tensor checkpoints and actual rollback. Research/source inventories distinguish reported claims from locally tested behavior; **no Astra reproduction or pretrained capability gain is claimed**.

## Start here

Requires Python 3.10+ and Git. Install this source, not an unrelated package with a similar name on PyPI. No model download, API key, GPU or upstream framework installation is needed for the NumPy demo.

```bash
git clone --branch v0.3.0 --single-branch https://github.com/Mr-Nobody-Anonymous/ouroboros.git
cd ouroboros
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
pip install -e '.[dev]'
ouroboros doctor
ouroboros demo --generations 3
python -m pytest -q -m 'not ml'
```

The release tag provides this experimental version even while its [pull request](https://github.com/Mr-Nobody-Anonymous/ouroboros/pull/1) remains unmerged; the historical default branch is not silently replaced. Core dependencies are NumPy, PyYAML and safetensors. Torch/Transformers/PEFT are optional.

The demo is a **tiny NumPy LoRA color-label classifier, not an LLM or math benchmark**. It computes actual cross-entropy gradients and scores held-out prompts. A plateau/regression is rejected rather than assigned an invented improvement. Its small fixture tests orchestration, not generalization at scale.

Inspect `outputs/<run-id>/run.json` for baseline/candidate predictions, domain scores, decisions, data fingerprints, configuration, final-test results and the latest accepted checkpoint. `checkpoints/<run-id>/generation_*/` contains real adapter weights and a SHA-256 manifest.

The legacy command `python -m ouroboros.cli --generations 3 --domain math_logic` routes to this explicitly labeled measured demo; it does not benchmark mathematics.

## Real local LLM fine-tuning

```bash
# Optional, substantially larger ML dependencies. Use a PyTorch build suitable for your hardware.
pip install -e '.[train]' -c constraints/ml-tested.txt
# Edit examples/local_sft.json: point model.model_name_or_path at YOUR local model directory.
ouroboros run --config examples/local_sft.json --check-only
ouroboros run --config examples/local_sft.json
```

The Transformers/PEFT backend performs **completion-masked causal-LM SFT with LoRA**, frozen base weights, gradient accumulation/clipping, greedy evaluation and safetensors adapter export. It loads only pre-downloaded local models, refuses remote model code and pickle weights, and never downloads a model implicitly. Model/dataset licenses still apply. Optional 4-bit QLoRA needs CUDA and the `quantization` extra; that GPU path has not been validated here.

Optional `data.test` is disjoint from train/replay/validation, blocked from generated curricula and evaluated once after selection—not used for promotion. Repeated validation can still overfit, and normalized prompt deduplication is not semantic decontamination.

See [local training and data formats](docs/TRAINING.md). Tests generate a tiny random GPT-2 locally and verify real CPU LoRA updates, frozen base tensors, label masking, rollback and export/reload without pretrained downloads. Those are **not** pretrained-model capability evaluations.

## Inspect and test weight edits

```bash
# Core installation: no Torch/model load; --values samples a bounded flat prefix.
ouroboros weights inspect --file /path/to/model.safetensors --limit 20 --values 8
# Requires Torch: repairs one entry of a 2×2 classifier, NOT an LLM benchmark.
ouroboros weights demo --output-dir outputs/weight-trials
# Schema/data check only; replace model/license/provenance/direction before a real trial.
ouroboros weights trial --config examples/weight_trial.json --check-only
# Use the actual accepted patch path returned by your trial, not this placeholder.
ouroboros weights verify-patch --patch outputs/weight-trials/ACTUAL_ID/patch
```

Direct trials use a frozen offline causal-LM loader shared with SFT, but do **not** insert LoRA. Entry/norm/payload-byte limits and measured validation/retention gates apply to the selected matrix. Accepted or not, the live parameter is restored; an accepted patch is a data-only proposal, not a modified full model.

Explicit apply/reverse APIs check complete before/after **target-tensor** hashes. They do not authenticate authors, bind all other model weights or establish legal rights. Same-process callbacks are trusted, and dense working copies still need RAM. Adapter-only checkpoints cannot capture arbitrary base-weight edits. See [weight editing](docs/WEIGHT_EDITING.md) for real-model setup, independent evaluation and these limits.

## Capability status

| Capability | Actual status |
| --- | --- |
| Bounded multi-generation adaptation | Measured baseline, candidate training, promotion gate, actual rollback, run/stage records |
| CPU reference training | Implemented and tested: analytical LoRA-classifier gradients |
| Local causal-LM training | Implemented and CPU integration-tested: Transformers + installed PEFT; optional QLoRA untested on GPU |
| Replay / continual-learning checks | Verified anchors; original/latest retention gates; no guarantee against collapse |
| Self-generated curricula | Explicit proposer + independent verifier, deduplication, provenance and held-out overlap rejection; no built-in frontier API calls |
| Weight inspection | Bounded read-only safetensors metadata/value prefixes without Torch; not semantic diagnosis |
| Direct weight edits | Measured one-matrix trials, entry/norm/payload-byte gates, unconditional restoration and accepted-only exact reversible sparse patches; no automatic layer/direction discovery |
| Final-test isolation | Optional disjoint `data.test`, blocked from proposals/selection, scored once after selection; not semantic decontamination |
| Model merging | Local task arithmetic, DARE, TIES, SLERP; mergekit recipe generation and explicit execution bridge |
| Factual editing | EasyEdit bridge calls installed upstream editor or raises; separate setup/method hyperparameters required |
| Architecture experiments | Validated blueprints, array-level head scaling/steering/pruning, copied Llama/Qwen-style layer-stack candidates; not automatic architecture search |
| Test-time compute / preferences | Generator/verifier/judge callbacks, Best-of-N, voting, correctness-filtered exports and preference curation; not default DPO/GRPO training |
| Adversarial rounds | Verified challenge/learner interface plus deterministic fixtures; not a trained minimax adversary |
| Code/loss changes | AST-targeted patch proposals and bounded arithmetic objective interpreter; generated Python never auto-executed/merged |
| Storage | Atomic tensor checkpoints, checksums and actual-size Pareto retention; only owned checkpoints eligible for pruning |

## Sources: twelve, not twelve automatically integrated engines

Seven pinned submodules and five preserved vendored snapshots are inventoried in [`external/sources.lock.json`](external/sources.lock.json). Existing files were not discarded or attributed to invented commits.

- **Existing:** EasyEdit, mergekit, SEAL, PEFT, TRL, repeng, self-instruct.
- **Added:** original weight-space experiment logs, WSP-Scope, WeightBench, lm-evaluation-harness, Darwin Gödel Machine.
- **Important:** WSP-Scope contains only a paper PDF and no runnable code/license grant at the locked revision. Research logs/reference source are not installed engines. Dependency environments/licenses differ.

```bash
# Fetch source only at locked revisions; no dependency installation or upstream execution.
ouroboros repos sync
ouroboros repos verify
# Or initialize all seven submodules directly:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1
```

See the [source inventory](external/README.md), [comparative analysis](analysis/REPOSITORY_DEEP_DIVE.md) and [integration boundaries](docs/INTEGRATIONS.md). The core runs without initializing submodules.

## Research, checked against primary sources

- [Paper index](papers/PAPERS_INDEX.md): 33 references, machine-readable [catalogue](papers/research.json) and [BibTeX](papers/references.bib).
- [Recent research review](papers/RECENT_RESEARCH_2026.md): **11 sources first released August 19–November 19, 2026**, with review depth/limitations recorded. A curated selection, not an exhaustive census.
- [The Astra/Qwen work](papers/Sparks_of_Weight_Space_Programming.md): MIT CSAIL's November 2 research **blog**, not Harvard or a verified peer-reviewed paper. Authors report MATH-500 **7.9% → 48.1%** (+40.2 percentage points) for a particular repetition repair; Ouroboros has not reproduced it.
- [Self-evolving architecture](docs/SELF_EVOLVING_ARCHITECTURE.md): how data, weights, evaluation, proposals and storage actually connect.

These are original summaries and links. Papers/checkpoints are not automatically redistributed; upstream rights/licenses remain. [Corrections to earlier claims](docs/AUDIT_AND_MIGRATION.md) include a miscited topology paper, overstated guarantees and the incorrect assertion that an old release contained a 600 MB bundle.

## Useful commands

```bash
ouroboros objective 'loss * beta / (1 + step)' --loss 2 --beta 0.1 --step 1
ouroboros storage audit --root .
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001
# Add --apply only after inspecting the pruning preview.
ouroboros evaluate --checkpoint checkpoints/RUN_ID/generation_0001 --data examples/data/validation.json
python scripts/build_release.py
```

Evaluate a NumPy demo checkpoint on `examples/data/demo_validation.json`, not the arithmetic example above. Replace `RUN_ID`/`ACTUAL_ID` with real identifiers. The release builder needs a clean Git tree; `--allow-dirty` is only for explicit local previews.

## Storage, Git LFS and releases

Code, configuration, small datasets and research notes stay readable Git files. Only approved files under `artifacts/objects/` use LFS attributes. Checkpoints, virtual environments, caches, trial outputs and release archives are ignored by default.

**No historical 600 MB cache/checkpoint payload was found in the starting checkout; v0.1.0 had zero uploaded assets.** LFS patterns cannot recover deleted files. The [artifact inventory](artifacts/manifest.json) is empty until real files are registered. With Git LFS installed, normal clone ordinarily retrieves tracked objects; use `GIT_LFS_SKIP_SMUDGE=1` for pointers, then `git lfs pull` when ready. No LFS payload upload is claimed here.

The [v0.3.0 pre-release](https://github.com/Mr-Nobody-Anonymous/ouroboros/releases/tag/v0.3.0) distributes core source and a checksum, **not model weights or upstream trees**. Obtain it through the release page or GitHub CLI:

```bash
gh release download v0.3.0 --repo Mr-Nobody-Anonymous/ouroboros \
  --pattern 'ouroboros-0.3.0-core.tar.gz*'
sha256sum -c ouroboros-0.3.0-core.tar.gz.sha256
```

[Storage/LFS guidance](docs/STORAGE_AND_RELEASES.md) covers registration, retrieval, limits, safe retention, deterministic bundles and what is not uploaded. v0.1.0/v0.2.0 remain unchanged. No automatic system-cache or unrelated-file cleanup occurs.

## Development and boundaries

```bash
python -m pytest -q
python -m unittest discover -s tests -v  # compatibility smoke tests; pytest runs the full suite
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
```

See [testing](docs/TESTING.md), [contributing](CONTRIBUTING.md), [agent instructions](AGENTS.md) and [security](SECURITY.md). Cooperative step/time budgets are not a hardened execution boundary. Never run generated Python or an upstream autonomous agent with host secrets, writable evaluation data or the Docker socket.

Original Ouroboros code is licensed under [MIT](LICENSE). Third-party sources/datasets retain their licenses.
