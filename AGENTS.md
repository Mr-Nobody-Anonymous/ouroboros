# Agent guide

This is Ouroboros, an experimental local model-adaptation toolkit. Read README.md's capability table and docs/AUDIT_AND_MIGRATION.md before making claims based on old class names or historical releases.

## Work safely

- Stay on the user's assigned working branch. Do not create hidden experiment branches, auto-merge generated code or rewrite published releases. Explain measured work and limitations before publishing it.
- Never fabricate benchmark gains, provenance, restored files, bridge execution or successful uploads.
- Core commands are offline-first. Cloning upstream source is not installation or permission to run autonomous upstream agents.
- Generated Python is a proposal, not trusted application code. Read SECURITY.md before introducing execution.
- Keep train/replay, validation and optional final-test data separate. Proposers must not receive held-out labels or access their files. Normalized deduplication is not semantic decontamination.
- Do not delete source trees, models, datasets, arbitrary checkpoint folders or system caches as a shortcut. Managed pruning must remain run-scoped and checksum verified.

## Setup / checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q -m 'not ml'
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
ouroboros run --config examples/local_sft.json --check-only
ouroboros weights trial --config examples/weight_trial.json --check-only
ouroboros demo --generations 3
```

Core dependencies include NumPy, PyYAML and safetensors, **not Torch**. `pip install -e '.[train]' -c constraints/ml-tested.txt` enables CPU Torch/direct-edit and tiny random local-LM tests without pretrained downloads. No GPU is assumed. Run the full `pytest` suite in that environment; mark new Torch-dependent tests `ml`. Check-only validates config/data, not actual model availability or target shape.

## Measured weight-edit contract

- Read docs/WEIGHT_EDITING.md before using the direct-edit API. Inspect metadata/prefix values without loading a model with `ouroboros weights inspect`.
- `run_weight_edit_trial` always restores its selected live parameter, including accepted trials, errors and cancellation. Only accepted candidates export a patch. Exit 0 can mean a measured rejection; inspect `passed`, `reason` and `model_restored`.
- `apply_weight_patch` is a separate explicit mutation; `reverse=True` requires the exact patched tensor. Hashes bind the target tensor, **not all model weights**, author identity or licensing rights.
- Same-process prediction/proposal/verifier callbacks are trusted. Trial rollback does not isolate other tensors/files; dense snapshots/sorts/hash copies can require large RAM. Use external quotas and isolation.
- Do not apply base edits inside PEFT SFT and assume adapter-only snapshots/checkpoints own them. Keep interventions separate unless their backend rollback/export contract is implemented and tested.
- `weights demo` is a **2×2 Torch classifier**. The training demo is a tiny NumPy classifier. Neither is an LLM/math benchmark or Astra reproduction.

## Map

- ouroboros/backends/: NumPy reference classifier, shared frozen LocalCausalLM loader and optional PEFT SFT backend.
- ouroboros/pipeline/self_loop.py: measured control flow, owned-state rollback, stage evidence and once-only post-selection final test.
- ouroboros/data.py: schema, provenance, replay and split boundaries.
- ouroboros/tensor_inspection.py: bounded read-only safetensors metadata/numeric prefixes without Torch.
- ouroboros/core/weight_trials.py: measured trials and data-only sparse patch verification/application/reversal.
- Other ouroboros/core/ modules: experimental arithmetic/architecture/proposal/evaluation/storage primitives.
- ouroboros/integrations/: explicit installed upstream calls, never fake fallbacks.
- external/sources.lock.json: seven pinned submodules and five preserved snapshots, with integration levels/licenses.
- papers/research.json: curated source-linked metadata; regenerate PAPERS_INDEX.md with scripts/update_paper_index.py.
- artifacts/manifest.json: starts empty; absent historical payloads are not recoverable from LFS patterns.

Keep new upstream experiments in separate environments, preserve licenses and state what was and was not run. Unit tests and fixtures do not establish pretrained-model capability gains. Use the release tag or active branch for installation while the current pull request is unmerged; do not claim the old default branch already contains this implementation.
