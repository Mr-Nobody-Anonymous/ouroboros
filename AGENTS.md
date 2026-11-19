# Agent guide

This is Ouroboros, an experimental local model-adaptation toolkit. Read README.md's capability table and docs/AUDIT_AND_MIGRATION.md before making claims based on old class names or the historical release.

## Work safely

- Stay on the user's assigned working branch. Do not create hidden experiment branches, auto-merge generated code or rewrite old releases.
- Never fabricate benchmark gains, provenance, restored files, bridge execution or successful uploads.
- Core commands are offline-first. Cloning upstream source is not installation or permission to run autonomous upstream agents.
- Generated Python is a proposal, not trusted application code. Read SECURITY.md before introducing execution.
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
ouroboros demo --generations 3
```

`pip install -e '.[train]' -c constraints/ml-tested.txt` enables local CPU integration tests that generate a tiny model without pretrained downloads. No GPU is assumed. `ouroboros run --config examples/local_sft.json --check-only` validates example configuration/data without loading a model.

## Map

- ouroboros/backends/: real NumPy reference classifier and optional Transformers/PEFT local-LM backend.
- ouroboros/pipeline/self_loop.py: measured control flow, actual rollback, ownership and evidence.
- ouroboros/data.py: schema, provenance, replay and split boundaries.
- ouroboros/core/: experimental tensor/architecture/proposal/evaluation/storage primitives.
- ouroboros/integrations/: explicit installed upstream calls, never fake fallbacks.
- external/sources.lock.json: seven pinned submodules and five preserved snapshots, with integration levels and licenses.
- papers/research.json: curated source-linked metadata; regenerate PAPERS_INDEX.md with scripts/update_paper_index.py.
- artifacts/manifest.json: starts empty; absent historical payloads are not recoverable from LFS patterns.

Keep new upstream experiments in separate environments, preserve licenses, and state what was and was not run. Unit tests and the tiny demo are not evidence of real pretrained-model capability gains.
