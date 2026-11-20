# Contributing to Ouroboros

Use a working branch and review changes before publishing. In an Arena session, keep all work on the session's assigned branch; do not create hidden experiment branches. Source mutation experiments should export patches rather than switch branches or merge themselves.

## Setup and checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
```

For a no-Torch environment run `python -m pytest -q -m 'not ml'`. Core safetensors inspection/patch-format tests must work there. Optional Torch/direct-edit and local-LM tests need the `train` extra, with the tested versions in `constraints/ml-tested.txt`. Tiny causal LMs/tokenizers are generated locally without pretrained downloads; mark all Torch-dependent tests `ml` so minimal installations remain lightweight.

`python scripts/install.py --profile train --dev` is the convenience environment setup (includes optional MCP/GGUF packages). MCP tests use the `mcp` marker and the real official SDK/stdio protocol. `init` must work from an installed wheel with no source tree. Ordinary pytest must never download pretrained data; the separate `scripts/run_pretrained_smoke.py --fetch` / GitHub job is explicitly network-enabled and checks mechanics, not a required accuracy gain. Preserve the read-only MCP default and explicit config allowlist; do not add generic execution/deletion tools.

Check both `ouroboros run --config examples/local_sft.json --check-only` and `ouroboros weights trial --config examples/weight_trial.json --check-only`. A schema check must not imply that a real model exists or that a direction matches its target.

## Standards

- Report measured predictions/losses, never invented gains or a successful bridge call that did no work.
- Define backend/update semantics, input shapes, rollback behavior and finite-value checks.
- Preserve exact data split separation and provenance; do not silently train on evaluation targets.
- Add regression tests for rollback, persistence and failure paths, not only output shapes.
- Treat replay and norm bounds as mitigations, not guarantees of no forgetting or collapse.
- Do not commit weights, caches, virtual environments or release archives. Register intentional artifacts and review their licenses before LFS uploads.
- Do not vendor more repository trees indiscriminately. Prefer pinned source-only submodules, document actual integration level, and retain upstream licenses. Preserve or explicitly migrate existing vendored modifications.
- Keep original code changes in `ouroboros/`/adapters where possible instead of editing opaque third-party snapshots. The source inventory must flag intentional upstream modifications, not hide them.
- Cite primary papers with exact titles/IDs/dates, distinguish blog/preprint/peer-reviewed status, and record what was actually reviewed. A research motivation is not an implemented algorithm or a reproduced benchmark.

After changing `papers/research.json`, run `python scripts/update_paper_index.py` and update the relevant digest. Source lock entries for vendored snapshots intentionally have `revision: null` where the starting commit cannot be established. Do not invent provenance.

Read [SECURITY.md](SECURITY.md) before proposing source execution or cleanup behavior.
