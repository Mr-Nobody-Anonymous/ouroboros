# Ouroboros 0.2.0 — measured local adaptation (pre-release)

This is an experimental branch release, not a claim of autonomous unlimited self-improvement or a pretrained-model benchmark win.

## Main changes

- Replace fabricated gradients/scores with real NumPy LoRA classifier optimization and an optional local Transformers/PEFT causal-LM SFT backend.
- Require measured improvement and original/latest-model retention; restore actual parameters and RNG on rejection, errors and cancellation.
- Save real tensor checkpoints with hashes, run ownership, data fingerprints, predictions, lineage and actual-byte Pareto retention.
- Add verified replay/proposal boundaries, reversible bounded tensor edits, numerical merge primitives and explicit architecture-copy experiments.
- Replace fake upstream success with explicit EasyEdit/mergekit interfaces. Generated source remains a review patch, never automatic execution.
- Repair submodule setup and lock twelve sources: seven pinned submodules plus five preserved, content-hashed snapshots. Five relevant repositories were newly cloned; not all sources are runnable integrations.
- Rewrite developer/agent setup, architecture, migration, integration, security, testing and storage documentation.
- Catalogue 33 research references, including 11 first released August–November 2026. Correct the MIT Astra/Qwen attribution, the unrelated topology arXiv citation, and earlier no-forgetting guarantees. Source-reported gains are not local results.

## Local verification

Python 3.11: **125 tests passed**, with three expected upstream PEFT Conv1D warnings. A separate no-ML environment passed **119 core tests** (six ML tests deselected); unittest discovery passed 14 compatibility checks. Ruff, documentation/catalogue checks, 12/12 source verification, configuration validation and source/wheel builds passed locally. Tiny random models are generated offline for CPU integration tests; pretrained LLMs are not downloaded.

Remote CI results must be checked separately. GPU/QLoRA, full MATH-500/GSM8K/IFEval runs, Astra reproduction, learned outer-loop RL, and real EasyEdit/mergekit experiments were **not** performed.

## Distribution and missing historical artifacts

The deterministic `ouroboros-0.2.0-core.tar.gz` builder creates an internal per-file hash manifest and a separate SHA-256 checksum. Its scope is original core source, documentation, tests, examples and upstream references; **no external source trees, pretrained weights, caches or virtual environment**. Obtain full development sources through Git clone and pinned submodule initialization. The Python source distribution and wheel are additional local build outputs, not a PyPI publication.

The historical 600+ MB payload was absent from the starting checkout and the old v0.1.0 release had no uploaded assets. It has **not** been recovered. The artifact manifest starts empty. LFS tracking/configuration scripts are prepared, but local Git LFS installation failed with an EOF download error; no LFS upload is claimed. Only files actually listed as attached assets on GitHub have been uploaded. The historical v0.1.0 release is not overwritten.

See the [README](../README.md), [testing evidence and limits](TESTING.md), [migration audit](AUDIT_AND_MIGRATION.md), and [storage/retrieval guide](STORAGE_AND_RELEASES.md).
