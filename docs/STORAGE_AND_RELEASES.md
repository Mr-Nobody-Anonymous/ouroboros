# Storage, artifacts, Git LFS and release bundles

## What was actually present?

The starting checkout at `97181d3` contains source trees and small research documents, not the previously claimed 600+ MB of purged benchmark caches or model checkpoints. No LFS pointers were found. The existing `v0.1.0` release's uploaded asset list was empty when checked on November 19, 2026. A release tag/source archive is not evidence that deleted weights were uploaded.

We cannot recover absent data by adding LFS patterns. [`artifacts/manifest.json`](../artifacts/manifest.json) starts with **zero registered files** and records the historical claim as unrecovered, with no invented per-file sizes. Dependencies/caches used for testing are local environment files, not recovered benchmark data or release assets.

## Keep source and artifacts separate

- Ordinary Git: code, small JSON/JSONL examples, configuration, research summaries, source locks and artifact manifests.
- Ignored local output: checkpoints, adapters, trial reports/patches, virtual environments, caches and release archives.
- Intentional LFS objects: only reviewed files explicitly placed under `artifacts/objects/` and force-staged.

The former blanket `*.jsonl` LFS rule was removed so agents can read small datasets directly. No history migration is performed. Third-party snapshots were preserved rather than deleted. New source-only submodules avoid copying more upstream histories/data into the main repository.

## Correct clone behavior

With Git LFS installed, **a normal clone ordinarily downloads actual tracked LFS objects during checkout**. To get pointers without that automatic transfer:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch v0.3.0 --single-branch https://github.com/Mr-Nobody-Anonymous/ouroboros.git
cd ouroboros
git lfs install --local
git lfs ls-files
# Fetch only when needed, or scope with --include:
git lfs pull --include='artifacts/objects/**'
python scripts/verify_artifacts.py
```

Git submodules and Git LFS solve different problems. Submodules pin another repository/commit; LFS pointers identify large objects. Run `git submodule update --init --depth 1` separately when you want upstream source. A GitHub source ZIP generally does not populate submodules; release/source archive handling of LFS content depends on repository/archive settings. Do not assume an archive includes external trees or LFS blobs. Use a reviewed tag or the active branch while the implementation PR is unmerged; the historical default branch is not silently replaced.

## Register a file that really exists

Install Git LFS through your OS package manager, then run `sh scripts/configure_lfs.sh`. The script configures local hooks/tracking only; it never uploads or rewrites history. A binary download attempt in the Arena sandbox failed with `EOF` while fetching a release asset, so local Git LFS binary installation was **not** completed here. The exact cause was not established. Attributes/scripts remain usable on a normally provisioned machine.

```bash
python scripts/register_artifact.py /path/to/an/existing/adapter.safetensors \
  --license 'YOUR_ACTUAL_LICENSE' --provenance 'local run ID / verified source revision'
python scripts/verify_artifacts.py
git check-attr filter -- artifacts/objects/ACTUAL_HASH/adapter.safetensors
# Review the actual path, license, size and storage quota first.
git add .gitattributes artifacts/manifest.json
git add -f artifacts/objects/ACTUAL_HASH/adapter.safetensors
git lfs ls-files
# Push only the actual working branch after review; LFS hooks upload its referenced objects.
```

`ACTUAL_HASH` is a placeholder, not an artifact already stored here. Registration copies a real file and records SHA-256/bytes/license/provenance; it leaves it ignored/unstaged until explicit approval. Verification checks local content, **not remote availability**. Confirm fresh-clone retrieval before promising distribution. Do not blindly upload caches: they may be redundant, proprietary, private or nondeterministic.

Git LFS has account-specific storage/bandwidth billing and per-object limits. It is not unlimited free storage and does not authorize redistribution. Check current host/account limits before large uploads; quota availability is not assumed. [GitHub LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) describes the service.

## Weight-trial outputs are not adapter checkpoints

The [weight-edit guide](WEIGHT_EDITING.md) describes `trial.json` and accepted-only `patch/patch.json` plus `patch/delta.safetensors`. A trial restores the live model, including acceptance; the patch is a separately applied proposal, not a saved full model. Its safetensors payload is capped at 64 MiB (or a smaller configured cap), while the JSON reader has a separate 1 MiB cap. Verification requires the exact two-file patch inventory. Reports may contain predictions and local paths—review before sharing.

Keep both patch files together. For an approved, licensed distribution, package just those files and optionally register the real archive rather than uploading all output directories:

```bash
# Replace ACTUAL_ID with the accepted trial ID; a rejected trial has no patch.
tar -C outputs/weight-trials/ACTUAL_ID -czf outputs/approved-weight-patch.tar.gz \
  patch/patch.json patch/delta.safetensors
# Review rights, provenance and contents, then optionally use register_artifact.py on this real file.
```

A tiny patch may not warrant LFS. Hashes do not authenticate its author, validate license assertions, bind every other tensor or prove capability gains. Dense trial snapshots/sorting/hashing still need RAM proportional to the selected matrix. **PEFT adapter-only checkpoints cannot preserve arbitrary base-weight edits.** Trial outputs are not eligible for managed checkpoint pruning, and no patch from a test fixture is registered as a production artifact by default.

## Safe checkpoint retention

```bash
ouroboros storage audit --root .
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001
# Only after reviewing the preview:
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001 --apply
```

Replace names with actual IDs/current checkpoint from `run.json`. Accuracy is maximized and actual bytes minimized. A frontier cap applies, but protected baseline/current points can exceed it. Only correctly named same-run hash-verified directories are considered; foreign, unknown, corrupt and symlinked paths are not deleted. The controller prunes its own run after persisting each decision. No system/model/dataset-cache deletion is performed by audit or ordinary commands. Keep exclusive ownership of run directories; this is not protection against a hostile concurrent filesystem writer.

## Deterministic release files

```bash
python scripts/build_release.py
# For a LOCAL preview before committing, explicitly add --allow-dirty.
```

Version 0.3.0 produces `artifacts/releases/ouroboros-0.3.0-core.tar.gz` plus a SHA-256 file; versioning leaves the published 0.2.0 bundle unchanged. The archive contains original source, docs, tests, small examples and reference manifests, **not upstream trees, model weights, benchmark corpora, caches, credentials or virtual environments**. `RELEASE_CONTENTS.json` records file hashes, commit, scope and an explicit dirty-preview flag. Tar/gzip metadata is normalized for deterministic bytes from identical inputs.

This is not a recovered 600 MB “complete model.” Obtain complete development sources through Git clone and pinned submodule initialization; preserved vendored sources arrive with the clone. Obtain large models/data separately under their original licenses. The Python wheel/sdist are also source distributions, not evidence of a PyPI upload or full-model release.

Publication is separate and reviewed: use a pre-release/draft for branch work, target its actual commit and do not overwrite v0.1.0/v0.2.0. The builder never uploads automatically; a tag alone does not prove assets were attached.

### GitHub-side publication

The [release workflow](../.github/workflows/release.yml) builds the same deterministic core archive when an authorized maintainer **publishes** a release, including a pre-release published from a draft. It checks out the tag, requires its name to equal `v` plus the package version, builds from a clean tree, checks SHA-256, then attaches the archive/checksum using the job's scoped GitHub token. It does not install training dependencies, run external agents, include upstream trees or upload model weights. Existing same-name assets are not overwritten. A manual retry input is available once the workflow is on the default branch.

Only the upload job receives `contents: write`; pull-request testing remains read-only. Review tagged source and permissions before releasing. No credentials are committed or placed in generated candidates. During 0.2.0 publication a direct sandbox upload failed with `EOF`; the GitHub-side workflow subsequently attached the verified archive/checksum. For each new release, check its actual asset inventory and workflow conclusion before promising a download. A previous release's successful job does not validate a new one.
