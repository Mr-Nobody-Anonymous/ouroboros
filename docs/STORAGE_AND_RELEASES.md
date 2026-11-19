# Storage, artifacts, Git LFS and release bundles

## What was actually present?

The starting checkout at `97181d3` contains source trees and small research documents, not the previously claimed 600+ MB of purged benchmark caches or model checkpoints. No LFS pointers were found. The existing `v0.1.0` release's uploaded asset list was empty when checked on November 19, 2026. A release tag/source archive is not evidence that deleted weights were uploaded.

We cannot recover absent data by adding LFS patterns. [`artifacts/manifest.json`](../artifacts/manifest.json) starts with **zero registered files** and records the historical claim as unrecovered, with no invented per-file sizes. Any newly installed dependencies/caches used for testing are local environment files, not recovered benchmark data or release assets.

## Keep source and artifacts separate

- Ordinary Git: code, small JSON/JSONL examples, configuration, research summaries, source locks and artifact manifests.
- Ignored local output: checkpoints, adapters, outputs, virtual environments, caches, release archives.
- Intentional LFS objects: only reviewed files explicitly placed under `artifacts/objects/` and force-staged.

The former blanket `*.jsonl` LFS rule was removed so agents can read small source datasets directly. No history migration is performed. Existing third-party snapshots were preserved rather than indiscriminately deleted. New source-only submodules avoid copying more upstream histories/data into the main repository.

## Correct clone behavior

With Git LFS installed, **a normal clone ordinarily downloads the actual tracked LFS objects during checkout**. To get pointers without that automatic transfer:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Mr-Nobody-Anonymous/ouroboros.git
cd ouroboros
git lfs install --local
git lfs ls-files
# Fetch only when needed, or scope with --include:
git lfs pull --include='artifacts/objects/**'
python scripts/verify_artifacts.py
```

Git submodules and Git LFS solve different problems. Submodules point to another Git repository/commit; LFS pointers identify large content objects. Run `git submodule update --init --depth 1` separately when you want upstream source. A GitHub source ZIP generally does not populate submodules, and release/source archive handling of LFS content depends on repository/archive settings. Do not assume one archive includes all external trees and all LFS blobs.

## Register a file that really exists

Install Git LFS through your OS package manager, then run `sh scripts/configure_lfs.sh`. The script configures local hooks/tracking only; it never uploads anything or rewrites history. A binary download attempt in the Arena sandbox failed with an `EOF` error while fetching a release asset, so local Git LFS binary installation was **not** completed here. The exact cause of that download failure was not established. The tracked attributes and scripts remain usable on a normally provisioned machine.

```bash
python scripts/register_artifact.py /path/to/an/existing/adapter.safetensors \
  --license 'YOUR_ACTUAL_LICENSE' --provenance 'local run ID / verified source revision'
python scripts/verify_artifacts.py
git check-attr filter -- artifacts/objects/ACTUAL_HASH/adapter.safetensors
# Review the real path printed by registration, license, size and storage quota first.
git add .gitattributes artifacts/manifest.json
git add -f artifacts/objects/ACTUAL_HASH/adapter.safetensors
git lfs ls-files
# Push only your actual working branch after review; Git LFS hooks upload its referenced objects.
```

`ACTUAL_HASH` is a placeholder, not an artifact already stored in this repository. Registration copies a real file, records SHA-256/bytes/license/provenance and leaves it ignored/unstaged until you explicitly approve it. The verify script checks local content, **not remote availability**. Confirm a fresh-clone retrieval before promising distribution. Do not blindly upload all caches: they can be redundant, proprietary, private or nondeterministic.

Git LFS has account-specific storage/bandwidth billing and per-object limits. It is not unlimited free storage and does not authorize redistribution. Check the current host/account limits before uploading large checkpoints; we do not invent quota availability. [GitHub LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) describes the current service.

## Safe checkpoint retention

```bash
ouroboros storage audit --root .
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001
# Only after reviewing the preview:
ouroboros storage prune --root checkpoints/RUN_ID --run-id RUN_ID --protect generation_0001 --apply
```

Replace names with the actual run ID/current checkpoint from `run.json`. Accuracy is maximized and actual bytes minimized. A frontier cap is applied, but protected baseline/current points may exceed the nominal cap. Only correctly named same-run hash-verified directories are considered; foreign, unknown, corrupt and symlinked paths are not deleted. The controller prunes its own run after persisting each decision. No system/model/dataset-cache deletion is performed by audit or ordinary CLI commands. Keep exclusive ownership of run directories; this is not a defense against a hostile concurrent filesystem writer.

## Deterministic release files

```bash
python scripts/build_release.py
# For a LOCAL preview before committing, explicitly add --allow-dirty.
```

The script produces `artifacts/releases/ouroboros-0.2.0-core.tar.gz` plus a SHA-256 file. It bundles original core source, docs, tests, examples and reference manifests; **no external source trees, weights, datasets/caches, credentials or virtual environment**. `RELEASE_CONTENTS.json` records file hashes, commit, scope and whether a dirty preview was explicitly requested. Tar/gzip metadata is normalized for deterministic bytes from identical inputs.

This core archive is not falsely called a recovered 600 MB “complete model.” To obtain the complete development source tree, Git clone the repository and initialize its pinned submodules; the preserved vendored sources arrive with the clone. Obtain large models/data separately under their original licenses.

Publication is a separate reviewed GitHub operation. Use a pre-release/draft for branch work, specify its actual commit, attach the generated archive/checksum if network permissions permit, and do not overwrite the historical v0.1.0 release or claim an asset uploaded merely because a release tag exists. The local release builder never uploads automatically.

### GitHub-side publication

The [release workflow](../.github/workflows/release.yml) builds the same deterministic core archive when an authorized maintainer **publishes** a release, including a pre-release published from a draft. It checks out that tag, requires its name to equal `v` plus the package version, builds from a clean tree, checks the SHA-256, then attaches the archive/checksum using the job's scoped GitHub token. It does not install training dependencies, run external agents, include upstream trees or upload model weights. Existing same-name assets are not silently overwritten. A manual retry input is available once the workflow is on the default branch.

Only the upload job receives `contents: write`; pull-request/core testing remains read-only. Review the tagged source and publication permissions before releasing. No credentials are committed or placed in generated candidates. During this refresh a direct sandbox release upload failed with `EOF` from the upload endpoint and left an empty draft; the GitHub-side workflow is the publication path, not proof of success. Check the release's actual asset list and workflow conclusion before promising a download.
