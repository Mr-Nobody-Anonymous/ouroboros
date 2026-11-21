#!/usr/bin/env sh
# Install Git LFS using your OS package manager first. This script never migrates history or uploads files.
set -eu
cd "$(dirname "$0")/.."
if ! git lfs version >/dev/null 2>&1; then
    printf '%s\n' 'Git LFS is not installed. Install it for your OS, then rerun this script.' >&2
    exit 1
fi
git lfs install --local
git lfs track 'artifacts/objects/**'
git lfs ls-files
printf '%s\n' 'LFS configured locally. No files were recovered, staged, migrated, or uploaded.'
