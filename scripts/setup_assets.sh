#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required. Install git-lfs, then rerun this script." >&2
    exit 1
fi
git submodule sync -- third_party/brainco-description
git submodule update --init -- third_party/brainco-description
git -C third_party/brainco-description lfs install --local
git -C third_party/brainco-description lfs pull
