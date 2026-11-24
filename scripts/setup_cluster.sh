#!/bin/bash
# Prepares a cluster for PosteriorBench jobs. Run it on a login node (they have internet access),
# once per cluster, after the first `cluv sync`:
#
#   cluv run rorqual bash scripts/setup_cluster.sh
#
# Where the project lives on $SCRATCH (Mila, Killarney, Trillium: CLUSTERS.md), which `cluv run`
# doesn't follow:
#
#   ssh mila 'UV_CACHE_DIR=$SCRATCH/.cache/uv bash -l $SCRATCH/repos/PosteriorBench/scripts/setup_cluster.sh'
#   ssh killarney 'bash -l $SCRATCH/repos/PosteriorBench/scripts/setup_cluster.sh'
#
# It installs the environment of methods/ (cluv sync only installs the root project's), and
# downloads the model and the corpus hits into $HF_HOME, so that jobs can run with
# HF_HUB_OFFLINE=1. Extra arguments go to scripts/prefetch.py (e.g. --hits benchmark).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep in sync with HF_HOME in [tool.cluv.env] of pyproject.toml.
export HF_HOME="${HF_HOME:-${SCRATCH:?SCRATCH is not set}/.cache/huggingface}"
unset HF_HUB_OFFLINE UV_OFFLINE

echo "== uv sync in $PWD and $PWD/methods"
uv sync --quiet
uv sync --quiet --directory methods

echo "== Downloading into $HF_HOME"
uv run python scripts/prefetch.py "$@"
echo "== Done: jobs on this cluster can now run offline."
