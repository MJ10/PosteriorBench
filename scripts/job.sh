#!/bin/bash
# PosteriorBench job for `cluv submit`: runs `uv run <command>` in the project synced on the cluster.
#
#   cluv submit mila -- python scripts/benchmark.py --sampler instruction
#   cluv submit rorqual --array=0-6 -- python scripts/benchmark.py --sampler original
#   cluv submit mila -- posteriorbench evaluate --task wc_1853/goblin_mention --sampler instruction
#
# The GPU and the account come from [tool.cluv] in pyproject.toml; flags given to `cluv submit`
# (e.g. --time=8:00:00) override the defaults below.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
set -eo pipefail

# This job's folder under results_path, where cluv also writes its Slurm log: `cluv sync` copies
# it back to ./logs on your machine. scripts/benchmark.py writes its results here.
export PB_RUN_DIR="${PB_RESULTS_PATH:?Submit this script with cluv submit}/${CLUV_CLUSTER}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$PB_RUN_DIR"

head=$(git rev-parse HEAD 2>/dev/null || true)
if [[ -n "$GIT_COMMIT" && "$head" != "$GIT_COMMIT" ]]; then
    echo "WARNING: submitted at ${GIT_COMMIT:0:8} but the project is now at ${head:0:8} (moved by a later cluv sync/submit)." >&2
fi
echo "Job $SLURM_JOB_ID on $CLUV_CLUSTER ($(hostname)), commit ${GIT_COMMIT:0:8}, results in $PB_RUN_DIR"
nvidia-smi -L 2>/dev/null || echo "No GPU visible"
echo "Running: uv run $*"

# --no-sync: cluv sync prepared the env on the login node. Array tasks running `uv run` at once
# could otherwise re-sync the shared .venv while the others import from it.
srun uv run --no-sync "$@"
