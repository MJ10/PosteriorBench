#!/bin/bash
# Job for the paper's methods (methods/, a uv project of its own) for `cluv submit`:
# runs `uv run <command>` in methods/.
#
#   cluv submit mila scripts/methods_job.sh -- posteriorbench-methods sft run --task wc_0346/em_dash --set budgets=131072
#   cluv submit rorqual scripts/methods_job.sh --time=16:00:00 -- posteriorbench-methods gfn run --task wc_0346/em_dash
#
# methods/.venv must exist on the cluster: run scripts/setup_cluster.sh there once (CLUSTERS.md).
#
# Runs write checkpoints (up to 165 GB for DPG), so the output root is not under results_path,
# which `cluv sync` copies back to your machine. Without --out, it is
# $PB_WORK_PATH/<cluster>_<job id>; with --out DIR, DIR (relative to methods/), e.g. to continue a
# training of an earlier run. Either way, the small files of each record (result*.json,
# construction.json, config.json, summary.jsonl) are copied to this job's results folder at the end.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -eo pipefail

job_name="${CLUV_CLUSTER}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
export PB_RUN_DIR="${PB_RESULTS_PATH:?Submit this script with cluv submit}/$job_name"
mkdir -p "$PB_RUN_DIR"

args=("$@")
out=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --out) out="${args[i + 1]}" ;;
        --out=*) out="${args[i]#--out=}" ;;
    esac
done
if [[ -z "$out" ]]; then
    out="${PB_WORK_PATH:?Submit this script with cluv submit}/$job_name"
    args+=(--out "$out")
fi
[[ "$out" = /* ]] || out="$PWD/methods/$out"

head=$(git rev-parse HEAD 2>/dev/null || true)
if [[ -n "$GIT_COMMIT" && "$head" != "$GIT_COMMIT" ]]; then
    echo "WARNING: submitted at ${GIT_COMMIT:0:8} but the project is now at ${head:0:8} (moved by a later cluv sync/submit)." >&2
fi
echo "Job $SLURM_JOB_ID on $CLUV_CLUSTER ($(hostname)), commit ${GIT_COMMIT:0:8}"
echo "Output root: $out; records copied to $PB_RUN_DIR/methods"
nvidia-smi -L 2>/dev/null || echo "No GPU visible"
echo "Running: uv run ${args[*]} (in methods/)"

status=0
srun uv --directory=methods run "${args[@]}" || status=$?

if [[ -d "$out" ]]; then
    rsync -a --prune-empty-dirs \
        --exclude 'training_runs/' --exclude 'cache/' --exclude 'draws/' \
        --include '*/' --include 'result*.json' --include 'construction.json' \
        --include 'config.json' --include 'summary.jsonl' --exclude '*' \
        "$out/" "$PB_RUN_DIR/methods/"
fi
exit $status
