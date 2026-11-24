# Running PosteriorBench on the Mila and DRAC clusters

This fork runs the benchmark and the paper's methods on the Mila cluster and on DRAC/PAICE
clusters (Rorqual, Fir, Nibi, Narval, Trillium, TamIA, Killarney, Vulcan) through
[cluv](https://github.com/mila-iqia/cluv). cluv clones this repository from GitHub on each cluster,
runs `uv sync` there, submits Slurm jobs and copies results back.

| File | What it is |
|---|---|
| `[tool.cluv]` in [`pyproject.toml`](pyproject.toml) | clusters, accounts, GPU request, job environment |
| [`scripts/job.sh`](scripts/job.sh) | default job: `uv run <command>` in the project (1 GPU, 4 CPUs, 32 GB, 3 h) |
| [`scripts/methods_job.sh`](scripts/methods_job.sh) | job for `methods/`: `uv run <command>` in `methods/` (1 GPU, 8 CPUs, 64 GB, 12 h) |
| [`scripts/benchmark.py`](scripts/benchmark.py) | the 21 tasks × 3 replicates for one sampler: resumable, split over a job array and over GPUs |
| [`scripts/setup_cluster.sh`](scripts/setup_cluster.sh) | one-time setup per cluster, on a login node: `methods/` environment and Hub downloads |
| [`scripts/prefetch.py`](scripts/prefetch.py) | downloads the model and corpus hits into `$HF_HOME` |

## Setup

Once, on your machine:

```bash
uv tool install cluster-uv      # the cluv command
git clone https://github.com/MJ10/PosteriorBench ~/Projects/posteriorbench
cd ~/Projects/posteriorbench && git checkout cluv && uv sync
```

The clusters are the `Host` entries of `~/.ssh/config` (with `ControlMaster`; `mila init` from
[milatools](https://github.com/mila-iqia/milatools) writes them). Then, each session:

```bash
cluv login          # one 2FA prompt per cluster; later commands reuse the connections
cluv status         # connected clusters, free GPUs, your jobs
```

Once per cluster, sync the project, then install `methods/` and download the model (1.8 GB) and the
corpus hits of the 108 scored pairs (3.4 GB) into `$SCRATCH/.cache/huggingface`, on the login
node, since most compute nodes have no internet:

```bash
cluv sync rorqual
cluv run rorqual bash scripts/setup_cluster.sh
```

Repeat for every cluster you use. On Mila, Killarney and Trillium the project lives on `$SCRATCH`
(Mila: your 100 GB `$HOME` quota is full; Killarney and Trillium: jobs can't start from `/home`),
which `cluv run` does not follow. Use ssh for those. On Mila, `UV_LINK_MODE=copy` keeps uv from hard-linking
the venv's files into the cache: on Mila's BeeGFS, jobs importing hard-linked files at once failed with
`FileNotFoundError`. Elsewhere hard links are fine, and they save space in a small `$HOME` (48 GiB on Fir):

```bash
cluv sync mila
ssh mila 'UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy bash -l $SCRATCH/repos/PosteriorBench/scripts/setup_cluster.sh'
ssh killarney 'bash -l $SCRATCH/repos/PosteriorBench/scripts/setup_cluster.sh'
```

Jobs run `uv run --no-sync`: they use the environments that `cluv sync` (root project) and
`setup_cluster.sh` (`methods/`) installed, and never write to them, so that the tasks of a job array can't
re-install packages under each other. If you change the dependencies of either project, run the
setup again (on Mila, with the command above: the uv cache in your full `$HOME` can't take new packages).
Don't run `cluv sync` or `cluv submit` while jobs are still importing on a cluster where the dependencies just changed.

`$SCRATCH` is purged on DRAC clusters (files unused for 60 days). If a job fails with
`OfflineModeIsEnabled` or `LocalEntryNotFoundError`, the cache is gone: run the setup again.

## Running the benchmark

The full benchmark for one sampler (63 evaluations, about 1.5 h on an H100) in one job:

```bash
cluv submit mila -- python scripts/benchmark.py --sampler instruction
```

`cluv submit` refuses a dirty working tree (commit first, or pass `--autocommit`), pushes, syncs
the cluster to your commit, and submits `scripts/job.sh`. Everything after `--` runs under
`uv run` in the project on the cluster.

To spread the work, use a job array: each array task runs every n-th evaluation. To race clusters,
use `first`. To fit on DRAC MIG slices, which usually start sooner, use `--vram`:

```bash
cluv submit rorqual --array=0-6 -- python scripts/benchmark.py --sampler instruction   # 7 jobs × 9 evaluations
cluv submit first -- python scripts/benchmark.py --sampler original                    # first cluster to start wins
cluv submit fir --vram=20GB -- python scripts/benchmark.py --sampler instruction       # races MIG slices and full GPUs
```

Your own sampler, from a factory `function(task=, model=, benchmark=) -> (Sampler, Ledger)`
([`examples/02_custom_sampler`](examples/02_custom_sampler)):

```bash
cluv submit mila -- python scripts/benchmark.py --sampler custom \
    --factory examples/02_custom_sampler/custom_sampler.py:build_sampler --method my-method
```

Other options: `--tasks scored` (the 108 pairs) or a comma-separated list, `--replicates 0,1,2`,
`--setting K=16`, `--workers`, `--batch-size`, `--dtype` (`python scripts/benchmark.py --help`).
On TamIA a job gets a whole node of 4 H100s, and `benchmark.py` runs one worker per GPU.

Results are written to `$SCRATCH/logs/posteriorbench/<cluster>_<job id>/benchmark/<method>/<setting>/<task>/rep<r>.json`,
next to the job's Slurm log. An evaluation whose file exists is skipped, so a requeued job, or
the same command submitted again, continues where it stopped. Within a job, the output folder is
per job id, so a new submission starts from scratch. To fill in the missing evaluations of an
earlier run, pass that run's folder: `--out $SCRATCH/logs/posteriorbench/mila_1234/benchmark`.

Follow the jobs and fetch the results:

```bash
cluv status
cluv sync                  # copies new results to ./logs (a symlink to ~/scratch/logs/posteriorbench)
uv run posteriorbench aggregate $(find logs/ -name 'rep*.json') --expect-tasks benchmark --expect-replicates 0,1,2
```

Single CLI commands work too: `cluv submit mila -- posteriorbench evaluate --task wc_1853/goblin_mention --sampler instruction`.

## Running the paper's methods

`methods/` is a uv project of its own: submit [`scripts/methods_job.sh`](scripts/methods_job.sh),
which runs the command in `methods/`. Pass `--time` for the longer families
([methods/README.md](methods/README.md#the-families) lists GPU-hours and disk):

```bash
cluv submit mila scripts/methods_job.sh -- posteriorbench-methods sft run --task wc_0346/em_dash --set budgets=131072
cluv submit rorqual scripts/methods_job.sh --time=20:00:00 -- posteriorbench-methods gfn run --task wc_0346/em_dash --keep-checkpoints all
```

Training methods write tens of GB of checkpoints, so a run's output root is
`$SCRATCH/posteriorbench/runs/<cluster>_<job id>`, outside `results_path`, which keeps `cluv sync` from
copying the checkpoints to your laptop. Pass `--out DIR` (absolute, or relative to `methods/`) to
continue an earlier training. At the end of the job, each record's small files (`result*.json`,
`construction.json`, `config.json`, `summary.jsonl`) are copied to the job's results folder, so
`cluv sync` brings them back. A method run is single-GPU, so on TamIA's 4-GPU nodes three GPUs
sit idle. Prefer the other clusters for `methods/`.

## Clusters and accounts

| Cluster | Account(s) | Notes |
|---|---|---|
| mila | (none) | compute nodes online; default `long` partition; no V100s (the locked torch has no kernels for them); project on `$SCRATCH` |
| rorqual, fir, nibi | `rrg-bengioy-ad`, `def-bengioy` | one job per account, the first to start is kept; fir and nibi compute nodes are online |
| narval | `def-bengioy` | |
| trillium-gpu | `rrg-bengioy-ad` | project on `$SCRATCH/repos/PosteriorBench` |
| tamia | `aip-bengioy` | whole nodes only: `--gpus=h100:4 --mem=0` |
| killarney, vulcan | `aip-bengioy` | killarney: project on `$SCRATCH/repos/PosteriorBench` |

To stop using a cluster, delete its table in `pyproject.toml` or run `cluv disable <cluster>`.
To change resources for one run, add sbatch flags before `--`, e.g. `cluv submit mila --gpus=a100l:1 --time=1:00:00 -- …`.

Jobs run with `UV_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline`, except on Mila, Fir and Nibi,
whose compute nodes have internet. A job runs from the synced project folder, and a later `cluv sync`
or `cluv submit` can check out another commit while it waits in the queue. The job's log then
starts with a `WARNING: submitted at … but the project is now at …` line, and each result records
the `git_commit` it was submitted at.

## Keeping up with upstream

`origin` is the fork (MJ10/PosteriorBench); `upstream` is TransluceAI/PosteriorBench:

```bash
git fetch upstream && git merge upstream/main
```
