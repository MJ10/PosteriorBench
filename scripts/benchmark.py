"""Run the benchmark (tasks x replicates) for one sampler, split over Slurm jobs and GPUs.

The README's loop over the 21 tasks and 3 replicates, made for cluster jobs:

- Each (task, replicate) is written to its own JSON file, `<out>/<method>/<setting>/<task>/rep<r>.json`,
  and skipped if that file exists, so a requeued or resubmitted job continues where it stopped.
- A job array splits the work: array task i of n runs every n-th evaluation (`--array=0-6` runs
  the 63 evaluations as 7 jobs of 9).
- Inside a job, one worker process per visible GPU (TamIA gives whole nodes of 4 H100s).

Submitted with cluv (scripts/job.sh sets PB_RUN_DIR, the job's results folder that `cluv sync`
copies back to ./logs):

    cluv submit mila -- python scripts/benchmark.py --sampler instruction
    cluv submit rorqual --array=0-6 -- python scripts/benchmark.py --sampler original
    cluv submit mila -- python scripts/benchmark.py --sampler custom \\
        --factory examples/02_custom_sampler/custom_sampler.py:build_sampler --method my-method

Then, on your machine: `uv run posteriorbench aggregate $(find logs -name 'rep*.json')
--expect-tasks benchmark --expect-replicates 0,1,2`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import importlib.util
import json
import multiprocessing
import os
import pathlib
import sys
import time
from typing import Any

import posteriorbench as pb
from posteriorbench import original_model, results

REFERENCE_METHODS = {
    "original": "original-model",
    "instruction": "instruction",
    "fallback": "fallback",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sampler", choices=["original", "instruction", "fallback", "custom"], required=True
    )
    parser.add_argument(
        "--factory",
        help="For --sampler custom: module:function or path/to/file.py:function, called as "
        "function(task=, model=, benchmark=), returning (Sampler, Ledger or LedgerTotals).",
    )
    parser.add_argument("--method", help="Method name recorded in the results.")
    parser.add_argument("--setting", help="Setting recorded in the results (e.g. K=16).")
    parser.add_argument(
        "--tasks",
        default="benchmark",
        help='"benchmark" (the 21 tasks), "scored" (the 108 scored pairs), or comma-separated ids.',
    )
    parser.add_argument("--replicates", default="0,1,2", help="Comma-separated replicate indices.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output root; may use $SCRATCH (default: $PB_RUN_DIR/benchmark in a cluv job, else "
        "runs/benchmark). Pass an earlier job's folder to fill in what it did not finish.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="i/n: run every n-th evaluation starting at the i-th (default: from the Slurm array).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes, one per GPU (default: the number of visible GPUs, at least 1).",
    )
    parser.add_argument("--dtype", choices=original_model.DTYPES, default="float32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--details", action="store_true", help="Keep draws and per-reference terms."
    )
    args = parser.parse_args(argv)
    if (args.sampler == "custom") != (args.factory is not None):
        parser.error("--factory is required with --sampler custom, and only then")
    return args


def method_name(args: argparse.Namespace) -> str:
    """The method label recorded in the results: --method, else the reference sampler's."""
    return str(args.method or REFERENCE_METHODS.get(args.sampler, args.sampler))


def array_shard() -> tuple[int, int]:
    """(index, count) of this job in its Slurm array, (0, 1) outside an array."""
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        return 0, 1
    index = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return index, count


def result_path(
    out: pathlib.Path, method: str, setting: str | None, task_id: str, rep: int
) -> pathlib.Path:
    return out / method / (setting or "none") / task_id.replace("/", "__") / f"rep{rep}.json"


@functools.cache
def load_factory(spec: str) -> Any:
    """The function named by `module:function` or `path/to/file.py:function`."""
    module_name, _, function_name = spec.rpartition(":")
    if not module_name:
        raise SystemExit("--factory takes module:function or path/to/file.py:function")
    if module_name.endswith(".py"):
        path = pathlib.Path(module_name).resolve()
        sys.path.insert(0, str(path.parent))  # so the file can import its neighbours
        spec_ = importlib.util.spec_from_file_location(path.stem, path)
        assert spec_ is not None and spec_.loader is not None
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(module_name)
    return getattr(module, function_name)


def build(
    args: argparse.Namespace, task: pb.Task, model: Any, bench: pb.Benchmark
) -> tuple[pb.Sampler, Any]:
    if args.sampler == "original":
        return pb.original_model_sampler(model, task), pb.LedgerTotals.zero()
    if args.sampler == "instruction":
        return pb.instruction_sampler(model, task), pb.LedgerTotals.zero()
    candidate, spend = load_factory(args.factory)(task=task, model=model, benchmark=bench)
    return candidate, spend.totals() if isinstance(spend, pb.Ledger) else spend


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    """Write atomically, so that a job killed mid-write leaves no partial file to skip over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.partial")
    partial.write_text(json.dumps(value, indent=1, allow_nan=True) + "\n")
    partial.replace(path)


def run_worker(
    args: argparse.Namespace, items: list[tuple[str, int, str]], worker: int, workers: int
) -> None:
    if not items:
        return
    bench = pb.Benchmark()
    settings = results.EvaluationSettings()
    method = method_name(args)
    model = None
    if args.sampler != "fallback":
        device = f"cuda:{worker}" if workers > 1 else None
        model = pb.OriginalModel.from_pretrained(
            dtype=args.dtype, device=device, batch_size=args.batch_size
        )
    for task_id, rep, path in items:
        started = time.time()
        task = bench.task(task_id)
        task.require_scored()
        if model is None:
            result = bench.fallback_result(task, seed=rep, replicate=rep, settings=settings)
            result = dataclasses.replace(result, method=method, setting=args.setting)
        else:
            candidate, spend = build(args, task, model, bench)
            result = bench.evaluate(
                task,
                candidate,
                model,
                ledger=spend,
                seed=rep,
                replicate=rep,
                settings=settings,
                method=method,
                setting=args.setting,
                keep_details=args.details,
            )
        run = {
            "seconds": round(time.time() - started, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cluster": os.environ.get("CLUV_CLUSTER"),
            "git_commit": os.environ.get("GIT_COMMIT"),
        }
        write_json(pathlib.Path(path), {**result.to_dict(), "run": run})
        print(
            f"[worker {worker}] {task_id} rep {rep}: hit rate {result.native_hit_rate:.3f}, "
            f"fwd KL {result.capped_forward_kl:.2f}, rev KL {result.reverse_kl:.2f} "
            f"({run['seconds']}s) -> {path}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    # --out may name $SCRATCH: cluv passes the command after `--` to the job unexpanded.
    out = pathlib.Path(os.path.expandvars(args.out)) if args.out else None
    out = out or pathlib.Path(os.environ.get("PB_RUN_DIR", "runs")) / "benchmark"
    bench = pb.Benchmark()
    if args.tasks in ("benchmark", "scored"):
        task_ids = [task.task_id for task in bench.tasks(args.tasks)]
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    replicates = [int(r) for r in args.replicates.split(",") if r.strip()]
    method = method_name(args)

    index, count = map(int, args.shard.split("/")) if args.shard else array_shard()
    every = [(t, r) for t in task_ids for r in replicates]
    mine = every[index::count]
    todo = [
        (t, r, str(result_path(out, method, args.setting, t, r)))
        for t, r in mine
        if not result_path(out, method, args.setting, t, r).exists()
    ]
    print(
        f"shard {index}/{count}: {len(mine)} of {len(every)} evaluations, {len(todo)} left to run; "
        f"results in {out}",
        flush=True,
    )

    workers = args.workers
    if workers is None:
        import torch

        workers = max(1, torch.cuda.device_count()) if args.sampler != "fallback" else 1
    workers = max(1, min(workers, len(todo)))
    if workers == 1:
        run_worker(args, todo, 0, 1)
        return
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=run_worker, args=(args, todo[w::workers], w, workers))
        for w in range(workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [w for w, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise SystemExit(f"workers {failed} failed; rerun the job to retry what is left")


if __name__ == "__main__":
    main()
