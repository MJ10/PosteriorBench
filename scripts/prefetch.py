"""Download what PosteriorBench jobs read from the Hugging Face Hub into the Hugging Face cache.

Compute nodes of most DRAC and PAICE clusters have no internet access, so the jobs run with
HF_HUB_OFFLINE=1 and read the model and the corpus hits from the cache ($HF_HOME). Run this on a
login node first; `scripts/setup_cluster.sh` does, and sets HF_HOME to the same place as the jobs.

    uv run python scripts/prefetch.py                     # model + hits of the 108 scored pairs
    uv run python scripts/prefetch.py --hits benchmark    # model + hits of the 21 benchmark tasks
    uv run python scripts/prefetch.py --corpus wc_1853    # also one prompt's whole corpus (~7 GB)
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

import posteriorbench as pb
from posteriorbench import data, original_model


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--hits",
        choices=["scored", "benchmark", "none"],
        default="scored",
        help="Corpus hits to download, which methods read to build samplers: the 108 scored pairs "
        "(3.4 GB), the 21 benchmark tasks, or none. Evaluation alone needs none.",
    )
    parser.add_argument(
        "--corpus",
        nargs="*",
        default=[],
        metavar="PROMPT_ID",
        help="Prompts whose whole corpus to download as well (about 7 GB each), for defining new "
        "behaviors.",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=[original_model.DEFAULT_MODEL],
        help="Hub models to download (default: the benchmark's model, at its pinned revision).",
    )
    args = parser.parse_args()

    print(
        f"Hugging Face cache: {os.environ.get('HF_HOME', '~/.cache/huggingface (HF_HOME unset)')}"
    )
    for name in args.model:
        revision = original_model.resolve_model_revision(name)
        path = snapshot_download(name, revision=revision)
        print(f"model {name}@{revision or 'main'} -> {path}")

    patterns: list[str] = []
    if args.hits != "none":
        tasks = pb.Benchmark().tasks(args.hits)
        patterns += [f"hits/{task.prompt_id}/{task.behavior}.parquet" for task in tasks]
    patterns += [f"corpus/{prompt_id}/*.parquet" for prompt_id in args.corpus]
    if patterns:
        path = snapshot_download(
            data.HUB_DATASET,
            repo_type=data.HUB_REPO_TYPE,
            revision=data.DATASET_REVISION,
            allow_patterns=patterns,
        )
        print(
            f"dataset {data.HUB_DATASET}@{data.DATASET_REVISION} ({len(patterns)} patterns) -> {path}"
        )


if __name__ == "__main__":
    main()
