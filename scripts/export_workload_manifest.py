from __future__ import annotations

import argparse
from pathlib import Path

from peakaware.models import TrainingTaskRegistry
from peakaware.workload_manifest import build_workload_manifest, write_workload_manifest


DEFAULT_TASKS = ("resnet50", "vit_b_16", "bert_base", "gpt2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the publication workload manifest and T-1 table.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON manifest path.")
    parser.add_argument("--t1-output", type=Path, required=True, help="Output T-1 Markdown path.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Registry task keys to export.")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compiler-mode", help="Compiler mode recorded in the environment snapshot.")
    args = parser.parse_args()

    registry = TrainingTaskRegistry.with_defaults()
    unknown = sorted(set(args.tasks) - set(registry.names()))
    if unknown:
        parser.error(f"unknown task keys: {', '.join(unknown)}")
    tasks = [registry.get(name) for name in args.tasks]
    manifest = build_workload_manifest(
        tasks,
        microbatch_size=args.microbatch_size,
        seed=args.seed,
        compiler_mode=args.compiler_mode,
    )
    write_workload_manifest(manifest, args.output, args.t1_output)


if __name__ == "__main__":
    main()
