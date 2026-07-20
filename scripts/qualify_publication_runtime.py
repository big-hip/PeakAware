from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.models.registry import TrainingTaskRegistry
from peakaware.publication.qualification import (
    PUBLICATION_BACKENDS,
    PUBLICATION_METHODS,
    build_qualification_slots,
    run_qualification_slots,
    validate_qualification_artifact_bundle,
    write_qualification_artifacts,
)


DEFAULT_TASKS = "resnet50,vit_b_16,bert_base,gpt2"


def _csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value")
    return values


def _budget_mib(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in _csv(text))
    if any(value <= 0 for value in values):
        raise ValueError("--budget-mib values must be positive")
    return tuple(value << 20 for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify publication runtimes in isolated spawn processes.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--backends", default=",".join(PUBLICATION_BACKENDS))
    parser.add_argument("--methods", default=",".join(PUBLICATION_METHODS))
    parser.add_argument("--budget-mib", default="4096")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument(
        "--min-cut-activation-memory-budget",
        type=float,
        default=None,
        help=(
            "Explicit torch._functorch.config.activation_memory_budget ratio for pytorch_min_cut. "
            "Omit to keep physical-budget conversion fail-closed."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--t1-output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    existing_outputs = [
        str(path)
        for path in (args.output_jsonl, args.manifest_json, args.t1_output, summary_path)
        if path.exists()
    ]
    if existing_outputs:
        parser.error(f"publication artifacts are immutable; paths already exist: {existing_outputs}")

    registry = TrainingTaskRegistry.with_defaults()
    task_names = _csv(args.tasks)
    unknown = sorted(set(task_names) - set(registry.names()))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; available: {registry.names()}")
    backends = _csv(args.backends)
    methods = _csv(args.methods)
    run_id = args.run_id or time.strftime("qualification-%Y%m%dT%H%M%SZ", time.gmtime())
    tasks = tuple(registry.get(name) for name in task_names)
    method_configs = {}
    if args.min_cut_activation_memory_budget is not None:
        if not 0.0 <= args.min_cut_activation_memory_budget <= 1.0:
            parser.error("--min-cut-activation-memory-budget must be in [0, 1]")
        method_configs["pytorch_min_cut"] = {
            "activation_memory_budget": args.min_cut_activation_memory_budget,
            "source": "explicit_cli_ratio",
        }
    warmups = {
        backend: args.warmup if args.warmup is not None else (5 if backend == "aot_eager" else 10)
        for backend in backends
    }
    manifest, slots = build_qualification_slots(
        tasks,
        run_id=run_id,
        backends=backends,
        methods=methods,
        memory_budgets_bytes=_budget_mib(args.budget_mib),
        replicates=args.replicates,
        microbatch_size=args.microbatch_size,
        device=args.device,
        base_seed=args.seed,
        attempt_index=args.attempt_index,
        repeat_count=args.repeats,
        timeout_s=args.timeout,
        warmup_steps_by_backend=warmups,
        method_configs=method_configs,
    )
    records = run_qualification_slots(
        slots,
        manifest,
        timeout_s=args.timeout,
        warmup_steps=args.warmup,
        repeat_count=args.repeats,
    )
    summary_path = write_qualification_artifacts(
        records,
        manifest,
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        t1_output=args.t1_output,
    )
    committed = validate_qualification_artifact_bundle(
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        t1_output=args.t1_output,
    )
    summary = {key: value for key, value in committed.items() if key != "artifact_commit"}
    summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["qualification_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
