from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils import _pytree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware import PeakAwareConfig, optimize_training
from peakaware.models import TrainingTaskRegistry


def _move_to_device(value: Any, device: torch.device) -> Any:
    return _pytree.tree_map(
        lambda item: item.to(device) if isinstance(item, torch.Tensor) else item,
        value,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile memory-allocating operators for one simulator-selected plan."
    )
    parser.add_argument("--task", default="gpt2_small_full_s128")
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--compiler-refinement-top-k", type=int, default=1)
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome-trace", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    task = TrainingTaskRegistry.with_defaults().get(args.task)
    model = task.build_model().to(device)
    optimizer = task.build_optimizer(model)
    example_args, example_kwargs = task.build_batch(args.microbatch)
    example_args = _move_to_device(example_args, device)
    example_kwargs = _move_to_device(example_kwargs, device)
    result = optimize_training(
        model,
        example_args,
        example_kwargs=example_kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=45000 << 20,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=12,
            max_greedy_candidates=12,
            validation_top_k=0,
            capture_backend="aot",
            enable_compile=True,
            compiler_refinement_top_k=args.compiler_refinement_top_k,
            selection_objective="min_time_then_peak",
            profile_db_path=args.profile_db,
        ),
    )

    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        output = result.executor.executable(*example_args, **example_kwargs)
        task.loss_fn(output).backward()
        optimizer.step()
    del output
    torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("peakaware_selected_fw"):
            output = result.executor.executable(*example_args, **example_kwargs)
            loss = task.loss_fn(output)
        with torch.profiler.record_function("peakaware_selected_bw"):
            loss.backward()
        with torch.profiler.record_function("peakaware_selected_optimizer"):
            optimizer.step()
    torch.cuda.synchronize(device)

    if args.chrome_trace is not None:
        args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.chrome_trace))

    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        device_memory = int(getattr(event, "self_device_memory_usage", 0) or 0)
        cuda_memory = int(getattr(event, "self_cuda_memory_usage", 0) or 0)
        allocated = device_memory if device_memory != 0 else cuda_memory
        if allocated <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "input_shapes": event.input_shapes,
                "self_device_memory_bytes": allocated,
                "device_memory_bytes": int(
                    getattr(event, "device_memory_usage", 0) or 0
                ),
                "count": int(event.count),
                "self_device_time_us": float(
                    getattr(event, "self_device_time_total", 0.0) or 0.0
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["self_device_memory_bytes"],
            row["device_memory_bytes"],
        ),
        reverse=True,
    )
    event_rows = []
    for event in profiler.events():
        device_memory = int(
            getattr(event, "self_device_memory_usage", 0) or 0
        )
        cuda_memory = int(getattr(event, "self_cuda_memory_usage", 0) or 0)
        allocated = device_memory if device_memory != 0 else cuda_memory
        if allocated <= 0:
            continue
        parents = []
        parent = getattr(event, "cpu_parent", None)
        while parent is not None and len(parents) < 8:
            parents.append(str(parent.name))
            parent = getattr(parent, "cpu_parent", None)
        event_rows.append(
            {
                "name": event.name,
                "input_shapes": getattr(event, "input_shapes", None),
                "self_device_memory_bytes": allocated,
                "cpu_parent_chain": parents,
            }
        )
    event_rows.sort(
        key=lambda row: row["self_device_memory_bytes"],
        reverse=True,
    )
    payload = {
        "schema_version": "selected-backward-memory-profile-v1",
        "task_name": args.task,
        "microbatch_size": args.microbatch,
        "selected_plan_id": result.selected_plan.plan_id,
        "candidate_gpu_measurements_used_for_selection": 0,
        "selected_plan_profile_executions": 1,
        "simulated_peak_bytes": result.selected_plan.estimated_peak_bytes,
        "top_positive_self_memory_operators": rows[:100],
        "top_positive_memory_events": event_rows[:300],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
