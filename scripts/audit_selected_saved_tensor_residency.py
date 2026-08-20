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


def _storage_key(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage()._cdata)


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _optimizer_tensors(optimizer: torch.optim.Optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                yield value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect physical storages retained by the simulator-selected AOT "
            "executable. This executes only the selected plan."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--budget-mib", type=int, default=45000)
    parser.add_argument("--compiler-refinement-top-k", type=int, default=1)
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
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
        memory_budget_bytes=args.budget_mib << 20,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=12,
            max_greedy_candidates=12,
            validation_top_k=0,
            capture_backend="aot",
            compiler_refinement_top_k=args.compiler_refinement_top_k,
            selection_objective="min_time_then_peak",
            profile_db_path=args.profile_db,
            enable_compile=args.enable_compile,
        ),
    )

    # Materialize optimizer state once, then inspect the same selected plan in
    # steady state. No alternative candidate is executed.
    optimizer.zero_grad(set_to_none=True)
    warm_output = result.executor.executable(*example_args, **example_kwargs)
    task.loss_fn(warm_output).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    external_tensors = tuple(model.parameters()) + tuple(model.buffers()) + tuple(
        _optimizer_tensors(optimizer)
    )
    external_keys = {_storage_key(tensor) for tensor in external_tensors}
    saved_by_key: dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        key = _storage_key(tensor)
        saved_by_key[key] = max(saved_by_key.get(key, 0), _storage_nbytes(tensor))
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    before_forward_bytes = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = result.executor.executable(*example_args, **example_kwargs)
        loss = task.loss_fn(output)
    torch.cuda.synchronize(device)
    after_forward_bytes = int(torch.cuda.memory_allocated(device))
    forward_peak_bytes = int(torch.cuda.max_memory_allocated(device))

    external_saved = {
        key: nbytes for key, nbytes in saved_by_key.items() if key in external_keys
    }
    internal_saved = {
        key: nbytes for key, nbytes in saved_by_key.items() if key not in external_keys
    }
    payload = {
        "schema_version": "selected-saved-storage-audit-v1",
        "task_name": args.task,
        "microbatch_size": args.microbatch,
        "selected_plan_id": result.selected_plan.plan_id,
        "candidate_gpu_measurements_used_for_selection": 0,
        "selected_plan_diagnostic_executions": 1,
        "outer_compile_enabled": args.enable_compile,
        "before_forward_bytes": before_forward_bytes,
        "after_forward_bytes": after_forward_bytes,
        "forward_peak_bytes": forward_peak_bytes,
        "forward_allocated_delta_bytes": after_forward_bytes - before_forward_bytes,
        "saved_storage_count": len(saved_by_key),
        "saved_storage_bytes": sum(saved_by_key.values()),
        "external_saved_storage_count": len(external_saved),
        "external_saved_storage_bytes": sum(external_saved.values()),
        "internal_saved_storage_count": len(internal_saved),
        "internal_saved_storage_bytes": sum(internal_saved.values()),
        "simulated_peak_bytes": result.selected_plan.estimated_peak_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
