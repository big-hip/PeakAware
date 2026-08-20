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

from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.cost.attention import ScaledDotProductAttentionCostProvider
from peakaware.cost.base import OpSignature, signature_for_op
from peakaware.cost.collector import collect_microbenchmark
from peakaware.cost.profile_db import ProfileDB, profile_signature_hash
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.models import TrainingTaskRegistry


def _csv_values(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one task is required")
    return values


def _move_to_device(value: Any, device: torch.device) -> Any:
    return _pytree.tree_map(
        lambda item: item.to(device) if isinstance(item, torch.Tensor) else item,
        value,
    )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).replace("torch.", "").lower()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(normalized, torch.float32)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _output_bytes(value: Any) -> int:
    leaves, _ = _pytree.tree_flatten(value)
    return sum(_tensor_bytes(item) for item in leaves if isinstance(item, torch.Tensor))


def _qkv_shapes(signature: OpSignature) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]:
    shapes = tuple(shape for shape in signature.input_shapes if len(shape) == 4)
    backward = "backward" in signature.target.lower()
    if backward and len(shapes) >= 4:
        return shapes[1], shapes[2], shapes[3], True
    if not backward and len(shapes) >= 3:
        return shapes[0], shapes[1], shapes[2], False
    raise ValueError(f"unsupported SDPA signature: {signature.target} {signature.input_shapes}")


def _collect_sdpa_signatures(
    task_name: str,
    *,
    device: torch.device,
    microbatch_size: int,
) -> tuple[OpSignature, ...]:
    task = TrainingTaskRegistry.with_defaults().get(task_name)
    model = task.build_model().to(device)
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(microbatch_size)
    args = tuple(_move_to_device(args, device))
    kwargs = dict(_move_to_device(kwargs, device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=total_memory,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec(str(device), True, total_memory),
        request_key=f"sdpa-profile:{task_name}:mb{microbatch_size}",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    if not report.valid:
        raise ValueError(f"invalid IR for {task_name}: {report.errors}")
    signatures: dict[str, OpSignature] = {}
    for op in ir.ops:
        if "scaled_dot_product" not in str(op.target).lower():
            continue
        signature = signature_for_op(ir, op)
        signatures.setdefault(profile_signature_hash(signature), signature)
    if not signatures:
        raise ValueError(f"no fused SDPA operators found in {task_name}")
    return tuple(signatures.values())


def _profile_signature(
    signature: OpSignature,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    db: ProfileDB,
    analytical: ScaledDotProductAttentionCostProvider,
) -> dict[str, Any]:
    query_shape, key_shape, value_shape, backward = _qkv_shapes(signature)
    dtype = _torch_dtype(signature.dtype)
    query = torch.randn(query_shape, device=device, dtype=dtype, requires_grad=backward)
    key = torch.randn(key_shape, device=device, dtype=dtype, requires_grad=backward)
    value = torch.randn(value_shape, device=device, dtype=dtype, requires_grad=backward)
    operator = torch.ops.aten._scaled_dot_product_efficient_attention.default
    if backward:
        prepared = operator(query, key, value, None, True, 0.0, False)
        grad_output = torch.randn_like(prepared[0])

        def execute() -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(
                prepared[0],
                (query, key, value),
                grad_output,
                retain_graph=True,
            )

        output_allocation_bytes = sum(_tensor_bytes(item) for item in (query, key, value))
    else:

        def execute() -> tuple[torch.Tensor, ...]:
            return operator(query, key, value, None, True, 0.0, False)

        output_allocation_bytes = _output_bytes(execute())
    result = collect_microbenchmark(
        signature,
        execute,
        warmup=warmup,
        repeats=repeats,
        db=db,
        output_allocation_bytes=output_allocation_bytes,
    )
    analytical_cost = analytical.estimate(signature)
    measured_us = float(result.record.p50_us)
    estimated_us = None if analytical_cost is None else float(analytical_cost.estimated_us)
    return {
        "signature_hash": profile_signature_hash(signature),
        "target": signature.target,
        "dtype": signature.dtype,
        "input_shapes": signature.input_shapes,
        "output_shapes": signature.output_shapes,
        "sample_count": result.record.sample_count,
        "p50_us": measured_us,
        "p90_us": result.record.p90_us,
        "mean_us": result.record.mean_us,
        "workspace_bytes": result.record.workspace_bytes,
        "analytical_us": estimated_us,
        "analytical_absolute_error_us": (
            None if estimated_us is None else abs(estimated_us - measured_us)
        ),
        "analytical_relative_error": (
            None
            if estimated_us is None or measured_us <= 0
            else abs(estimated_us - measured_us) / measured_us
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile fused SDPA forward/backward signatures captured from registered tasks."
    )
    parser.add_argument("--tasks", default="bert_base_full_s128,vit_b_16")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--profile-db", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.microbatch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid microbenchmark iteration or microbatch configuration")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fused SDPA profiling requires CUDA")
    db = ProfileDB(args.profile_db)
    analytical = ScaledDotProductAttentionCostProvider()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    task_signature_counts: dict[str, int] = {}
    for task_name in _csv_values(args.tasks):
        signatures = _collect_sdpa_signatures(
            task_name,
            device=device,
            microbatch_size=args.microbatch_size,
        )
        task_signature_counts[task_name] = len(signatures)
        for signature in signatures:
            key = profile_signature_hash(signature)
            if key in seen:
                continue
            seen.add(key)
            row = _profile_signature(
                signature,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                db=db,
                analytical=analytical,
            )
            row["task_name"] = task_name
            rows.append(row)
    relative_errors = [float(row["analytical_relative_error"]) for row in rows]
    payload = {
        "schema_version": "sdpa-profile-v1",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_properties(device).name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "profile_db": str(args.profile_db),
        "microbatch_size": args.microbatch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "task_signature_counts": task_signature_counts,
        "unique_profile_count": len(rows),
        "mean_analytical_relative_error": (
            None if not relative_errors else sum(relative_errors) / len(relative_errors)
        ),
        "max_analytical_relative_error": None if not relative_errors else max(relative_errors),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
