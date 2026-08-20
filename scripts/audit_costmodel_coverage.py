from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
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
from peakaware.cost import build_composite_provider
from peakaware.cost.base import signature_for_op
from peakaware.cost.legacy_adapter import default_legacy_hardware
from peakaware.cost.profile_db import profile_signature_hash
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.models import TrainingTaskRegistry
from peakaware.plugins import ServiceKind, build_default_registry


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


def _hardware_payload(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda_available": torch.cuda.is_available()}
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _legacy_hardware_payload(hardware: str) -> dict[str, Any]:
    chip_name, _, topo_name = hardware.partition(",")
    chip_path = (
        ROOT
        / "Costmodel/atencost/backend/analytical_model/hardware/chip_configs"
        / f"{chip_name}.json"
    )
    topo_path = (
        ROOT
        / "Costmodel/atencost/backend/analytical_model/hardware/topo_configs"
        / f"{(topo_name or chip_name)}.json"
    )
    return {
        "hardware_id": hardware,
        "chip_config_path": str(chip_path),
        "topology_config_path": str(topo_path),
        "chip": json.loads(chip_path.read_text(encoding="utf-8")),
        "topology": json.loads(topo_path.read_text(encoding="utf-8")),
    }


def _source_family(source: str) -> str:
    if source == "profile_db_exact":
        return "profile_db_exact"
    if source == "profile_db_interpolated":
        return "profile_db_interpolated"
    if source == "structural_zero":
        return "structural_zero"
    if source == "metadata_view_zero":
        return "metadata_zero"
    if source.startswith("analytical:"):
        return "analytical"
    if source == "legacy_adapter:static_fallback":
        return "static_fallback"
    if source.startswith("legacy_adapter:atencost_analytical:"):
        return "analytical"
    if "roofline" in source:
        return "roofline_fallback"
    return "other"


def _audit_task(
    task_name: str,
    *,
    device: torch.device,
    microbatch_size: int,
    profile_db: Path | None,
) -> dict[str, Any]:
    task = TrainingTaskRegistry.with_defaults().get(task_name)
    model = task.build_model().to(device)
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(microbatch_size)
    args = tuple(_move_to_device(args, device))
    kwargs = dict(_move_to_device(kwargs, device))
    config = PeakAwareConfig(capture_backend="aot")
    total_memory = None
    if device.type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=total_memory or (1 << 40),
        config=config,
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec(str(device), device.type == "cuda", total_memory),
        request_key=f"cost-coverage:{task_name}:mb{microbatch_size}",
    )

    started = time.perf_counter()
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    if not report.valid:
        raise ValueError(f"invalid IR for {task_name}: {report.errors}")
    registry = build_default_registry(profile_db_path=profile_db)
    provider = build_composite_provider(
        tuple(
            record.service
            for record in registry.services_for(ServiceKind.COST_PROVIDER)
        )
    )

    query_cache: dict[Any, Any] = {}
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_time_us: defaultdict[str, float] = defaultdict(float)
    family_time_us: defaultdict[str, float] = defaultdict(float)
    hardware_versions: Counter[str] = Counter()
    software_versions: Counter[str] = Counter()
    target_rows: dict[tuple[str, str], dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []

    for op in ir.ops:
        signature = signature_for_op(ir, op)
        signature_key = profile_signature_hash(signature)
        query = query_cache.get(signature_key)
        if query is None:
            query = provider.estimate_with_provenance(signature)
            query_cache[signature_key] = query
        if query is None:
            unresolved.append(
                {
                    "op_name": op.name,
                    "target": op.target,
                    "input_shapes": signature.input_shapes,
                    "output_shapes": signature.output_shapes,
                    "dtype": signature.dtype,
                }
            )
            continue
        cost = query.cost
        source = str(cost.source)
        family = _source_family(source)
        source_counts[source] += 1
        family_counts[family] += 1
        source_time_us[source] += float(cost.estimated_us)
        family_time_us[family] += float(cost.estimated_us)
        hardware_versions[str(cost.hardware_version)] += 1
        software_versions[str(cost.software_version)] += 1
        key = (str(op.target), source)
        row = target_rows.setdefault(
            key,
            {
                "task_name": task_name,
                "target": str(op.target),
                "source": source,
                "source_family": family,
                "count": 0,
                "estimated_time_us": 0.0,
                "min_confidence": 1.0,
                "example_input_shapes": signature.input_shapes,
                "example_output_shapes": signature.output_shapes,
                "dtype": signature.dtype,
            },
        )
        row["count"] += 1
        row["estimated_time_us"] += float(cost.estimated_us)
        row["min_confidence"] = min(float(row["min_confidence"]), float(cost.confidence))

    op_count = len(ir.ops)
    estimated_total_us = sum(source_time_us.values())
    fallback_count = sum(
        count
        for family, count in family_counts.items()
        if family in {"static_fallback", "roofline_fallback"}
    )
    fallback_time_us = sum(
        value
        for family, value in family_time_us.items()
        if family in {"static_fallback", "roofline_fallback"}
    )
    return {
        "task_name": task_name,
        "display_name": None if task.workload is None else task.workload.display_name,
        "microbatch_size": microbatch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "capture_backend": capture.backend,
        "capture_failure_count": len(capture.failures),
        "ir_op_count": op_count,
        "unique_signature_count": len(query_cache),
        "unique_target_count": len({op.target for op in ir.ops}),
        "unresolved_op_count": len(unresolved),
        "fallback_op_count": fallback_count,
        "fallback_op_rate": None if op_count == 0 else fallback_count / op_count,
        "estimated_total_us": estimated_total_us,
        "fallback_estimated_time_us": fallback_time_us,
        "fallback_estimated_time_rate": (
            None if estimated_total_us <= 0 else fallback_time_us / estimated_total_us
        ),
        "source_family_counts": dict(family_counts),
        "source_family_time_us": dict(family_time_us),
        "source_counts": dict(source_counts),
        "hardware_versions": dict(hardware_versions),
        "software_versions": dict(software_versions),
        "unresolved_ops": unresolved,
        "target_rows": sorted(
            target_rows.values(),
            key=lambda row: (
                row["source_family"] not in {"static_fallback", "roofline_fallback"},
                -int(row["count"]),
                str(row["target"]),
            ),
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _aggregate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    op_count = sum(int(task["ir_op_count"]) for task in tasks)
    fallback_count = sum(int(task["fallback_op_count"]) for task in tasks)
    total_us = sum(float(task["estimated_total_us"]) for task in tasks)
    fallback_us = sum(float(task["fallback_estimated_time_us"]) for task in tasks)
    return {
        "task_count": len(tasks),
        "ir_op_count": op_count,
        "fallback_op_count": fallback_count,
        "fallback_op_rate": None if op_count == 0 else fallback_count / op_count,
        "estimated_total_us": total_us,
        "fallback_estimated_time_us": fallback_us,
        "fallback_estimated_time_rate": None if total_us <= 0 else fallback_us / total_us,
        "unresolved_op_count": sum(int(task["unresolved_op_count"]) for task in tasks),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# Cost Model Coverage Audit",
        "",
        f"Hardware: `{payload['legacy_hardware']['hardware_id']}` / "
        f"`{payload['runtime_hardware'].get('name', payload['runtime_hardware']['device'])}`.",
        "",
        "| Task | IR ops | Unique signatures | Structural | Metadata views | Analytical | Profile exact | Profile interpolated | Static fallback | Fallback rate | Fallback time share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in payload["tasks"]:
        families = task["source_family_counts"]
        lines.append(
            f"| {task['display_name']} | {task['ir_op_count']} | {task['unique_signature_count']} | "
            f"{families.get('structural_zero', 0)} | {families.get('metadata_zero', 0)} | "
            f"{families.get('analytical', 0)} | {families.get('profile_db_exact', 0)} | "
            f"{families.get('profile_db_interpolated', 0)} | {families.get('static_fallback', 0)} | "
            f"{task['fallback_op_rate']:.2%} | {task['fallback_estimated_time_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Aggregate fallback: {aggregate['fallback_op_count']}/{aggregate['ir_op_count']} "
            f"({aggregate['fallback_op_rate']:.2%} of ops; "
            f"{aggregate['fallback_estimated_time_rate']:.2%} of estimated time).",
            "",
            "`unresolved_op_count` must be zero. Static fallback is still considered an uncovered cost-model route and must be replaced by profiling, analytical modeling, or an explicit similarity mapping before the model is considered complete.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit per-operator Cost Model coverage and provenance on registered workloads."
    )
    parser.add_argument(
        "--tasks",
        default="bert_base_full_s128,gpt2_small_full_s128,resnet50,vit_b_16",
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    legacy_hardware = default_legacy_hardware()
    task_rows = [
        _audit_task(
            task_name,
            device=device,
            microbatch_size=args.microbatch_size,
            profile_db=args.profile_db,
        )
        for task_name in _csv_values(args.tasks)
    ]
    payload = {
        "schema_version": "costmodel-coverage-v1",
        "runtime_hardware": _hardware_payload(device),
        "legacy_hardware": _legacy_hardware_payload(legacy_hardware),
        "profile_db": None if args.profile_db is None else str(args.profile_db),
        "tasks": task_rows,
        "aggregate": _aggregate(task_rows),
    }
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "costmodel_coverage.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        root / "costmodel_targets.csv",
        [row for task in task_rows for row in task["target_rows"]],
    )
    _write_markdown(root / "COSTMODEL_COVERAGE_REPORT.md", payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
