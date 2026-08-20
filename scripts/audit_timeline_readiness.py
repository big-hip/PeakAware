from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIMELINE_FIELDS: tuple[str, ...] = (
    "selected_actual_memory_timeline",
    "selected_simulated_memory_timeline",
    "selected_actual_memory_trace",
    "selected_actual_sampled_memory_trace",
    "selected_actual_overall_sampled_memory_trace",
    "selected_simulated_memory_event_trace",
    "selected_lowered_fx_l2_simulated_memory_event_trace",
)

GPU_FIELDS: tuple[str, ...] = (
    "gpu_util_trace",
    "selected_gpu_util_trace",
    "profiler_kernel_trace",
    "selected_profiler_kernel_trace",
    "gpu_compute_summary",
    "selected_gpu_compute_summary",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether experiment records can generate actual-vs-simulated memory and GPU-utilization timelines."
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--require-gpu-util", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records_json)
    report = audit_records(records, require_gpu_util=args.require_gpu_util)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    if not report["timeline_ready"]:
        raise SystemExit(2)
    if args.require_gpu_util and not report["gpu_ready"]:
        raise SystemExit(3)


def audit_records(records: list[dict[str, Any]], *, require_gpu_util: bool = False) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    field_counts = _field_counts(ok_records, TIMELINE_FIELDS)
    gpu_field_counts = _field_counts(ok_records, GPU_FIELDS)
    timeline_ready_records = [
        record
        for record in ok_records
        if (
            _has_points(record.get("selected_actual_overall_sampled_memory_trace"))
            or _has_points(record.get("selected_actual_sampled_memory_trace"))
            or _has_points(record.get("selected_actual_memory_trace"))
            or _has_points(record.get("selected_actual_memory_timeline"))
        )
        and (
            _has_points(record.get("selected_lowered_fx_l2_simulated_memory_event_trace"))
            or _has_points(record.get("selected_simulated_memory_event_trace"))
            or _has_points(record.get("selected_simulated_memory_timeline"))
        )
    ]
    gpu_ready_records = [
        record
        for record in ok_records
        if _has_gpu_data(record)
    ]
    missing_examples = [
        _record_brief(record)
        for record in ok_records
        if record not in timeline_ready_records
    ][:10]
    gpu_missing_examples = [
        _record_brief(record)
        for record in ok_records
        if record not in gpu_ready_records
    ][:10]
    return {
        "record_count": len(records),
        "ok_record_count": len(ok_records),
        "timeline_ready": bool(timeline_ready_records),
        "timeline_ready_record_count": len(timeline_ready_records),
        "timeline_ready_rate": _rate(len(timeline_ready_records), len(ok_records)),
        "gpu_ready": bool(gpu_ready_records),
        "gpu_ready_record_count": len(gpu_ready_records),
        "gpu_ready_rate": _rate(len(gpu_ready_records), len(ok_records)),
        "require_gpu_util": require_gpu_util,
        "field_counts": field_counts,
        "gpu_field_counts": gpu_field_counts,
        "missing_timeline_examples": missing_examples,
        "missing_gpu_examples": gpu_missing_examples if require_gpu_util else [],
        "expected_actual_fields": (
            "selected_actual_overall_sampled_memory_trace",
            "selected_actual_sampled_memory_trace",
            "selected_actual_memory_trace",
            "selected_actual_memory_timeline",
        ),
        "expected_simulated_fields": (
            "selected_lowered_fx_l2_simulated_memory_event_trace",
            "selected_simulated_memory_event_trace",
            "selected_simulated_memory_timeline",
        ),
        "expected_gpu_fields": GPU_FIELDS,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _field_counts(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return {field: sum(1 for record in records if _has_points(record.get(field)) or isinstance(record.get(field), dict)) for field in fields}


def _has_points(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0


def _has_gpu_data(record: dict[str, Any]) -> bool:
    if _has_points(record.get("gpu_util_trace")) or _has_points(record.get("selected_gpu_util_trace")):
        return True
    for field in ("gpu_compute_summary", "selected_gpu_compute_summary"):
        summary = record.get(field)
        if isinstance(summary, dict) and summary.get("status") == "ok" and int(summary.get("sample_count") or 0) > 0:
            return True
    return False


def _rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def _record_brief(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": record.get("task_name"),
        "variant_name": record.get("variant_name"),
        "selected_plan_id": record.get("selected_plan_id"),
        "budget_bytes": record.get("budget_bytes"),
        "matrix_pass_index": record.get("matrix_pass_index"),
    }


if __name__ == "__main__":
    main()
