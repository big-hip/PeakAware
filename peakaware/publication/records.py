from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peakaware.experiments import ExperimentRecord, experiment_records_from_dicts


RECORD_VALIDATION_SCHEMA_VERSION = "0.1"
_OK_STATUSES = {"ok"}
_FAILURE_STATUSES = {
    "failed",
    "budget_violation",
    "oom",
    "timeout",
    "correctness_failure",
    "unsupported",
    "infra_failure",
}


def validate_publication_records(
    records_json: str | Path,
    *,
    require_frozen: bool = False,
    require_runtime_identity: bool = False,
) -> dict[str, Any]:
    path = Path(records_json)
    errors: list[str] = []
    warnings: list[str] = []
    records: tuple[ExperimentRecord, ...] = ()
    if not path.is_file():
        return _payload(path, (), (f"missing records file: {path}",), (), require_frozen, require_runtime_identity)
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, list):
            raise ValueError("records JSON must contain a list")
        records = experiment_records_from_dicts(raw_payload)
    except Exception as exc:
        return _payload(path, (), (f"invalid records file: {exc}",), (), require_frozen, require_runtime_identity)
    if not records:
        errors.append("records file contains no rows")
    for index, record in enumerate(records):
        _validate_record(index, record, errors, warnings, require_runtime_identity=require_runtime_identity)
    if require_frozen:
        warnings.append("records validation cannot prove EV-* ledger freeze without an evidence manifest")
    return _payload(path, records, tuple(errors), tuple(warnings), require_frozen, require_runtime_identity)


def _validate_record(
    index: int,
    record: ExperimentRecord,
    errors: list[str],
    warnings: list[str],
    *,
    require_runtime_identity: bool,
) -> None:
    prefix = f"record[{index}]"
    if not record.task_name:
        errors.append(f"{prefix} missing task_name")
    if record.budget_bytes <= 0:
        errors.append(f"{prefix} budget_bytes must be positive")
    known_statuses = _OK_STATUSES | _FAILURE_STATUSES
    if record.status not in known_statuses:
        warnings.append(f"{prefix} uses non-standard status {record.status!r}")
    if record.status == "ok":
        if not record.selected_plan_id:
            errors.append(f"{prefix} ok row missing selected_plan_id")
        if record.measured_peak_bytes is None:
            errors.append(f"{prefix} ok row missing measured_peak_bytes")
        if record.measured_step_us is None:
            errors.append(f"{prefix} ok row missing measured_step_us")
        if record.all_save_measured_peak_bytes is None:
            errors.append(f"{prefix} ok row missing all_save_measured_peak_bytes")
        if not record.measured_plan_results:
            warnings.append(f"{prefix} ok row has no measured_plan_results")
        if require_runtime_identity and record.selected_aot_partition_runtime is not True:
            errors.append(f"{prefix} missing selected lowered-AOT runtime identity")
    if record.measured_peak_bytes is not None and record.measured_peak_bytes < 0:
        errors.append(f"{prefix} measured_peak_bytes must be non-negative")
    if record.measured_step_us is not None and record.measured_step_us <= 0:
        errors.append(f"{prefix} measured_step_us must be positive")


def _payload(
    path: Path,
    records: tuple[ExperimentRecord, ...],
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    require_frozen: bool,
    require_runtime_identity: bool,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    return {
        "schema_version": RECORD_VALIDATION_SCHEMA_VERSION,
        "records_json": str(path),
        "ok": not errors,
        "require_frozen": require_frozen,
        "require_runtime_identity": require_runtime_identity,
        "record_count": len(records),
        "status_counts": status_counts,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": list(errors),
        "warnings": list(warnings),
    }
