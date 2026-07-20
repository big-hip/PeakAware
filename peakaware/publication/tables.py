from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from peakaware.experiments import ExperimentRecord


TABLE_SCHEMA_VERSION = "0.1"
PUBLICATION_TABLE_IDS = ("T2", "T3", "T4", "T5", "T6", "T7", "T8")


@dataclass(frozen=True)
class TableArtifact:
    table_id: str
    output_dir: Path
    row_count: int
    status: str


def build_publication_tables(
    records: Sequence[ExperimentRecord],
    output_root: str | Path,
    *,
    status: str = "draft",
    source_paths: Sequence[str | Path] = (),
) -> tuple[TableArtifact, ...]:
    if status not in {"draft", "provisional", "frozen"}:
        raise ValueError("status must be draft, provisional, or frozen")
    if status == "frozen" and not source_paths:
        raise ValueError("frozen tables require checksum-able source paths")
    builders: dict[str, Callable[[Sequence[ExperimentRecord]], list[dict[str, Any]]]] = {
        "T2": _rows_t2_baseline_coverage,
        "T3": _rows_t3_budget_results,
        "T4": _rows_t4_prediction_accuracy,
        "T5": _rows_t5_root_causes,
        "T6": _rows_t6_costs,
        "T7": _rows_t7_runtime_correctness,
        "T8": _rows_t8_ablation,
    }
    root = Path(output_root)
    artifacts: list[TableArtifact] = []
    for table_id in PUBLICATION_TABLE_IDS:
        rows = builders[table_id](records)
        table_dir = root / _table_dir_name(table_id)
        _write_table_directory(table_id, rows, table_dir, status, source_paths)
        artifacts.append(TableArtifact(table_id, table_dir, len(rows), status))
    return tuple(artifacts)


def validate_publication_table_artifacts(
    table_root: str | Path,
    *,
    require_frozen: bool = False,
    expected_table_ids: Sequence[str] = PUBLICATION_TABLE_IDS,
) -> dict[str, Any]:
    root = Path(table_root)
    results = [_validate_table_dir(root / _table_dir_name(table_id), table_id, require_frozen) for table_id in expected_table_ids]
    error_count = sum(len(result["errors"]) for result in results)
    warning_count = sum(len(result["warnings"]) for result in results)
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "table_root": str(root),
        "ok": error_count == 0,
        "require_frozen": require_frozen,
        "expected_table_ids": list(expected_table_ids),
        "error_count": error_count,
        "warning_count": warning_count,
        "tables": results,
    }


def _table_dir_name(table_id: str) -> str:
    return {
        "T2": "T2_baseline_coverage",
        "T3": "T3_budget_results",
        "T4": "T4_prediction_accuracy",
        "T5": "T5_root_causes",
        "T6": "T6_topk_costs",
        "T7": "T7_runtime_correctness",
        "T8": "T8_ablation",
    }[table_id]


def _rows_t2_baseline_coverage(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        for plan in record.measured_plan_results:
            strategy = str(plan.get("plan_id", "unknown"))
            row = counts.setdefault(strategy, {"measured_rows": 0, "correct_rows": 0, "feasible_rows": 0})
            row["measured_rows"] += 1
            if plan.get("correctness_passed") is not False:
                row["correct_rows"] += 1
            if plan.get("measured_peak_bytes") is not None and int(plan["measured_peak_bytes"]) <= record.budget_bytes:
                row["feasible_rows"] += 1
    return [
        {"strategy": strategy, **values, "status": "proxy_or_internal"}
        for strategy, values in sorted(counts.items())
    ]


def _rows_t3_budget_results(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault((record.task_name, record.variant_name), []).append(record)
    rows = []
    for (task, variant), group in sorted(grouped.items()):
        ok = sum(1 for record in group if record.status == "ok")
        failed = len(group) - ok
        budget_safe = sum(
            1
            for record in group
            if record.status == "ok"
            and record.measured_peak_bytes is not None
            and record.measured_peak_bytes <= record.budget_bytes
        )
        rows.append(
            {
                "task_name": task,
                "variant": variant,
                "records": len(group),
                "ok": ok,
                "failed": failed,
                "budget_safe": budget_safe,
                "success_rate": None if not group else ok / len(group),
            }
        )
    return rows


def _rows_t4_prediction_accuracy(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows = []
    for task in sorted({record.task_name for record in records}):
        task_records = [record for record in records if record.task_name == task and record.status == "ok"]
        raw = [abs(record.selected_prediction_relative_error) for record in task_records if record.selected_prediction_relative_error is not None]
        calibrated = [
            abs(record.selected_calibrated_prediction_relative_error)
            for record in task_records
            if record.selected_calibrated_prediction_relative_error is not None
        ]
        rows.append(
            {
                "task_name": task,
                "n": len(task_records),
                "raw_mean_ape": _mean(raw),
                "raw_p90_ape": _percentile(raw, 0.9),
                "calibrated_mean_ape": _mean(calibrated),
                "calibrated_p90_ape": _percentile(calibrated, 0.9),
                "phase_accuracy": _phase_accuracy(task_records),
            }
        )
    return rows


def _rows_t5_root_causes(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        cause = record.diagnostic_primary_cause or ("failed:" + str(record.error_type) if record.status != "ok" else "unknown")
        counts[(record.task_name, cause)] = counts.get((record.task_name, cause), 0) + 1
    return [
        {"task_name": task, "cause": cause, "count": count}
        for (task, cause), count in sorted(counts.items())
    ]


def _rows_t6_costs(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows = []
    for task in sorted({record.task_name for record in records}):
        task_records = [record for record in records if record.task_name == task and record.status == "ok"]
        rows.append(
            {
                "task_name": task,
                "n": len(task_records),
                "mean_candidate_count": _mean([record.candidate_count for record in task_records]),
                "mean_measured_candidate_count": _mean([record.measured_candidate_count for record in task_records]),
                "mean_optimization_total_us": _mean_optional([record.optimization_total_us for record in task_records]),
                "mean_amortization_steps": _mean_optional([record.optimization_amortization_steps for record in task_records]),
            }
        )
    return rows


def _rows_t7_runtime_correctness(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows = []
    for task in sorted({record.task_name for record in records}):
        task_records = [record for record in records if record.task_name == task]
        ok_records = [record for record in task_records if record.status == "ok"]
        rows.append(
            {
                "task_name": task,
                "records": len(task_records),
                "ok": len(ok_records),
                "failed": len(task_records) - len(ok_records),
                "selected_aot_runtime": sum(1 for record in ok_records if record.selected_aot_partition_runtime),
                "selected_activation_checkpoint": sum(1 for record in ok_records if record.selected_activation_checkpoint),
                "failure_types": ";".join(sorted({str(record.error_type) for record in task_records if record.error_type})),
            }
        )
    return rows


def _rows_t8_ablation(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.variant_name, []).append(record)
    rows = []
    for variant, group in sorted(grouped.items()):
        ok_records = [record for record in group if record.status == "ok"]
        rows.append(
            {
                "variant": variant,
                "records": len(group),
                "ok": len(ok_records),
                "failed": len(group) - len(ok_records),
                "mean_samples_per_second": _mean_optional([record.samples_per_second for record in ok_records]),
                "mean_peak_bytes": _mean_optional([record.measured_peak_bytes for record in ok_records]),
                "mean_prediction_ape": _mean_optional(
                    [None if record.selected_prediction_relative_error is None else abs(record.selected_prediction_relative_error) for record in ok_records]
                ),
            }
        )
    return rows


def _write_table_directory(
    table_id: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    status: str,
    source_paths: Sequence[str | Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = output_dir / "source.csv"
    _write_csv(rows, source_csv)
    columns = list(rows[0]) if rows else ["no_rows"]
    _write_json({"schema_version": TABLE_SCHEMA_VERSION, "table_id": table_id, "columns": columns, "row_count": len(rows)}, output_dir / "source.schema.json")
    (output_dir / "table.md").write_text(_markdown_table(rows) + "\n", encoding="utf-8")
    (output_dir / "table.tex").write_text(_latex_table(rows) + "\n", encoding="utf-8")
    (output_dir / "caption.md").write_text(_caption(table_id, status) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "table_id": table_id,
        "status": status,
        "source_paths": [str(path) for path in source_paths],
        "source_checksums": _source_checksums(source_paths),
        "source_csv_sha256": _sha256(source_csv.read_bytes()),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "row_count": len(rows),
    }
    _write_json(provenance, output_dir / "provenance.json")


def _validate_table_dir(table_dir: Path, table_id: str, require_frozen: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = ("table.md", "table.tex", "source.csv", "source.schema.json", "caption.md", "provenance.json")
    if not table_dir.is_dir():
        return {"table_dir": str(table_dir), "ok": False, "errors": ["missing table directory"], "warnings": [], "row_count": None, "status": None}
    for name in required:
        if not (table_dir / name).is_file():
            errors.append(f"missing {name}")
    row_count = None
    if (table_dir / "source.csv").is_file():
        with (table_dir / "source.csv").open(encoding="utf-8", newline="") as handle:
            row_count = len(list(csv.DictReader(handle)))
    schema = _load_json(table_dir / "source.schema.json", errors)
    provenance = _load_json(table_dir / "provenance.json", errors)
    status = None
    if schema:
        if schema.get("table_id") != table_id:
            errors.append("source.schema.json table_id mismatch")
        if row_count is not None and schema.get("row_count") != row_count:
            errors.append("source.schema.json row_count mismatch")
    if provenance:
        status = provenance.get("status")
        if provenance.get("table_id") != table_id:
            errors.append("provenance.json table_id mismatch")
        if row_count is not None and provenance.get("row_count") != row_count:
            errors.append("provenance.json row_count mismatch")
        if (table_dir / "source.csv").is_file() and provenance.get("source_csv_sha256") != _sha256((table_dir / "source.csv").read_bytes()):
            errors.append("source_csv_sha256 mismatch")
        if require_frozen and status != "frozen":
            errors.append("expected frozen table status")
    if row_count == 0:
        warnings.append("source.csv has no data rows")
    return {"table_dir": str(table_dir), "ok": not errors, "errors": errors, "warnings": warnings, "row_count": row_count, "status": status}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = list(rows[0]) if rows else ["no_rows"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0]) if rows else ["no_rows"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _latex_table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0]) if rows else ["no_rows"]
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(_escape_latex(column) for column in columns) + r" \\ \hline"]
    for row in rows:
        lines.append(" & ".join(_escape_latex(_cell(row.get(column))) for column in columns) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _caption(table_id: str, status: str) -> str:
    return f"# {table_id} Caption\n\nStatus: `{status}`.\n\nDraft table generated from experiment records; final text must wait for frozen EV evidence."


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _mean(values: Sequence[float | int]) -> float | None:
    return None if not values else sum(float(value) for value in values) / len(values)


def _mean_optional(values: Sequence[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return _mean(present)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _phase_accuracy(records: Sequence[ExperimentRecord]) -> float | None:
    comparable = [record for record in records if record.selected_peak_phase_match is not None]
    if not comparable:
        return None
    return sum(1 for record in comparable if record.selected_peak_phase_match) / len(comparable)


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid {path.name}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return None
    return payload


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_checksums(paths: Sequence[str | Path]) -> dict[str, str]:
    checksums = {}
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            checksums[str(path)] = _sha256(path.read_bytes())
    return checksums


def _escape_latex(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
