from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from peakaware.experiments import ExperimentRecord
from peakaware.workload_manifest import canonical_json


FIGURE_SCHEMA_VERSION = "0.1"
PUBLICATION_FIGURE_IDS = ("F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")

_STRATEGY_STYLE = {
    "all_save": ("#000000", "o"),
    "eager_all_save": ("#000000", "o"),
    "block_checkpoint": ("#E69F00", "s"),
    "block_ac": ("#E69F00", "s"),
    "pytorch_sac": ("#CC79A7", "P"),
    "sac": ("#CC79A7", "P"),
    "torch_min_cut": ("#0072B2", "^"),
    "pytorch_min_cut": ("#0072B2", "^"),
    "peakaware": ("#009E73", "D"),
    "selected": ("#009E73", "D"),
}


@dataclass(frozen=True)
class FigureArtifact:
    figure_id: str
    output_dir: Path
    row_count: int
    status: str


@dataclass(frozen=True)
class FigureValidationResult:
    figure_dir: Path
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int | None
    status: str | None


def build_publication_figures(
    records: Sequence[ExperimentRecord],
    output_root: str | Path,
    *,
    status: str = "draft",
    ev_ids: Sequence[str] = (),
    source_paths: Sequence[str | Path] = (),
    command: str | None = None,
) -> tuple[FigureArtifact, ...]:
    """Generate draft/provisional publication figure directories from experiment records.

    The generated artifacts intentionally do not claim ``frozen`` status unless the
    caller passes it explicitly.  They are derived from records and include a
    figure-level source CSV so later evidence freezing can audit every plotted
    point.
    """

    if status not in {"draft", "provisional", "frozen"}:
        raise ValueError("status must be draft, provisional, or frozen")
    if status == "frozen":
        if not ev_ids:
            raise ValueError("frozen figures must reference at least one EV-* evidence id")
        missing_sources = [str(path) for path in source_paths if not Path(path).is_file()]
        if missing_sources:
            raise ValueError(f"frozen figures require checksum-able source files: {missing_sources}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure_builders: dict[str, Callable[[Sequence[ExperimentRecord]], list[dict[str, Any]]]] = {
        "F2": _rows_f2_pareto,
        "F3": _rows_f3_budget_feasibility,
        "F4": _rows_f4_baseline_comparison,
        "F5": _rows_f5_prediction_parity,
        "F6": _rows_f6_diagnostic_waterfall,
        "F7": _rows_f7_phase_peaks,
        "F8": _rows_f8_topk_ranking,
        "F9": _rows_f9_optimization_cost,
        "F10": _rows_f10_ablation,
    }
    artifacts: list[FigureArtifact] = []
    for figure_id in PUBLICATION_FIGURE_IDS:
        rows = figure_builders[figure_id](records)
        figure_dir = root / _figure_dir_name(figure_id)
        _write_figure_directory(
            figure_id=figure_id,
            rows=rows,
            output_dir=figure_dir,
            status=status,
            ev_ids=tuple(ev_ids),
            source_paths=tuple(str(path) for path in source_paths),
            command=command,
        )
        artifacts.append(FigureArtifact(figure_id, figure_dir, len(rows), status))
    return tuple(artifacts)


def validate_publication_figure_artifacts(
    figure_root: str | Path,
    *,
    require_frozen: bool = False,
    expected_figure_ids: Sequence[str] = PUBLICATION_FIGURE_IDS,
) -> dict[str, Any]:
    root = Path(figure_root)
    results: list[FigureValidationResult] = []
    expected_dirs = {_figure_dir_name(figure_id): figure_id for figure_id in expected_figure_ids}
    for dirname, figure_id in expected_dirs.items():
        figure_dir = root / dirname
        results.append(_validate_figure_dir(figure_dir, figure_id, require_frozen=require_frozen))
    missing = [str(result.figure_dir) for result in results if "missing figure directory" in result.errors]
    error_count = sum(len(result.errors) for result in results)
    warning_count = sum(len(result.warnings) for result in results)
    return {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_root": str(root),
        "ok": error_count == 0,
        "require_frozen": require_frozen,
        "expected_figure_ids": list(expected_figure_ids),
        "missing_directories": missing,
        "error_count": error_count,
        "warning_count": warning_count,
        "figures": [
            {
                "figure_dir": str(result.figure_dir),
                "ok": result.ok,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "row_count": result.row_count,
                "status": result.status,
            }
            for result in results
        ],
    }


def _validate_figure_dir(
    figure_dir: Path,
    figure_id: str,
    *,
    require_frozen: bool,
) -> FigureValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    row_count: int | None = None
    status: str | None = None
    required = (
        "figure.svg",
        "source.csv",
        "source.schema.json",
        "plot_config.json",
        "caption.md",
        "provenance.json",
        "render.log",
    )
    if not figure_dir.is_dir():
        return FigureValidationResult(
            figure_dir,
            False,
            ("missing figure directory",),
            (),
            None,
            None,
        )
    for name in required:
        if not (figure_dir / name).is_file():
            errors.append(f"missing {name}")
    if (figure_dir / "figure.svg").is_file():
        try:
            ET.parse(figure_dir / "figure.svg")
        except ET.ParseError as exc:
            errors.append(f"invalid figure.svg: {exc}")
    source_rows: list[dict[str, str]] = []
    if (figure_dir / "source.csv").is_file():
        with (figure_dir / "source.csv").open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        row_count = len(source_rows)
    schema = _load_json_for_validation(figure_dir / "source.schema.json", errors)
    provenance = _load_json_for_validation(figure_dir / "provenance.json", errors)
    plot_config = _load_json_for_validation(figure_dir / "plot_config.json", errors)
    if schema:
        if schema.get("figure_id") != figure_id:
            errors.append("source.schema.json figure_id mismatch")
        if row_count is not None and schema.get("row_count") != row_count:
            errors.append("source.schema.json row_count mismatch")
    if plot_config and plot_config.get("figure_id") != figure_id:
        errors.append("plot_config.json figure_id mismatch")
    if provenance:
        status = provenance.get("status")
        if provenance.get("figure_id") != figure_id:
            errors.append("provenance.json figure_id mismatch")
        if row_count is not None and provenance.get("row_count") != row_count:
            errors.append("provenance.json row_count mismatch")
        if (figure_dir / "source.csv").is_file():
            actual_checksum = _sha256((figure_dir / "source.csv").read_bytes())
            if provenance.get("source_csv_sha256") != actual_checksum:
                errors.append("source_csv_sha256 mismatch")
        if require_frozen and status != "frozen":
            errors.append("expected frozen figure status")
        if status == "frozen":
            if not provenance.get("ev_ids"):
                errors.append("frozen figure missing ev_ids")
            if not provenance.get("source_checksums"):
                errors.append("frozen figure missing source_checksums")
            if not provenance.get("generator_commit"):
                errors.append("frozen figure missing generator_commit")
            if provenance.get("environment_lock_checksum") is None:
                warnings.append("frozen figure missing environment_lock_checksum")
    if row_count == 0:
        warnings.append("source.csv has no data rows")
    return FigureValidationResult(
        figure_dir,
        not errors,
        tuple(errors),
        tuple(warnings),
        row_count,
        None if status is None else str(status),
    )


def _load_json_for_validation(path: Path, errors: list[str]) -> dict[str, Any] | None:
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


def _figure_dir_name(figure_id: str) -> str:
    names = {
        "F2": "F2_pareto",
        "F3": "F3_budget_feasibility",
        "F4": "F4_baseline_comparison",
        "F5": "F5_prediction_parity",
        "F6": "F6_diagnostic_waterfall",
        "F7": "F7_phase_peaks",
        "F8": "F8_topk_ranking",
        "F9": "F9_optimization_cost",
        "F10": "F10_ablation",
    }
    return names[figure_id]


def _record_backend(record: ExperimentRecord) -> str:
    config = record.config_fingerprint or {}
    compile_backend = config.get("compile_backend")
    capture_backend = config.get("capture_backend")
    if capture_backend is not None and compile_backend is not None:
        return f"{capture_backend}/{compile_backend}"
    if compile_backend is not None:
        return str(compile_backend)
    if capture_backend is not None:
        return str(capture_backend)
    if config.get("enable_inductor"):
        return "inductor"
    if config.get("enable_compile"):
        return "aot_eager"
    return "eager"


def _workload_key(record: ExperimentRecord) -> str:
    return f"{record.task_name}:mb{record.microbatch_size}:{_record_backend(record)}"


def _budget_ratio(record: ExperimentRecord) -> float | None:
    if record.all_save_measured_peak_bytes in (None, 0):
        return None
    return record.budget_bytes / record.all_save_measured_peak_bytes


def _strategy_label(plan_id: str | None) -> str:
    if not plan_id:
        return "unknown"
    aliases = {
        "all_save": "all_save",
        "torch_min_cut": "torch_min_cut",
        "pytorch_min_cut": "pytorch_min_cut",
        "block_checkpoint": "block_checkpoint",
        "block_ac": "block_ac",
        "pytorch_sac": "pytorch_sac",
        "sac": "sac",
        "peakaware": "peakaware",
    }
    return aliases.get(plan_id, plan_id)


def _iter_measured_plan_rows(record: ExperimentRecord) -> Iterable[dict[str, Any]]:
    for plan in record.measured_plan_results:
        yield dict(plan)
    if record.status == "ok" and record.selected_plan_id and record.measured_peak_bytes is not None:
        existing = {str(row.get("plan_id")) for row in record.measured_plan_results}
        if record.selected_plan_id not in existing:
            yield {
                "plan_id": record.selected_plan_id,
                "estimated_peak_bytes": record.selected_estimated_peak_bytes,
                "measured_peak_bytes": record.measured_peak_bytes,
                "measured_peak_phase": record.measured_peak_phase,
                "measured_step_us": record.measured_step_us,
                "correctness_passed": True,
                "calibrated_estimated_peak_bytes": None
                if record.selected_calibrated_prediction_error_bytes is None
                or record.measured_peak_bytes is None
                else record.measured_peak_bytes - record.selected_calibrated_prediction_error_bytes,
            }


def _rows_f2_pareto(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        all_save_peak = record.all_save_measured_peak_bytes
        all_save_step = record.all_save_measured_step_us
        if not all_save_peak or not all_save_step:
            continue
        for plan in _iter_measured_plan_rows(record):
            peak = plan.get("measured_peak_bytes")
            step = plan.get("measured_step_us")
            if peak is None or step is None:
                continue
            step = float(step)
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "budget_bytes": record.budget_bytes,
                    "budget_ratio": _budget_ratio(record),
                    "strategy": _strategy_label(plan.get("plan_id")),
                    "measured_peak_bytes": int(peak),
                    "normalized_peak": int(peak) / int(all_save_peak),
                    "measured_step_us": step,
                    "normalized_throughput": float(all_save_step) / max(step, 1.0),
                    "status": record.status,
                    "correctness_passed": plan.get("correctness_passed"),
                }
            )
    return rows


def _rows_f3_budget_feasibility(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        plan_rows = list(_iter_measured_plan_rows(record))
        if not plan_rows:
            plan_rows = [{"plan_id": record.selected_plan_id, "measured_peak_bytes": record.measured_peak_bytes}]
        for plan in plan_rows:
            measured_peak = plan.get("measured_peak_bytes")
            success = (
                record.status == "ok"
                and isinstance(measured_peak, (int, float))
                and float(measured_peak) <= record.budget_bytes
                and plan.get("correctness_passed", True) is not False
            )
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "budget_bytes": record.budget_bytes,
                    "budget_ratio": _budget_ratio(record),
                    "strategy": _strategy_label(plan.get("plan_id")),
                    "all_save_measured_peak_bytes": record.all_save_measured_peak_bytes,
                    "measured_peak_bytes": measured_peak,
                    "status": record.status,
                    "feasibility_status": record.feasibility_status,
                    "budget_satisfied": success,
                }
            )
    return rows


def _rows_f4_baseline_comparison(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if (
            record.status != "ok"
            or record.selected_plan_id is None
            or record.measured_peak_bytes is None
            or record.measured_step_us is None
        ):
            continue
        selected_peak = int(record.measured_peak_bytes)
        selected_step = float(record.measured_step_us)
        if selected_step <= 0:
            continue
        for plan in _iter_measured_plan_rows(record):
            plan_id = plan.get("plan_id")
            baseline_peak = plan.get("measured_peak_bytes")
            baseline_step = plan.get("measured_step_us")
            if plan_id is None or str(plan_id) == record.selected_plan_id:
                continue
            if not isinstance(baseline_peak, (int, float)) or not isinstance(baseline_step, (int, float)):
                continue
            if baseline_peak <= 0 or baseline_step <= 0:
                continue
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "matched_key": f"{_workload_key(record)}:{record.budget_bytes}:{record.variant_name}",
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "budget_bytes": record.budget_bytes,
                    "budget_ratio": _budget_ratio(record),
                    "variant": record.variant_name,
                    "selected_strategy": _strategy_label(record.selected_plan_id),
                    "baseline_strategy": _strategy_label(str(plan_id)),
                    "selected_peak_bytes": selected_peak,
                    "baseline_peak_bytes": int(baseline_peak),
                    "peak_ratio_vs_baseline": selected_peak / float(baseline_peak),
                    "peak_delta_vs_baseline_bytes": selected_peak - int(baseline_peak),
                    "selected_step_us": selected_step,
                    "baseline_step_us": float(baseline_step),
                    "speedup_vs_baseline": float(baseline_step) / selected_step,
                    "correctness_passed": plan.get("correctness_passed"),
                }
            )
    return rows


def _rows_f5_prediction_parity(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for plan in _iter_measured_plan_rows(record):
            measured = plan.get("measured_peak_bytes")
            estimated = plan.get("estimated_peak_bytes")
            calibrated = plan.get("calibrated_estimated_peak_bytes")
            if measured is None:
                continue
            for estimate_kind, estimate in (("raw", estimated), ("calibrated", calibrated)):
                if estimate is None:
                    continue
                measured_value = int(measured)
                estimate_value = int(estimate)
                rows.append(
                    {
                        "config_fingerprint": canonical_json(record.config_fingerprint),
                        "workload": _workload_key(record),
                        "task_name": record.task_name,
                        "backend": _record_backend(record),
                        "strategy": _strategy_label(plan.get("plan_id")),
                        "estimate_kind": estimate_kind,
                        "estimated_peak_bytes": estimate_value,
                        "measured_peak_bytes": measured_value,
                        "signed_error_bytes": estimate_value - measured_value,
                        "ape": None if measured_value == 0 else abs(estimate_value - measured_value) / measured_value,
                        "measured_peak_phase": plan.get("measured_peak_phase"),
                    }
                )
    return rows


def _rows_f7_phase_peaks(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status != "ok":
            continue
        plan_rows = list(_iter_measured_plan_rows(record))
        for plan in plan_rows:
            metrics = plan.get("phase_metrics") or {}
            phases = (
                ("FW", metrics.get("fw_peak_bytes")),
                ("BW", metrics.get("bw_peak_bytes")),
                ("OPT", metrics.get("optimizer_peak_bytes")),
            )
            if all(peak is None for _, peak in phases) and plan.get("plan_id") == record.selected_plan_id:
                phases = (
                    ("FW", record.measured_fw_peak_bytes),
                    ("BW", record.measured_bw_peak_bytes),
                    ("OPT", record.measured_optimizer_peak_bytes),
                )
            for phase, peak in phases:
                if peak is None:
                    continue
                rows.append(
                    {
                        "config_fingerprint": canonical_json(record.config_fingerprint),
                        "workload": _workload_key(record),
                        "task_name": record.task_name,
                        "backend": _record_backend(record),
                        "budget_bytes": record.budget_bytes,
                        "budget_ratio": _budget_ratio(record),
                        "strategy": _strategy_label(plan.get("plan_id")),
                        "phase": phase,
                        "phase_peak_bytes": int(peak),
                        "measured_peak_bytes": plan.get("measured_peak_bytes"),
                        "measured_peak_phase": plan.get("measured_peak_phase"),
                    }
                )
    return rows


def _rows_f6_diagnostic_waterfall(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        baseline = record.baseline_estimated_peak_bytes or record.all_save_measured_peak_bytes
        previous: int | None = None
        for item in record.diagnostic_counterfactuals:
            level = item.get("level")
            peak = item.get("candidate_peak_bytes")
            if level is None or peak is None:
                continue
            peak_value = int(peak)
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "strategy": _strategy_label(record.selected_plan_id),
                    "level": level,
                    "candidate_peak_bytes": peak_value,
                    "delta_from_previous_bytes": None if previous is None else peak_value - previous,
                    "gain_vs_baseline_bytes": None if baseline is None else int(baseline) - peak_value,
                    "candidate_peak_phase": item.get("candidate_peak_phase"),
                    "diagnostic_primary_cause": record.diagnostic_primary_cause,
                    "counterfactual_status": item.get("status"),
                    "confidence": item.get("confidence"),
                }
            )
            previous = peak_value
    return rows


def _rows_f8_topk_ranking(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        candidates = [
            plan
            for plan in _iter_measured_plan_rows(record)
            if plan.get("estimated_peak_bytes") is not None
            and plan.get("measured_peak_bytes") is not None
            and plan.get("measured_step_us") is not None
        ]
        estimated_order = {
            str(plan.get("plan_id")): rank
            for rank, plan in enumerate(
                sorted(candidates, key=lambda plan: (int(plan["estimated_peak_bytes"]), float(plan["measured_step_us"]))),
                start=1,
            )
        }
        measured_order = {
            str(plan.get("plan_id")): rank
            for rank, plan in enumerate(
                sorted(candidates, key=lambda plan: (int(plan["measured_peak_bytes"]), float(plan["measured_step_us"]))),
                start=1,
            )
        }
        best_measured_step = min((float(plan["measured_step_us"]) for plan in candidates), default=None)
        for plan in candidates:
            plan_id = str(plan.get("plan_id"))
            step = float(plan["measured_step_us"])
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "budget_bytes": record.budget_bytes,
                    "strategy": _strategy_label(plan_id),
                    "candidate_id": plan_id,
                    "estimated_rank": estimated_order[plan_id],
                    "measured_rank": measured_order[plan_id],
                    "rank_error": estimated_order[plan_id] - measured_order[plan_id],
                    "estimated_peak_bytes": int(plan["estimated_peak_bytes"]),
                    "measured_peak_bytes": int(plan["measured_peak_bytes"]),
                    "measured_step_us": step,
                    "normalized_step_regret": None
                    if best_measured_step in (None, 0)
                    else step / best_measured_step - 1.0,
                    "top_k": record.config_fingerprint.get("top_k") if isinstance(record.config_fingerprint, dict) else None,
                }
            )
    return rows


def _rows_f9_optimization_cost(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = (
        ("capture", "optimization_capture_us"),
        ("ir_build", "optimization_ir_build_us"),
        ("analysis", "optimization_analysis_us"),
        ("executor_build", "optimization_executor_build_us"),
        ("candidate_validation_measurement", "optimization_candidate_validation_measurement_us"),
    )
    for record in records:
        base = {
            "config_fingerprint": canonical_json(record.config_fingerprint),
            "workload": _workload_key(record),
            "task_name": record.task_name,
            "backend": _record_backend(record),
            "strategy": _strategy_label(record.selected_plan_id),
            "budget_bytes": record.budget_bytes,
            "optimization_total_us": record.optimization_total_us,
            "cache_hit_rate": record.cache_hit_rate,
            "cache_total_hits": record.cache_total_hits,
            "cache_total_misses": record.cache_total_misses,
            "amortization_steps": record.optimization_amortization_steps,
        }
        for stage, attr in stages:
            value = getattr(record, attr)
            if value is None:
                continue
            rows.append({**base, "stage": stage, "stage_time_us": float(value)})
    return rows


def _rows_f10_ablation(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault((_workload_key(record), record.budget_bytes, _record_backend(record)), []).append(record)
    for (_workload, _budget, _backend), group in grouped.items():
        full = next((record for record in group if record.variant_name in {"full", "diagnostic_hints_on"}), None)
        if full is None:
            full = next((record for record in group if record.status == "ok"), None)
        for record in group:
            rows.append(
                {
                    "config_fingerprint": canonical_json(record.config_fingerprint),
                    "workload": _workload_key(record),
                    "task_name": record.task_name,
                    "backend": _record_backend(record),
                    "budget_bytes": record.budget_bytes,
                    "variant": record.variant_name,
                    "strategy": _strategy_label(record.selected_plan_id),
                    "budget_satisfied": record.status == "ok"
                    and record.measured_peak_bytes is not None
                    and record.measured_peak_bytes <= record.budget_bytes,
                    "throughput": record.samples_per_second,
                    "prediction_ape": None
                    if record.selected_prediction_relative_error is None
                    else abs(record.selected_prediction_relative_error),
                    "optimization_total_us": record.optimization_total_us,
                    "candidate_count": record.candidate_count,
                    "throughput_delta_vs_full": None
                    if full is None
                    or full.samples_per_second is None
                    or record.samples_per_second is None
                    else record.samples_per_second - full.samples_per_second,
                    "peak_delta_vs_full_bytes": None
                    if full is None
                    or full.measured_peak_bytes is None
                    or record.measured_peak_bytes is None
                    else record.measured_peak_bytes - full.measured_peak_bytes,
                }
            )
    return rows


def _write_figure_directory(
    *,
    figure_id: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    status: str,
    ev_ids: Sequence[str],
    source_paths: Sequence[str],
    command: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.csv"
    _write_csv(rows, source_path)
    schema = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_id": figure_id,
        "columns": list(rows[0]) if rows else [],
        "row_count": len(rows),
    }
    _write_json(schema, output_dir / "source.schema.json")
    _write_json(_plot_config(figure_id, status), output_dir / "plot_config.json")
    (output_dir / "caption.md").write_text(_caption(figure_id, status) + "\n", encoding="utf-8")
    (output_dir / "figure.svg").write_text(_render_svg(figure_id, rows, status), encoding="utf-8")
    (output_dir / "render.log").write_text(
        f"figure_id={figure_id}\nstatus={status}\nrow_count={len(rows)}\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_id": figure_id,
        "ev_ids": list(ev_ids),
        "source_paths": list(source_paths),
        "source_checksums": _source_checksums(source_paths),
        "source_csv_sha256": _sha256(source_path.read_bytes()),
        "generator_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "environment_lock_checksum": None,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "command": command,
        "row_count": len(rows),
    }
    _write_json(provenance, output_dir / "provenance.json")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = list(rows[0]) if rows else ["no_rows"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_checksums(paths: Sequence[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.exists() and path.is_file():
            checksums[str(path)] = _sha256(path.read_bytes())
    return checksums


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _plot_config(figure_id: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_id": figure_id,
        "status": status,
        "renderer": "peakaware.publication.figures.svg",
        "palette": {name: {"color": color, "marker": marker} for name, (color, marker) in _STRATEGY_STYLE.items()},
    }


def _caption(figure_id: str, status: str) -> str:
    captions = {
        "F2": "F-2 draft Pareto plot: normalized allocated peak and normalized throughput are computed within each workload against matched all-save measurements.",
        "F3": "F-3 draft budget feasibility plot: a workload cell is satisfied only when measured allocated peak does not exceed the registered budget.",
        "F4": "F-4 draft baseline comparison plot: selected plans are paired against measured baseline candidates with peak ratio and speedup exported per matched cell.",
        "F5": "F-5 draft prediction parity plot: raw and calibrated estimated peaks are shown against measured allocated peak with a y=x reference.",
        "F6": "F-6 draft diagnostic waterfall: D0-D5 counterfactual peaks are exported with per-level deltas and diagnostic causes.",
        "F7": "F-7 draft phase peak plot: FW, BW, and optimizer phase peaks are shown separately and are not summed.",
        "F8": "F-8 draft Top-K ranking plot: estimated and measured ranks are compared for each measured candidate.",
        "F9": "F-9 draft optimization cost plot: optimization stages, cache reuse, and amortization metadata are exported per workload.",
        "F10": "F-10 draft ablation plot: variants are paired against the full/default row within workload and budget cells.",
    }
    return f"# {figure_id} Caption\n\nStatus: `{status}`.\n\n{captions[figure_id]}"


def _render_svg(figure_id: str, rows: list[dict[str, Any]], status: str) -> str:
    if figure_id == "F2":
        return _scatter_svg(
            title="F-2 normalized peak-throughput Pareto",
            rows=rows,
            x_key="normalized_peak",
            y_key="normalized_throughput",
            group_key="strategy",
            status=status,
            x_label="normalized allocated peak",
            y_label="normalized throughput",
            reference=(1.0, 1.0),
        )
    if figure_id == "F3":
        return _feasibility_svg(rows, status)
    if figure_id == "F4":
        return _scatter_svg(
            title="F-4 selected vs measured baselines",
            rows=rows,
            x_key="peak_ratio_vs_baseline",
            y_key="speedup_vs_baseline",
            group_key="baseline_strategy",
            status=status,
            x_label="selected peak / baseline peak",
            y_label="selected speedup vs baseline",
            reference=(1.0, 1.0),
        )
    if figure_id == "F5":
        return _scatter_svg(
            title="F-5 estimated vs measured peak",
            rows=rows,
            x_key="measured_peak_bytes",
            y_key="estimated_peak_bytes",
            group_key="estimate_kind",
            status=status,
            x_label="measured peak bytes",
            y_label="estimated peak bytes",
            parity=True,
        )
    if figure_id == "F6":
        return _level_svg(rows, status, "F-6 D0-D5 diagnostic peaks", "level", "candidate_peak_bytes")
    if figure_id == "F7":
        return _phase_svg(rows, status)
    if figure_id == "F8":
        return _scatter_svg(
            title="F-8 estimated vs measured rank",
            rows=rows,
            x_key="estimated_rank",
            y_key="measured_rank",
            group_key="strategy",
            status=status,
            x_label="estimated rank",
            y_label="measured rank",
            parity=True,
        )
    if figure_id == "F9":
        return _level_svg(rows, status, "F-9 optimization cost stages", "stage", "stage_time_us")
    if figure_id == "F10":
        return _scatter_svg(
            title="F-10 ablation deltas",
            rows=rows,
            x_key="peak_delta_vs_full_bytes",
            y_key="throughput_delta_vs_full",
            group_key="variant",
            status=status,
            x_label="peak delta vs full bytes",
            y_label="throughput delta vs full",
            reference=(0.0, 0.0),
        )
    raise ValueError(f"unsupported figure_id: {figure_id}")


def _finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2.0
    return start + (value - low) * (end - start) / (high - low)


def _scatter_svg(
    *,
    title: str,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
    status: str,
    x_label: str,
    y_label: str,
    reference: tuple[float, float] | None = None,
    parity: bool = False,
) -> str:
    width, height = 760, 500
    left, right, top, bottom = 90, 35, 70, 90
    x_values = _finite_values(rows, x_key)
    y_values = _finite_values(rows, y_key)
    if parity:
        both = x_values + y_values
        x_values = y_values = both
    x_low, x_high = _bounds(x_values)
    y_low, y_high = _bounds(y_values)
    marks: list[str] = []
    if parity:
        x1 = _scale(x_low, x_low, x_high, left, width - right)
        y1 = _scale(x_low, y_low, y_high, height - bottom, top)
        x2 = _scale(x_high, x_low, x_high, left, width - right)
        y2 = _scale(x_high, y_low, y_high, height - bottom, top)
        marks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#777" stroke-dasharray="5 4"/>')
    if reference is not None:
        x_ref = _scale(reference[0], x_low, x_high, left, width - right)
        y_ref = _scale(reference[1], y_low, y_high, height - bottom, top)
        marks.append(f'<line x1="{x_ref:.1f}" y1="{top}" x2="{x_ref:.1f}" y2="{height-bottom}" stroke="#999" stroke-dasharray="4 4"/>')
        marks.append(f'<line x1="{left}" y1="{y_ref:.1f}" x2="{width-right}" y2="{y_ref:.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        color = _style_color(str(row.get(group_key, "")))
        cx = _scale(float(x), x_low, x_high, left, width - right)
        cy = _scale(float(y), y_low, y_high, height - bottom, top)
        label = _escape(str(row.get(group_key, "")))
        marks.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"><title>{label}</title></circle>')
    return _svg_frame(width, height, title, status, x_label, y_label, marks, x_low, x_high, y_low, y_high)


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.1, 1.0)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _style_color(key: str) -> str:
    if key == "raw":
        return "#56B4E9"
    if key == "calibrated":
        return "#D55E00"
    return _STRATEGY_STYLE.get(key, ("#666666", "o"))[0]


def _svg_frame(
    width: int,
    height: int,
    title: str,
    status: str,
    x_label: str,
    y_label: str,
    marks: Sequence[str],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> str:
    axis = [
        f'<line x1="90" y1="{height-90}" x2="{width-35}" y2="{height-90}" stroke="#111"/>',
        f'<line x1="90" y1="{height-90}" x2="90" y2="70" stroke="#111"/>',
        f'<text x="{width/2:.1f}" y="{height-35}" text-anchor="middle" class="axis">{_escape(x_label)}</text>',
        f'<text x="22" y="{height/2:.1f}" transform="rotate(-90 22 {height/2:.1f})" text-anchor="middle" class="axis">{_escape(y_label)}</text>',
        f'<text x="90" y="{height-68}" class="tick">{x_low:.3g}</text>',
        f'<text x="{width-35}" y="{height-68}" text-anchor="end" class="tick">{x_high:.3g}</text>',
        f'<text x="82" y="{height-90}" text-anchor="end" class="tick">{y_low:.3g}</text>',
        f'<text x="82" y="74" text-anchor="end" class="tick">{y_high:.3g}</text>',
    ]
    body = "\n  ".join(axis + list(marks))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 18px Arial, sans-serif; fill: #111; }}
    .subtitle {{ font: 400 12px Arial, sans-serif; fill: #666; }}
    .axis {{ font: 400 12px Arial, sans-serif; fill: #222; }}
    .tick {{ font: 400 10px Arial, sans-serif; fill: #555; }}
  </style>
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="32" y="34" class="title">{_escape(title)}</text>
  <text x="32" y="54" class="subtitle">status: {status}; generated from source.csv rows</text>
  {body}
</svg>
'''


def _feasibility_svg(rows: list[dict[str, Any]], status: str) -> str:
    width, height = 760, 500
    left, right, top, bottom = 90, 35, 70, 90
    ratios = sorted({round(float(row["budget_ratio"]), 3) for row in rows if isinstance(row.get("budget_ratio"), (int, float))})
    strategies = sorted({str(row.get("strategy")) for row in rows})
    aggregated: list[dict[str, Any]] = []
    for strategy in strategies:
        for ratio in ratios:
            matched = [row for row in rows if str(row.get("strategy")) == strategy and isinstance(row.get("budget_ratio"), (int, float)) and round(float(row["budget_ratio"]), 3) == ratio]
            if not matched:
                continue
            success = sum(1 for row in matched if row.get("budget_satisfied") is True)
            aggregated.append({"strategy": strategy, "ratio": ratio, "rate": success / len(matched)})
    marks: list[str] = []
    x_low, x_high = _bounds(ratios)
    y_low, y_high = 0.0, 1.0
    for row in aggregated:
        cx = _scale(float(row["ratio"]), x_low, x_high, left, width - right)
        cy = _scale(float(row["rate"]), y_low, y_high, height - bottom, top)
        color = _style_color(str(row["strategy"]))
        marks.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"><title>{_escape(str(row["strategy"]))}: {row["rate"]:.2f}</title></circle>')
    return _svg_frame(width, height, "F-3 budget feasibility", status, "budget / all-save measured peak", "satisfied workload-cell rate", marks, x_low, x_high, y_low, y_high)


def _phase_svg(rows: list[dict[str, Any]], status: str) -> str:
    width, height = 860, 520
    phases = ("FW", "BW", "OPT")
    grouped: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        strategy = str(row.get("strategy"))
        phase = str(row.get("phase"))
        peak = row.get("phase_peak_bytes")
        if phase in phases and isinstance(peak, int):
            grouped.setdefault(strategy, {}).setdefault(phase, []).append(peak)
    medians: list[dict[str, Any]] = []
    for strategy, phase_values in sorted(grouped.items()):
        for phase in phases:
            values = phase_values.get(phase, [])
            if values:
                medians.append({"strategy": strategy, "phase": phase, "peak": _median(values)})
    y_values = [float(row["peak"]) for row in medians]
    y_low, y_high = _bounds(y_values)
    left, top, bottom = 90, 70, 95
    plot_width = width - left - 45
    slot_count = max(len(medians), 1)
    bar_w = max(12, min(34, plot_width / slot_count * 0.65))
    marks: list[str] = []
    for index, row in enumerate(medians):
        x = left + (index + 0.5) * plot_width / slot_count
        y = _scale(float(row["peak"]), y_low, y_high, height - bottom, top)
        color = _style_color(str(row["strategy"]))
        marks.append(f'<rect x="{x - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height-bottom-y:.1f}" fill="{color}"><title>{_escape(str(row["strategy"]))} {row["phase"]}: {row["peak"]}</title></rect>')
        marks.append(f'<text x="{x:.1f}" y="{height-72}" text-anchor="middle" class="tick">{_escape(str(row["phase"]))}</text>')
    return _svg_frame(width, height, "F-7 phase peaks", status, "phase medians by strategy", "allocated peak bytes", marks, 0.0, float(slot_count), y_low, y_high)


def _level_svg(
    rows: list[dict[str, Any]],
    status: str,
    title: str,
    category_key: str,
    value_key: str,
) -> str:
    width, height = 860, 520
    grouped: dict[str, list[float]] = {}
    for row in rows:
        category = row.get(category_key)
        value = row.get(value_key)
        if category is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        grouped.setdefault(str(category), []).append(float(value))
    categories = sorted(grouped)
    medians = [(category, _median_float(grouped[category])) for category in categories]
    y_values = [value for _, value in medians]
    y_low, y_high = _bounds(y_values)
    left, top, bottom = 90, 70, 95
    plot_width = width - left - 45
    slot_count = max(len(medians), 1)
    bar_w = max(14, min(42, plot_width / slot_count * 0.62))
    marks: list[str] = []
    for index, (category, value) in enumerate(medians):
        x = left + (index + 0.5) * plot_width / slot_count
        y = _scale(value, y_low, y_high, height - bottom, top)
        marks.append(f'<rect x="{x - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height-bottom-y:.1f}" fill="#56B4E9"><title>{_escape(category)}: {value:.3g}</title></rect>')
        marks.append(f'<text x="{x:.1f}" y="{height-72}" text-anchor="middle" class="tick">{_escape(category)}</text>')
    return _svg_frame(width, height, title, status, category_key, value_key, marks, 0.0, float(slot_count), y_low, y_high)


def _median(values: Sequence[int]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _median_float(values: Sequence[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
