from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from peakaware.api import optimize_training
from peakaware.config import PeakAwareConfig
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import export_plan_artifact_json, summarize_result
from peakaware.search.exact import solve_exact_small_graph
from peakaware.search.plan import plan_identity_key


@dataclass(frozen=True)
class ExperimentCase:
    variant_name: str
    config_fingerprint: dict[str, Any]
    task_name: str
    microbatch_size: int
    budget_bytes: int
    matrix_pass_index: int = 0
    matrix_pass_count: int = 1


@dataclass(frozen=True)
class ExperimentRecord:
    variant_name: str
    config_fingerprint: dict[str, Any]
    task_name: str
    microbatch_size: int
    budget_bytes: int
    status: str
    selected_plan_id: str | None
    selected_plan_key: str | None
    graph_key: str | None
    selected_saved_value_ids: tuple[int, ...]
    selected_effective_saved_value_ids: tuple[int, ...]
    selected_estimated_peak_bytes: int | None
    baseline_plan_id: str | None
    baseline_estimated_peak_bytes: int | None
    selected_estimated_peak_reduction_bytes: int | None
    measured_peak_bytes: int | None
    measured_peak_reserved_bytes: int | None
    measured_budget_headroom_bytes: int | None
    all_save_measured_peak_bytes: int | None
    all_save_measured_step_us: float | None
    selected_measured_peak_reduction_vs_all_save_bytes: int | None
    selected_step_time_delta_vs_all_save_us: float | None
    selected_samples_per_second_speedup_vs_all_save: float | None
    measured_step_us: float | None
    measurement_repeats: int | None
    measurement_warmup_steps: int | None
    samples_per_second: float | None
    feasibility_status: str | None
    baseline_peak_phase: str | None
    selected_peak_phase: str | None
    measured_peak_phase: str | None
    selected_peak_phase_match: bool | None
    measured_fw_us: float | None
    measured_bw_us: float | None
    measured_optimizer_us: float | None
    measured_fw_peak_bytes: int | None
    measured_bw_peak_bytes: int | None
    measured_optimizer_peak_bytes: int | None
    diagnostic_primary_cause: str | None
    diagnostic_normalized_saved_reduction_bytes: int | None
    diagnostic_realization_gap_bytes: int | None
    diagnostic_total_expectation_gap_bytes: int | None
    diagnostic_counterfactuals: tuple[dict[str, Any], ...]
    measured_candidate_count: int
    measured_plan_results: tuple[dict[str, Any], ...]
    selected_prediction_error_bytes: int | None
    selected_prediction_relative_error: float | None
    selected_calibrated_prediction_error_bytes: int | None
    selected_calibrated_prediction_relative_error: float | None
    selected_feasibility_prediction_match: bool | None
    simulation_accuracy_candidate_count: int
    simulation_accuracy_mean_absolute_error_bytes: float | None
    simulation_accuracy_max_absolute_error_bytes: int | None
    simulation_accuracy_mean_absolute_relative_error: float | None
    simulation_accuracy_within_10_percent_rate: float | None
    cache_total_hits: int
    cache_total_misses: int
    cache_hit_rate: float | None
    cache_layer_hits: dict[str, int]
    cache_layer_misses: dict[str, int]
    optimization_total_us: float | None
    optimization_capture_us: float | None
    optimization_ir_build_us: float | None
    optimization_analysis_us: float | None
    optimization_executor_build_us: float | None
    optimization_candidate_validation_measurement_us: float | None
    optimization_amortization_steps: float | None
    actual_joint_capture_count: int
    candidate_count: int
    fallback_plan_ids: tuple[str, ...]
    diagnostic_hints_enabled: bool | None = None
    diagnostic_hint_count: int = 0
    diagnostic_hint_kinds: tuple[str, ...] = ()
    diagnostic_hint_candidate_match_count: int = 0
    diagnostic_hint_order_changed: bool = False
    diagnostic_hint_order_delta_count: int = 0
    repaired_candidate_count: int = 0
    repair_success_count: int = 0
    feasible_before_repair_count: int = 0
    feasible_after_repair_count: int = 0
    exact_plan_id: str | None = None
    exact_plan_key: str | None = None
    exact_estimated_peak_bytes: int | None = None
    exact_estimated_step_us: float | None = None
    selected_exact_peak_gap_bytes: int | None = None
    exact_error_type: str | None = None
    exact_error_message: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    matrix_pass_index: int = 0
    matrix_pass_count: int = 1


@dataclass(frozen=True)
class ExperimentSummary:
    total_records: int
    ok_records: int
    failed_records: int
    success_rate: float | None
    environment_fingerprint: dict[str, Any]
    variant_counts: dict[str, int]
    budget_violation_count: int
    budget_violation_rate: float | None
    max_feasible_microbatch: int | None
    mean_samples_per_second: float | None
    mean_measured_step_us: float | None
    mean_measured_peak_bytes: float | None
    mean_measured_budget_headroom_bytes: float | None
    mean_measured_peak_reduction_vs_all_save_bytes: float | None
    mean_selected_step_time_delta_vs_all_save_us: float | None
    mean_selected_samples_per_second_speedup_vs_all_save: float | None
    mean_estimated_peak_reduction_bytes: float | None
    aggregate_cache_hit_rate: float | None
    total_cache_hits: int
    total_cache_misses: int
    cache_layer_hits: dict[str, int]
    cache_layer_misses: dict[str, int]
    cache_layer_hit_rates: dict[str, float | None]
    mean_optimization_total_us: float | None
    mean_optimization_capture_us: float | None
    mean_optimization_ir_build_us: float | None
    mean_optimization_analysis_us: float | None
    mean_optimization_executor_build_us: float | None
    mean_optimization_candidate_validation_measurement_us: float | None
    mean_optimization_amortization_steps: float | None
    total_actual_joint_capture_count: int
    selected_prediction_count: int
    mean_selected_prediction_absolute_error_bytes: float | None
    p50_selected_prediction_absolute_error_bytes: float | None
    p90_selected_prediction_absolute_error_bytes: float | None
    max_selected_prediction_absolute_error_bytes: int | None
    mean_selected_prediction_absolute_relative_error: float | None
    p50_selected_prediction_absolute_relative_error: float | None
    p90_selected_prediction_absolute_relative_error: float | None
    mean_selected_calibrated_prediction_absolute_error_bytes: float | None
    p50_selected_calibrated_prediction_absolute_error_bytes: float | None
    p90_selected_calibrated_prediction_absolute_error_bytes: float | None
    mean_selected_calibrated_prediction_absolute_relative_error: float | None
    p50_selected_calibrated_prediction_absolute_relative_error: float | None
    p90_selected_calibrated_prediction_absolute_relative_error: float | None
    simulation_accuracy_candidate_count: int
    mean_simulation_accuracy_absolute_error_bytes: float | None
    p50_simulation_accuracy_absolute_error_bytes: float | None
    p90_simulation_accuracy_absolute_error_bytes: float | None
    max_simulation_accuracy_absolute_error_bytes: int | None
    mean_simulation_accuracy_absolute_relative_error: float | None
    p50_simulation_accuracy_absolute_relative_error: float | None
    p90_simulation_accuracy_absolute_relative_error: float | None
    mean_simulation_accuracy_within_10_percent_rate: float | None
    mean_calibrated_simulation_accuracy_absolute_error_bytes: float | None
    p50_calibrated_simulation_accuracy_absolute_error_bytes: float | None
    p90_calibrated_simulation_accuracy_absolute_error_bytes: float | None
    max_calibrated_simulation_accuracy_absolute_error_bytes: int | None
    mean_calibrated_simulation_accuracy_absolute_relative_error: float | None
    p50_calibrated_simulation_accuracy_absolute_relative_error: float | None
    p90_calibrated_simulation_accuracy_absolute_relative_error: float | None
    calibrated_simulation_accuracy_within_10_percent_rate: float | None
    phase_classification_count: int
    phase_classification_accuracy: float | None
    feasible_classification_count: int
    feasible_classification_accuracy: float | None
    root_cause_counts: dict[str, int]
    selected_peak_phase_counts: dict[str, int]
    measured_peak_phase_counts: dict[str, int]
    diagnostic_hints_enabled_count: int
    diagnostic_hint_count: int
    diagnostic_hint_kind_counts: dict[str, int]
    diagnostic_hint_candidate_match_count: int
    diagnostic_hint_order_changed_count: int
    diagnostic_hint_order_delta_count: int
    repaired_candidate_count: int
    repair_success_count: int
    repair_success_rate: float | None
    feasible_before_repair_count: int
    feasible_after_repair_count: int
    mean_diagnostic_normalized_saved_reduction_bytes: float | None
    mean_diagnostic_realization_gap_bytes: float | None
    mean_diagnostic_total_expectation_gap_bytes: float | None
    exact_success_count: int
    exact_failure_count: int
    mean_selected_exact_peak_gap_bytes: float | None


def _plan_by_id(summary: dict[str, Any], plan_id: str) -> dict[str, Any] | None:
    return next((plan for plan in summary.get("plans", ()) if plan.get("plan_id") == plan_id), None)


def _config_fingerprint(config: PeakAwareConfig, device: str = "cpu") -> dict[str, Any]:
    compile_backend = "inductor" if config.enable_inductor else "aot_eager" if config.enable_compile else "eager"
    return {
        "top_k": config.top_k,
        "device": device,
        "enable_compile": config.enable_compile,
        "enable_inductor": config.enable_inductor,
        "compile_backend": compile_backend,
        "capture_backend": config.capture_backend,
        "selection_objective": config.selection_objective,
        "enable_diagnostic_hints": config.enable_diagnostic_hints,
        "safety_margin_bytes": config.safety_margin_bytes,
        "safety_margin_ratio": config.safety_margin_ratio,
        "measurement_warmup_steps": config.measurement_warmup_steps,
        "measurement_repeats": config.measurement_repeats,
        "isolate_candidate_measurement": config.isolate_candidate_measurement,
        "candidate_worker_timeout_s": config.candidate_worker_timeout_s,
        "profile_db_path": str(config.profile_db_path or "none"),
        "cache_root": str(config.cache_root or "none"),
        "manual_saved_value_ids": tuple(tuple(sorted(item)) for item in config.manual_saved_value_ids),
        "precision": dict(config.precision_fingerprint()),
    }


def _measured_candidate_by_id(summary: dict[str, Any], plan_id: str) -> dict[str, Any] | None:
    return next((item for item in summary.get("measured_candidates", ()) if item.get("plan_id") == plan_id), None)


def _all_save_peak_residual(summary: dict[str, Any]) -> int | None:
    all_save_plan = _plan_by_id(summary, "all_save")
    all_save_measured = _measured_candidate_by_id(summary, "all_save")
    if all_save_plan is None or all_save_measured is None:
        return None
    estimated = all_save_plan.get("estimated_peak_bytes")
    measured = all_save_measured.get("peak_bytes")
    if estimated is None or measured is None:
        return None
    return int(measured) - int(estimated)


def _calibrate_measured_plan_rows(rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    all_save = next((row for row in rows if row.get("plan_id") == "all_save"), None)
    if all_save is None or all_save.get("estimated_peak_bytes") is None or all_save.get("measured_peak_bytes") is None:
        return rows
    residual = int(all_save["measured_peak_bytes"]) - int(all_save["estimated_peak_bytes"])
    calibrated = []
    for row in rows:
        next_row = dict(row)
        if (
            next_row.get("calibrated_prediction_error_bytes") is None
            and next_row.get("estimated_peak_bytes") is not None
            and next_row.get("measured_peak_bytes") is not None
        ):
            estimated = int(next_row["estimated_peak_bytes"]) + residual
            error = int(next_row["measured_peak_bytes"]) - estimated
            next_row["calibrated_estimated_peak_bytes"] = estimated
            next_row["calibrated_prediction_error_bytes"] = error
            next_row["calibrated_prediction_relative_error"] = None if estimated == 0 else error / estimated
        calibrated.append(next_row)
    return tuple(calibrated)


def _measured_plan_results(summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    plans_by_id = {plan["plan_id"]: plan for plan in summary.get("plans", ())}
    all_save_residual = _all_save_peak_residual(summary)
    rows: list[dict[str, Any]] = []
    for measured in summary.get("measured_candidates", ()):
        plan = plans_by_id.get(measured["plan_id"], {})
        prediction = measured.get("prediction_error") or {}
        estimated = plan.get("estimated_peak_bytes")
        measured_peak = measured.get("peak_bytes")
        calibrated_estimated = None
        calibrated_error = None
        calibrated_relative = None
        if estimated is not None and measured_peak is not None and all_save_residual is not None:
            calibrated_estimated = int(estimated) + all_save_residual
            calibrated_error = int(measured_peak) - calibrated_estimated
            calibrated_relative = None if calibrated_estimated == 0 else calibrated_error / calibrated_estimated
        rows.append(
            {
                "plan_id": measured["plan_id"],
                "estimated_peak_bytes": estimated,
                "estimated_step_us": plan.get("estimated_step_us"),
                "estimated_feasible": prediction.get("estimated_feasible"),
                "measured_peak_bytes": measured_peak,
                "measured_step_us": measured.get("step_us"),
                "measured_peak_phase": measured.get("peak_phase"),
                "measured_feasible": prediction.get("measured_feasible"),
                "prediction_error_bytes": prediction.get("error_bytes"),
                "calibrated_estimated_peak_bytes": calibrated_estimated,
                "calibrated_prediction_error_bytes": calibrated_error,
                "calibrated_prediction_relative_error": calibrated_relative,
                "correctness_passed": measured.get("correctness_passed"),
            }
        )
    return _calibrate_measured_plan_rows(tuple(rows))


def _diagnostic_counterfactuals(summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    diagnostic = summary.get("diagnostic") or {}
    rows: list[dict[str, Any]] = []
    for item in diagnostic.get("counterfactuals", ()):
        candidate_peak = item.get("candidate_peak") or {}
        rows.append(
            {
                "level": item.get("level"),
                "status": item.get("status"),
                "candidate_peak_bytes": candidate_peak.get("live_bytes"),
                "candidate_peak_phase": candidate_peak.get("phase"),
                "peak_gain_bytes": item.get("peak_gain_bytes"),
                "confidence": item.get("confidence"),
                "unavailable_reason": item.get("unavailable_reason"),
            }
        )
    return tuple(rows)


def _resolve_experiment_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device}")
    return resolved


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    return value


def _record_success(
    case: ExperimentCase,
    summary: dict[str, Any],
    exact: dict[str, Any] | None = None,
) -> ExperimentRecord:
    measured = summary["measured"]
    phase_metrics = measured.get("phase_metrics", {})
    diagnostic = summary.get("diagnostic")
    expectation = {} if diagnostic is None else diagnostic.get("expectation", {})
    selected_correction = summary.get("topk_correction", {}).get("selected")
    simulation_accuracy = summary.get("topk_correction", {}).get("simulation_accuracy", {})
    search_diagnostics = summary.get("search_diagnostics") or {}
    baseline = _plan_by_id(summary, "all_save")
    measured_baseline = _measured_candidate_by_id(summary, "all_save")
    selected_plan = _plan_by_id(summary, summary["selected_plan_id"])
    baseline_peak = None if baseline is None else baseline.get("peak_snapshot")
    selected_peak = None if selected_plan is None else selected_plan.get("peak_snapshot")
    baseline_peak_bytes = None if baseline is None else int(baseline["estimated_peak_bytes"])
    selected_peak_bytes = int(summary["estimated_peak_bytes"])
    cache = summary.get("cache", {})
    optimization_cost = summary.get("optimization_cost", {})
    step_us = float(measured["step_us"])
    measured_peak_bytes = int(measured["peak_bytes"])
    all_save_peak = None if measured_baseline is None else int(measured_baseline["peak_bytes"])
    all_save_step_us = None if measured_baseline is None else float(measured_baseline["step_us"])
    measured_plan_results = _measured_plan_results(summary)
    selected_measured_plan = next(
        (row for row in measured_plan_results if row.get("plan_id") == summary["selected_plan_id"]),
        None,
    )
    exact = exact or {}
    return ExperimentRecord(
        task_name=case.task_name,
        variant_name=case.variant_name,
        config_fingerprint=case.config_fingerprint,
        microbatch_size=case.microbatch_size,
        budget_bytes=case.budget_bytes,
        status="ok",
        selected_plan_id=summary["selected_plan_id"],
        selected_plan_key=summary["selected_plan_key"],
        graph_key=summary["graph_key"],
        selected_saved_value_ids=tuple(summary["selected_saved_value_ids"]),
        selected_effective_saved_value_ids=tuple(summary["selected_effective_saved_value_ids"]),
        selected_estimated_peak_bytes=selected_peak_bytes,
        baseline_plan_id=None if baseline is None else baseline["plan_id"],
        baseline_estimated_peak_bytes=baseline_peak_bytes,
        selected_estimated_peak_reduction_bytes=None
        if baseline_peak_bytes is None
        else baseline_peak_bytes - selected_peak_bytes,
        measured_peak_bytes=measured_peak_bytes,
        measured_peak_reserved_bytes=int(measured.get("reserved_peak_bytes", 0)),
        measured_budget_headroom_bytes=case.budget_bytes - measured_peak_bytes,
        all_save_measured_peak_bytes=all_save_peak,
        all_save_measured_step_us=all_save_step_us,
        selected_measured_peak_reduction_vs_all_save_bytes=None
        if all_save_peak is None
        else all_save_peak - measured_peak_bytes,
        selected_step_time_delta_vs_all_save_us=None
        if all_save_step_us is None
        else all_save_step_us - step_us,
        selected_samples_per_second_speedup_vs_all_save=None
        if all_save_step_us is None
        else all_save_step_us / max(step_us, 1.0),
        measured_step_us=step_us,
        measurement_repeats=None
        if "measurement_repeats" not in phase_metrics
        else int(phase_metrics["measurement_repeats"]),
        measurement_warmup_steps=None
        if "measurement_warmup_steps" not in phase_metrics
        else int(phase_metrics["measurement_warmup_steps"]),
        samples_per_second=case.microbatch_size * 1_000_000.0 / max(step_us, 1.0),
        feasibility_status=summary["feasibility"]["status"],
        baseline_peak_phase=None if baseline_peak is None else baseline_peak["phase"],
        selected_peak_phase=None if selected_peak is None else selected_peak["phase"],
        measured_peak_phase=measured.get("peak_phase"),
        selected_peak_phase_match=None if selected_correction is None else selected_correction.get("phase_match"),
        measured_fw_us=None if "fw_us" not in phase_metrics else float(phase_metrics["fw_us"]),
        measured_bw_us=None if "bw_us" not in phase_metrics else float(phase_metrics["bw_us"]),
        measured_optimizer_us=None
        if "optimizer_us" not in phase_metrics
        else float(phase_metrics["optimizer_us"]),
        measured_fw_peak_bytes=None
        if "fw_peak_bytes" not in phase_metrics
        else int(phase_metrics["fw_peak_bytes"]),
        measured_bw_peak_bytes=None
        if "bw_peak_bytes" not in phase_metrics
        else int(phase_metrics["bw_peak_bytes"]),
        measured_optimizer_peak_bytes=None
        if "optimizer_peak_bytes" not in phase_metrics
        else int(phase_metrics["optimizer_peak_bytes"]),
        diagnostic_primary_cause=None if diagnostic is None else diagnostic["primary_cause"],
        diagnostic_normalized_saved_reduction_bytes=expectation.get("normalized_saved_reduction"),
        diagnostic_realization_gap_bytes=expectation.get("realization_gap"),
        diagnostic_total_expectation_gap_bytes=expectation.get("total_expectation_gap"),
        diagnostic_counterfactuals=_diagnostic_counterfactuals(summary),
        measured_candidate_count=len(summary["measured_candidates"]),
        measured_plan_results=measured_plan_results,
        selected_prediction_error_bytes=None if selected_correction is None else selected_correction["error_bytes"],
        selected_prediction_relative_error=None if selected_correction is None else selected_correction["relative_error"],
        selected_calibrated_prediction_error_bytes=None
        if selected_measured_plan is None
        else selected_measured_plan.get("calibrated_prediction_error_bytes"),
        selected_calibrated_prediction_relative_error=None
        if selected_measured_plan is None
        else selected_measured_plan.get("calibrated_prediction_relative_error"),
        selected_feasibility_prediction_match=None
        if selected_correction is None
        else selected_correction["feasibility_match"],
        simulation_accuracy_candidate_count=int(simulation_accuracy.get("candidate_count", 0)),
        simulation_accuracy_mean_absolute_error_bytes=simulation_accuracy.get("mean_absolute_error_bytes"),
        simulation_accuracy_max_absolute_error_bytes=simulation_accuracy.get("max_absolute_error_bytes"),
        simulation_accuracy_mean_absolute_relative_error=simulation_accuracy.get("mean_absolute_relative_error"),
        simulation_accuracy_within_10_percent_rate=simulation_accuracy.get("within_10_percent_rate"),
        cache_total_hits=int(cache.get("total_hits", 0)),
        cache_total_misses=int(cache.get("total_misses", 0)),
        cache_hit_rate=cache.get("hit_rate"),
        cache_layer_hits={str(layer): int(count) for layer, count in cache.get("layer_hits", {}).items()},
        cache_layer_misses={str(layer): int(count) for layer, count in cache.get("layer_misses", {}).items()},
        optimization_total_us=optimization_cost.get("total_optimization_us"),
        optimization_capture_us=optimization_cost.get("capture_us"),
        optimization_ir_build_us=optimization_cost.get("ir_build_us"),
        optimization_analysis_us=optimization_cost.get("analysis_us"),
        optimization_executor_build_us=optimization_cost.get("executor_build_us"),
        optimization_candidate_validation_measurement_us=optimization_cost.get(
            "candidate_validation_measurement_us"
        ),
        optimization_amortization_steps=optimization_cost.get("amortization_steps"),
        actual_joint_capture_count=int(optimization_cost.get("actual_joint_capture_count", 0)),
        candidate_count=len(summary["plans"]),
        fallback_plan_ids=tuple(summary["fallback_plan_ids"]),
        diagnostic_hints_enabled=search_diagnostics.get("diagnostic_hints_enabled"),
        diagnostic_hint_count=int(search_diagnostics.get("diagnostic_hint_count", 0)),
        diagnostic_hint_kinds=tuple(search_diagnostics.get("diagnostic_hint_kinds", ())),
        diagnostic_hint_candidate_match_count=int(
            search_diagnostics.get("diagnostic_hint_candidate_match_count", 0)
        ),
        diagnostic_hint_order_changed=bool(search_diagnostics.get("diagnostic_hint_order_changed", False)),
        diagnostic_hint_order_delta_count=int(search_diagnostics.get("diagnostic_hint_order_delta_count", 0)),
        repaired_candidate_count=int(search_diagnostics.get("repaired_candidate_count", 0)),
        repair_success_count=int(search_diagnostics.get("repair_success_count", 0)),
        feasible_before_repair_count=int(search_diagnostics.get("feasible_before_repair_count", 0)),
        feasible_after_repair_count=int(search_diagnostics.get("feasible_after_repair_count", 0)),
        exact_plan_id=exact.get("plan_id"),
        exact_plan_key=exact.get("plan_key"),
        exact_estimated_peak_bytes=exact.get("estimated_peak_bytes"),
        exact_estimated_step_us=exact.get("estimated_step_us"),
        selected_exact_peak_gap_bytes=exact.get("selected_peak_gap_bytes"),
        exact_error_type=exact.get("error_type"),
        exact_error_message=exact.get("error_message"),
        matrix_pass_index=case.matrix_pass_index,
        matrix_pass_count=case.matrix_pass_count,
    )


def _record_failure(case: ExperimentCase, exc: Exception) -> ExperimentRecord:
    return ExperimentRecord(
        task_name=case.task_name,
        variant_name=case.variant_name,
        config_fingerprint=case.config_fingerprint,
        microbatch_size=case.microbatch_size,
        budget_bytes=case.budget_bytes,
        status="failed",
        selected_plan_id=None,
        selected_plan_key=None,
        graph_key=None,
        selected_saved_value_ids=(),
        selected_effective_saved_value_ids=(),
        selected_estimated_peak_bytes=None,
        baseline_plan_id=None,
        baseline_estimated_peak_bytes=None,
        selected_estimated_peak_reduction_bytes=None,
        measured_peak_bytes=None,
        measured_peak_reserved_bytes=None,
        measured_budget_headroom_bytes=None,
        all_save_measured_peak_bytes=None,
        all_save_measured_step_us=None,
        selected_measured_peak_reduction_vs_all_save_bytes=None,
        selected_step_time_delta_vs_all_save_us=None,
        selected_samples_per_second_speedup_vs_all_save=None,
        measured_step_us=None,
        measurement_repeats=None,
        measurement_warmup_steps=None,
        samples_per_second=None,
        feasibility_status=None,
        baseline_peak_phase=None,
        selected_peak_phase=None,
        measured_peak_phase=None,
        selected_peak_phase_match=None,
        measured_fw_us=None,
        measured_bw_us=None,
        measured_optimizer_us=None,
        measured_fw_peak_bytes=None,
        measured_bw_peak_bytes=None,
        measured_optimizer_peak_bytes=None,
        diagnostic_primary_cause=None,
        diagnostic_normalized_saved_reduction_bytes=None,
        diagnostic_realization_gap_bytes=None,
        diagnostic_total_expectation_gap_bytes=None,
        diagnostic_counterfactuals=(),
        measured_candidate_count=0,
        measured_plan_results=(),
        selected_prediction_error_bytes=None,
        selected_prediction_relative_error=None,
        selected_calibrated_prediction_error_bytes=None,
        selected_calibrated_prediction_relative_error=None,
        selected_feasibility_prediction_match=None,
        simulation_accuracy_candidate_count=0,
        simulation_accuracy_mean_absolute_error_bytes=None,
        simulation_accuracy_max_absolute_error_bytes=None,
        simulation_accuracy_mean_absolute_relative_error=None,
        simulation_accuracy_within_10_percent_rate=None,
        cache_total_hits=0,
        cache_total_misses=0,
        cache_hit_rate=None,
        cache_layer_hits={},
        cache_layer_misses={},
        optimization_total_us=None,
        optimization_capture_us=None,
        optimization_ir_build_us=None,
        optimization_analysis_us=None,
        optimization_executor_build_us=None,
        optimization_candidate_validation_measurement_us=None,
        optimization_amortization_steps=None,
        actual_joint_capture_count=0,
        candidate_count=0,
        fallback_plan_ids=(),
        diagnostic_hints_enabled=None,
        diagnostic_hint_count=0,
        diagnostic_hint_kinds=(),
        repaired_candidate_count=0,
        repair_success_count=0,
        feasible_before_repair_count=0,
        feasible_after_repair_count=0,
        error_type=type(exc).__name__,
        error_message=str(exc),
        matrix_pass_index=case.matrix_pass_index,
        matrix_pass_count=case.matrix_pass_count,
    )


def run_experiment_matrix(
    *,
    task_names: tuple[str, ...],
    microbatch_sizes: tuple[int, ...],
    budget_bytes: tuple[int, ...],
    config: PeakAwareConfig | None = None,
    registry: TrainingTaskRegistry | None = None,
    include_exact_baseline: bool = False,
    exact_max_candidate_count: int = 12,
    variant_name: str = "default",
    device: str = "cpu",
    plan_artifact_dir: str | Path | None = None,
    matrix_pass_index: int = 0,
    matrix_pass_count: int = 1,
) -> tuple[ExperimentRecord, ...]:
    if not task_names:
        raise ValueError("task_names must not be empty")
    if not microbatch_sizes:
        raise ValueError("microbatch_sizes must not be empty")
    if not budget_bytes:
        raise ValueError("budget_bytes must not be empty")
    if any(value <= 0 for value in microbatch_sizes):
        raise ValueError("microbatch_sizes must be positive")
    if any(value <= 0 for value in budget_bytes):
        raise ValueError("budget_bytes must be positive")
    if exact_max_candidate_count < 0:
        raise ValueError("exact_max_candidate_count must be non-negative")
    if matrix_pass_index < 0 or matrix_pass_count <= 0 or matrix_pass_index >= matrix_pass_count:
        raise ValueError("matrix pass index/count is invalid")
    registry = registry or TrainingTaskRegistry.with_defaults()
    config = config or PeakAwareConfig()
    resolved_device = _resolve_experiment_device(device)
    config_fingerprint = _config_fingerprint(config, str(resolved_device))
    if plan_artifact_dir is not None:
        Path(plan_artifact_dir).mkdir(parents=True, exist_ok=True)
    records: list[ExperimentRecord] = []
    for task_name in task_names:
        task = registry.get(task_name)
        for microbatch_size in microbatch_sizes:
            for budget in budget_bytes:
                case = ExperimentCase(
                    variant_name,
                    config_fingerprint,
                    task_name,
                    microbatch_size,
                    budget,
                    matrix_pass_index=matrix_pass_index,
                    matrix_pass_count=matrix_pass_count,
                )
                model = task.build_model().to(resolved_device)
                optimizer = task.build_optimizer(model)
                args, kwargs = task.build_batch(microbatch_size)
                args = _move_batch_to_device(args, resolved_device)
                kwargs = _move_batch_to_device(kwargs, resolved_device)
                try:
                    result = optimize_training(
                        model,
                        args,
                        example_kwargs=kwargs,
                        loss_fn=task.loss_fn,
                        optimizer=optimizer,
                        memory_budget_bytes=budget,
                        config=config,
                    )
                except Exception as exc:
                    records.append(_record_failure(case, exc))
                    continue
                summary = summarize_result(result)
                if plan_artifact_dir is not None:
                    stem = _safe_artifact_stem(
                        variant_name,
                        f"pass{matrix_pass_index}",
                        task_name,
                        f"mb{microbatch_size}",
                        f"budget{budget}",
                        summary["selected_plan_key"][:12],
                    )
                    export_plan_artifact_json(result, Path(plan_artifact_dir) / f"{stem}.json")
                exact = None
                if include_exact_baseline:
                    exact = _run_exact_baseline(case, result, exact_max_candidate_count)
                records.append(_record_success(case, summary, exact))
    return tuple(records)


def _run_exact_baseline(case: ExperimentCase, result: Any, max_candidate_count: int = 12) -> dict[str, Any]:
    if result.analysis is None:
        return {"error_type": "MissingAnalysis", "error_message": "result did not include analysis"}
    try:
        exact = solve_exact_small_graph(
            result.analysis.ir,
            result.analysis.fixed_timeline,
            budget_bytes=case.budget_bytes,
            safety_margin_bytes=result.selected_plan.safety_margin_bytes,
            max_candidate_count=max_candidate_count,
        )
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error_message": str(exc)}
    exact_key = plan_identity_key(
        exact.plan.graph_key,
        exact.plan.saved_value_ids | exact.plan.mandatory_value_ids,
        exact.plan.budget_bytes,
    )
    return {
        "plan_id": exact.plan.plan_id,
        "plan_key": exact_key,
        "estimated_peak_bytes": exact.simulation.estimated_peak_bytes,
        "estimated_step_us": exact.simulation.estimated_step_us,
        "selected_peak_gap_bytes": result.selected_plan.estimated_peak_bytes - exact.simulation.estimated_peak_bytes,
    }


def experiment_records_to_dicts(records: tuple[ExperimentRecord, ...]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def experiment_records_from_dicts(rows: list[dict[str, Any]]) -> tuple[ExperimentRecord, ...]:
    tuple_fields = {
        "selected_saved_value_ids",
        "selected_effective_saved_value_ids",
        "diagnostic_counterfactuals",
        "measured_plan_results",
        "fallback_plan_ids",
        "diagnostic_hint_kinds",
    }
    records = []
    for row in rows:
        normalized = dict(row)
        normalized.setdefault("selected_calibrated_prediction_error_bytes", None)
        normalized.setdefault("selected_calibrated_prediction_relative_error", None)
        normalized.setdefault("measured_fw_us", None)
        normalized.setdefault("measured_bw_us", None)
        normalized.setdefault("measured_optimizer_us", None)
        normalized.setdefault("measured_fw_peak_bytes", None)
        normalized.setdefault("measured_bw_peak_bytes", None)
        normalized.setdefault("measured_optimizer_peak_bytes", None)
        normalized.setdefault("diagnostic_hint_candidate_match_count", 0)
        normalized.setdefault("diagnostic_hint_order_changed", False)
        normalized.setdefault("diagnostic_hint_order_delta_count", 0)
        for field in tuple_fields:
            if field in normalized:
                normalized[field] = tuple(normalized[field])
        if "measured_plan_results" in normalized:
            normalized["measured_plan_results"] = _calibrate_measured_plan_rows(
                tuple(normalized["measured_plan_results"])
            )
        if normalized["selected_calibrated_prediction_error_bytes"] is None:
            selected_row = next(
                (
                    row
                    for row in normalized.get("measured_plan_results", ())
                    if row.get("plan_id") == normalized.get("selected_plan_id")
                ),
                None,
            )
            if selected_row is not None:
                normalized["selected_calibrated_prediction_error_bytes"] = selected_row.get(
                    "calibrated_prediction_error_bytes"
                )
                normalized["selected_calibrated_prediction_relative_error"] = selected_row.get(
                    "calibrated_prediction_relative_error"
                )
        records.append(ExperimentRecord(**normalized))
    return tuple(records)


def _safe_artifact_stem(*parts: object) -> str:
    text = "_".join(str(part) for part in parts)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _mean(values: list[float | int]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _counts(values: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _tuple_counts(values: list[tuple[str, ...]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for items in values:
        for item in items:
            counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def _sum_layer_counts(records: list[ExperimentRecord], field_name: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        counts = getattr(record, field_name)
        for layer, count in counts.items():
            totals[layer] = totals.get(layer, 0) + int(count)
    return dict(sorted(totals.items()))


def _layer_hit_rates(hits: dict[str, int], misses: dict[str, int]) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for layer in sorted(set(hits) | set(misses)):
        total = hits.get(layer, 0) + misses.get(layer, 0)
        rates[layer] = None if total == 0 else hits.get(layer, 0) / total
    return rates


def _environment_fingerprint() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": tuple(cuda_devices),
    }


def summarize_experiment_records(records: tuple[ExperimentRecord, ...]) -> ExperimentSummary:
    total = len(records)
    ok = [record for record in records if record.status == "ok"]
    failed_count = total - len(ok)
    violations = [
        record
        for record in ok
        if record.measured_peak_bytes is not None and record.measured_peak_bytes > record.budget_bytes
    ]
    feasible_microbatches = [
        record.microbatch_size
        for record in ok
        if record.measured_peak_bytes is not None and record.measured_peak_bytes <= record.budget_bytes
    ]
    samples = [record.samples_per_second for record in ok if record.samples_per_second is not None]
    steps = [record.measured_step_us for record in ok if record.measured_step_us is not None]
    peaks = [record.measured_peak_bytes for record in ok if record.measured_peak_bytes is not None]
    headrooms = [
        record.measured_budget_headroom_bytes
        for record in ok
        if record.measured_budget_headroom_bytes is not None
    ]
    reductions = [
        record.selected_estimated_peak_reduction_bytes
        for record in ok
        if record.selected_estimated_peak_reduction_bytes is not None
    ]
    measured_peak_reductions = [
        record.selected_measured_peak_reduction_vs_all_save_bytes
        for record in ok
        if record.selected_measured_peak_reduction_vs_all_save_bytes is not None
    ]
    selected_step_time_deltas = [
        record.selected_step_time_delta_vs_all_save_us
        for record in ok
        if record.selected_step_time_delta_vs_all_save_us is not None
    ]
    selected_speedups = [
        record.selected_samples_per_second_speedup_vs_all_save
        for record in ok
        if record.selected_samples_per_second_speedup_vs_all_save is not None
    ]
    selected_abs_errors = [
        abs(record.selected_prediction_error_bytes)
        for record in ok
        if record.selected_prediction_error_bytes is not None
    ]
    selected_abs_relative_errors = [
        abs(record.selected_prediction_relative_error)
        for record in ok
        if record.selected_prediction_relative_error is not None
    ]
    selected_calibrated_abs_errors = [
        abs(record.selected_calibrated_prediction_error_bytes)
        for record in ok
        if record.selected_calibrated_prediction_error_bytes is not None
    ]
    selected_calibrated_abs_relative_errors = [
        abs(record.selected_calibrated_prediction_relative_error)
        for record in ok
        if record.selected_calibrated_prediction_relative_error is not None
    ]
    simulation_accuracy_abs_errors = [
        abs(int(row["prediction_error_bytes"]))
        for record in ok
        for row in record.measured_plan_results
        if row.get("prediction_error_bytes") is not None
    ]
    simulation_accuracy_abs_relative_errors = [
        abs(float(row["prediction_error_bytes"]) / float(row["estimated_peak_bytes"]))
        for record in ok
        for row in record.measured_plan_results
        if row.get("prediction_error_bytes") is not None and row.get("estimated_peak_bytes")
    ]
    calibrated_simulation_accuracy_abs_errors = [
        abs(int(row["calibrated_prediction_error_bytes"]))
        for record in ok
        for row in record.measured_plan_results
        if row.get("calibrated_prediction_error_bytes") is not None
    ]
    calibrated_simulation_accuracy_abs_relative_errors = [
        abs(float(row["calibrated_prediction_relative_error"]))
        for record in ok
        for row in record.measured_plan_results
        if row.get("calibrated_prediction_relative_error") is not None
    ]
    simulation_accuracy_counts = [
        record.simulation_accuracy_candidate_count
        for record in ok
        if record.simulation_accuracy_candidate_count > 0
    ]
    simulation_accuracy_mean_abs_errors = [
        record.simulation_accuracy_mean_absolute_error_bytes
        for record in ok
        if record.simulation_accuracy_mean_absolute_error_bytes is not None
    ]
    simulation_accuracy_max_abs_errors = [
        record.simulation_accuracy_max_absolute_error_bytes
        for record in ok
        if record.simulation_accuracy_max_absolute_error_bytes is not None
    ]
    simulation_accuracy_mean_relative_errors = [
        record.simulation_accuracy_mean_absolute_relative_error
        for record in ok
        if record.simulation_accuracy_mean_absolute_relative_error is not None
    ]
    simulation_accuracy_within_10 = [
        record.simulation_accuracy_within_10_percent_rate
        for record in ok
        if record.simulation_accuracy_within_10_percent_rate is not None
    ]
    phase_matches = [
        record.selected_peak_phase_match
        for record in ok
        if record.selected_peak_phase_match is not None
    ]
    feasible_matches = [
        record.selected_feasibility_prediction_match
        for record in ok
        if record.selected_feasibility_prediction_match is not None
    ]
    normalized_saved = [
        record.diagnostic_normalized_saved_reduction_bytes
        for record in ok
        if record.diagnostic_normalized_saved_reduction_bytes is not None
    ]
    realization_gaps = [
        record.diagnostic_realization_gap_bytes
        for record in ok
        if record.diagnostic_realization_gap_bytes is not None
    ]
    total_expectation_gaps = [
        record.diagnostic_total_expectation_gap_bytes
        for record in ok
        if record.diagnostic_total_expectation_gap_bytes is not None
    ]
    total_hits = sum(record.cache_total_hits for record in records)
    total_misses = sum(record.cache_total_misses for record in records)
    layer_hits = _sum_layer_counts(records, "cache_layer_hits")
    layer_misses = _sum_layer_counts(records, "cache_layer_misses")
    optimization_totals = [record.optimization_total_us for record in ok if record.optimization_total_us is not None]
    optimization_captures = [
        record.optimization_capture_us
        for record in ok
        if record.optimization_capture_us is not None
    ]
    optimization_ir_builds = [
        record.optimization_ir_build_us
        for record in ok
        if record.optimization_ir_build_us is not None
    ]
    optimization_analyses = [
        record.optimization_analysis_us
        for record in ok
        if record.optimization_analysis_us is not None
    ]
    optimization_executor_builds = [
        record.optimization_executor_build_us
        for record in ok
        if record.optimization_executor_build_us is not None
    ]
    optimization_candidate_validations = [
        record.optimization_candidate_validation_measurement_us
        for record in ok
        if record.optimization_candidate_validation_measurement_us is not None
    ]
    optimization_amortization_steps = [
        record.optimization_amortization_steps
        for record in ok
        if record.optimization_amortization_steps is not None
    ]
    repaired_candidate_count = sum(record.repaired_candidate_count for record in ok)
    repair_success_count = sum(record.repair_success_count for record in ok)
    exact_success = [record for record in ok if record.exact_plan_key is not None]
    exact_failures = [record for record in ok if record.exact_error_type is not None]
    exact_gaps = [
        record.selected_exact_peak_gap_bytes
        for record in exact_success
        if record.selected_exact_peak_gap_bytes is not None
    ]
    return ExperimentSummary(
        total_records=total,
        ok_records=len(ok),
        failed_records=failed_count,
        success_rate=None if total == 0 else len(ok) / total,
        environment_fingerprint=_environment_fingerprint(),
        variant_counts=_counts([record.variant_name for record in records]),
        budget_violation_count=len(violations),
        budget_violation_rate=None if not ok else len(violations) / len(ok),
        max_feasible_microbatch=max(feasible_microbatches) if feasible_microbatches else None,
        mean_samples_per_second=_mean(samples),
        mean_measured_step_us=_mean(steps),
        mean_measured_peak_bytes=_mean(peaks),
        mean_measured_budget_headroom_bytes=_mean(headrooms),
        mean_measured_peak_reduction_vs_all_save_bytes=_mean(measured_peak_reductions),
        mean_selected_step_time_delta_vs_all_save_us=_mean(selected_step_time_deltas),
        mean_selected_samples_per_second_speedup_vs_all_save=_mean(selected_speedups),
        mean_estimated_peak_reduction_bytes=_mean(reductions),
        aggregate_cache_hit_rate=None if total_hits + total_misses == 0 else total_hits / (total_hits + total_misses),
        total_cache_hits=total_hits,
        total_cache_misses=total_misses,
        cache_layer_hits=layer_hits,
        cache_layer_misses=layer_misses,
        cache_layer_hit_rates=_layer_hit_rates(layer_hits, layer_misses),
        mean_optimization_total_us=_mean(optimization_totals),
        mean_optimization_capture_us=_mean(optimization_captures),
        mean_optimization_ir_build_us=_mean(optimization_ir_builds),
        mean_optimization_analysis_us=_mean(optimization_analyses),
        mean_optimization_executor_build_us=_mean(optimization_executor_builds),
        mean_optimization_candidate_validation_measurement_us=_mean(optimization_candidate_validations),
        mean_optimization_amortization_steps=_mean(optimization_amortization_steps),
        total_actual_joint_capture_count=sum(record.actual_joint_capture_count for record in records),
        selected_prediction_count=len(selected_abs_errors),
        mean_selected_prediction_absolute_error_bytes=_mean(selected_abs_errors),
        p50_selected_prediction_absolute_error_bytes=_percentile(selected_abs_errors, 0.50),
        p90_selected_prediction_absolute_error_bytes=_percentile(selected_abs_errors, 0.90),
        max_selected_prediction_absolute_error_bytes=None if not selected_abs_errors else max(selected_abs_errors),
        mean_selected_prediction_absolute_relative_error=_mean(selected_abs_relative_errors),
        p50_selected_prediction_absolute_relative_error=_percentile(selected_abs_relative_errors, 0.50),
        p90_selected_prediction_absolute_relative_error=_percentile(selected_abs_relative_errors, 0.90),
        mean_selected_calibrated_prediction_absolute_error_bytes=_mean(selected_calibrated_abs_errors),
        p50_selected_calibrated_prediction_absolute_error_bytes=_percentile(selected_calibrated_abs_errors, 0.50),
        p90_selected_calibrated_prediction_absolute_error_bytes=_percentile(selected_calibrated_abs_errors, 0.90),
        mean_selected_calibrated_prediction_absolute_relative_error=_mean(
            selected_calibrated_abs_relative_errors
        ),
        p50_selected_calibrated_prediction_absolute_relative_error=_percentile(
            selected_calibrated_abs_relative_errors,
            0.50,
        ),
        p90_selected_calibrated_prediction_absolute_relative_error=_percentile(
            selected_calibrated_abs_relative_errors,
            0.90,
        ),
        simulation_accuracy_candidate_count=sum(simulation_accuracy_counts),
        mean_simulation_accuracy_absolute_error_bytes=_mean(simulation_accuracy_abs_errors)
        if simulation_accuracy_abs_errors
        else _mean(simulation_accuracy_mean_abs_errors),
        p50_simulation_accuracy_absolute_error_bytes=_percentile(simulation_accuracy_abs_errors, 0.50),
        p90_simulation_accuracy_absolute_error_bytes=_percentile(simulation_accuracy_abs_errors, 0.90),
        max_simulation_accuracy_absolute_error_bytes=None
        if not simulation_accuracy_abs_errors and not simulation_accuracy_max_abs_errors
        else max(simulation_accuracy_abs_errors or simulation_accuracy_max_abs_errors),
        mean_simulation_accuracy_absolute_relative_error=_mean(simulation_accuracy_abs_relative_errors)
        if simulation_accuracy_abs_relative_errors
        else _mean(simulation_accuracy_mean_relative_errors),
        p50_simulation_accuracy_absolute_relative_error=_percentile(simulation_accuracy_abs_relative_errors, 0.50),
        p90_simulation_accuracy_absolute_relative_error=_percentile(simulation_accuracy_abs_relative_errors, 0.90),
        mean_simulation_accuracy_within_10_percent_rate=_mean(simulation_accuracy_within_10),
        mean_calibrated_simulation_accuracy_absolute_error_bytes=_mean(
            calibrated_simulation_accuracy_abs_errors
        ),
        p50_calibrated_simulation_accuracy_absolute_error_bytes=_percentile(
            calibrated_simulation_accuracy_abs_errors,
            0.50,
        ),
        p90_calibrated_simulation_accuracy_absolute_error_bytes=_percentile(
            calibrated_simulation_accuracy_abs_errors,
            0.90,
        ),
        max_calibrated_simulation_accuracy_absolute_error_bytes=None
        if not calibrated_simulation_accuracy_abs_errors
        else max(calibrated_simulation_accuracy_abs_errors),
        mean_calibrated_simulation_accuracy_absolute_relative_error=_mean(
            calibrated_simulation_accuracy_abs_relative_errors
        ),
        p50_calibrated_simulation_accuracy_absolute_relative_error=_percentile(
            calibrated_simulation_accuracy_abs_relative_errors,
            0.50,
        ),
        p90_calibrated_simulation_accuracy_absolute_relative_error=_percentile(
            calibrated_simulation_accuracy_abs_relative_errors,
            0.90,
        ),
        calibrated_simulation_accuracy_within_10_percent_rate=None
        if not calibrated_simulation_accuracy_abs_relative_errors
        else sum(1 for error in calibrated_simulation_accuracy_abs_relative_errors if error <= 0.10)
        / len(calibrated_simulation_accuracy_abs_relative_errors),
        phase_classification_count=len(phase_matches),
        phase_classification_accuracy=None
        if not phase_matches
        else sum(1 for matched in phase_matches if matched) / len(phase_matches),
        feasible_classification_count=len(feasible_matches),
        feasible_classification_accuracy=None
        if not feasible_matches
        else sum(1 for matched in feasible_matches if matched) / len(feasible_matches),
        root_cause_counts=_counts([record.diagnostic_primary_cause for record in ok]),
        selected_peak_phase_counts=_counts([record.selected_peak_phase for record in ok]),
        measured_peak_phase_counts=_counts([record.measured_peak_phase for record in ok]),
        diagnostic_hints_enabled_count=sum(1 for record in ok if record.diagnostic_hints_enabled),
        diagnostic_hint_count=sum(record.diagnostic_hint_count for record in ok),
        diagnostic_hint_kind_counts=_tuple_counts([record.diagnostic_hint_kinds for record in ok]),
        diagnostic_hint_candidate_match_count=sum(record.diagnostic_hint_candidate_match_count for record in ok),
        diagnostic_hint_order_changed_count=sum(1 for record in ok if record.diagnostic_hint_order_changed),
        diagnostic_hint_order_delta_count=sum(record.diagnostic_hint_order_delta_count for record in ok),
        repaired_candidate_count=repaired_candidate_count,
        repair_success_count=repair_success_count,
        repair_success_rate=None
        if repaired_candidate_count == 0
        else repair_success_count / repaired_candidate_count,
        feasible_before_repair_count=sum(record.feasible_before_repair_count for record in ok),
        feasible_after_repair_count=sum(record.feasible_after_repair_count for record in ok),
        mean_diagnostic_normalized_saved_reduction_bytes=_mean(normalized_saved),
        mean_diagnostic_realization_gap_bytes=_mean(realization_gaps),
        mean_diagnostic_total_expectation_gap_bytes=_mean(total_expectation_gaps),
        exact_success_count=len(exact_success),
        exact_failure_count=len(exact_failures),
        mean_selected_exact_peak_gap_bytes=_mean(exact_gaps),
    )


def experiment_summary_to_dict(summary: ExperimentSummary) -> dict[str, Any]:
    return asdict(summary)


def summarize_experiment_records_by_variant(records: tuple[ExperimentRecord, ...]) -> dict[str, ExperimentSummary]:
    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.variant_name, []).append(record)
    return {
        variant_name: summarize_experiment_records(tuple(variant_records))
        for variant_name, variant_records in sorted(grouped.items())
    }


def experiment_variant_summaries_to_dict(
    summaries: dict[str, ExperimentSummary],
) -> dict[str, dict[str, Any]]:
    return {variant_name: experiment_summary_to_dict(summary) for variant_name, summary in summaries.items()}


def _record_pair_key(record: ExperimentRecord) -> tuple[str, int, int, int]:
    return (record.task_name, record.microbatch_size, record.budget_bytes, record.matrix_pass_index)


def _metric_delta(on_value: float | int | None, off_value: float | int | None) -> float | None:
    if on_value is None or off_value is None:
        return None
    return float(on_value) - float(off_value)


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return _mean(filtered)


def _hint_pair_conclusion(row: dict[str, Any]) -> str:
    if row["on_status"] == "ok" and row["off_status"] != "ok":
        return "improved_success"
    if row["on_status"] != "ok" and row["off_status"] == "ok":
        return "regressed_success"
    if row["on_status"] != "ok" or row["off_status"] != "ok":
        return "inconclusive"
    violation_delta = row["budget_violation_delta"]
    throughput_delta = row["samples_per_second_delta"]
    candidate_delta = row["candidate_count_delta"]
    measured_candidate_delta = row["measured_candidate_count_delta"]
    order_delta = row["diagnostic_hint_order_delta_count_delta"]
    if violation_delta is not None and violation_delta < 0:
        return "improved_budget"
    if throughput_delta is not None and throughput_delta > 0:
        return "improved_throughput"
    if measured_candidate_delta is not None and measured_candidate_delta < 0:
        return "improved_search"
    if candidate_delta is not None and candidate_delta < 0:
        return "improved_search"
    if violation_delta is not None and violation_delta > 0:
        return "regressed_budget"
    if throughput_delta is not None and throughput_delta < 0:
        return "regressed_throughput"
    if measured_candidate_delta is not None and measured_candidate_delta > 0:
        return "regressed_search"
    if candidate_delta is not None and candidate_delta > 0:
        return "regressed_search"
    if order_delta is not None and order_delta > 0:
        return "changed_search_order"
    return "neutral"


def _hint_ablation_verdict(conclusion_counts: dict[str, int]) -> str:
    if not conclusion_counts:
        return "no_pairs"
    improved = sum(count for kind, count in conclusion_counts.items() if kind.startswith("improved_"))
    regressed = sum(count for kind, count in conclusion_counts.items() if kind.startswith("regressed_"))
    changed_order = conclusion_counts.get("changed_search_order", 0)
    inconclusive = conclusion_counts.get("inconclusive", 0)
    neutral = conclusion_counts.get("neutral", 0)
    if improved and not regressed:
        return "improved" if improved > neutral + inconclusive + changed_order else "mixed"
    if regressed and not improved:
        return "regressed" if regressed > neutral + inconclusive + changed_order else "mixed"
    if improved and regressed:
        return "mixed"
    if changed_order:
        return "changed_search_order" if changed_order > neutral + inconclusive else "mixed"
    if neutral:
        return "neutral"
    return "inconclusive"


def summarize_hint_ablation(records: tuple[ExperimentRecord, ...]) -> dict[str, Any]:
    by_variant: dict[str, dict[tuple[str, int, int, int], ExperimentRecord]] = {
        "diagnostic_hints_on": {},
        "diagnostic_hints_off": {},
    }
    for record in records:
        if record.variant_name in by_variant:
            by_variant[record.variant_name][_record_pair_key(record)] = record
    paired_keys = sorted(set(by_variant["diagnostic_hints_on"]) & set(by_variant["diagnostic_hints_off"]))
    rows: list[dict[str, Any]] = []
    for task_name, microbatch_size, budget_bytes, matrix_pass_index in paired_keys:
        on_record = by_variant["diagnostic_hints_on"][(task_name, microbatch_size, budget_bytes, matrix_pass_index)]
        off_record = by_variant["diagnostic_hints_off"][(task_name, microbatch_size, budget_bytes, matrix_pass_index)]
        row = {
            "task_name": task_name,
            "microbatch_size": microbatch_size,
            "budget_bytes": budget_bytes,
            "matrix_pass_index": matrix_pass_index,
            "on_status": on_record.status,
            "off_status": off_record.status,
            "success_delta": int(on_record.status == "ok") - int(off_record.status == "ok"),
            "budget_violation_delta": int(
                on_record.status == "ok"
                and on_record.measured_peak_bytes is not None
                and on_record.measured_peak_bytes > on_record.budget_bytes
            )
            - int(
                off_record.status == "ok"
                and off_record.measured_peak_bytes is not None
                and off_record.measured_peak_bytes > off_record.budget_bytes
            ),
            "samples_per_second_delta": _metric_delta(on_record.samples_per_second, off_record.samples_per_second),
            "measured_step_us_delta": _metric_delta(on_record.measured_step_us, off_record.measured_step_us),
            "measured_peak_bytes_delta": _metric_delta(on_record.measured_peak_bytes, off_record.measured_peak_bytes),
            "candidate_count_delta": _metric_delta(on_record.candidate_count, off_record.candidate_count),
            "measured_candidate_count_delta": _metric_delta(
                on_record.measured_candidate_count,
                off_record.measured_candidate_count,
            ),
            "repair_success_count_delta": _metric_delta(
                on_record.repair_success_count,
                off_record.repair_success_count,
            ),
            "diagnostic_hint_count_delta": _metric_delta(
                on_record.diagnostic_hint_count,
                off_record.diagnostic_hint_count,
            ),
            "diagnostic_hint_candidate_match_count_delta": _metric_delta(
                on_record.diagnostic_hint_candidate_match_count,
                off_record.diagnostic_hint_candidate_match_count,
            ),
            "diagnostic_hint_order_changed_delta": _metric_delta(
                int(on_record.diagnostic_hint_order_changed),
                int(off_record.diagnostic_hint_order_changed),
            ),
            "diagnostic_hint_order_delta_count_delta": _metric_delta(
                on_record.diagnostic_hint_order_delta_count,
                off_record.diagnostic_hint_order_delta_count,
            ),
        }
        row["conclusion"] = _hint_pair_conclusion(row)
        rows.append(row)
    conclusion_counts = _counts([row["conclusion"] for row in rows])
    improved_count = sum(count for kind, count in conclusion_counts.items() if kind.startswith("improved_"))
    regressed_count = sum(count for kind, count in conclusion_counts.items() if kind.startswith("regressed_"))
    changed_order_count = conclusion_counts.get("changed_search_order", 0)
    return {
        "pair_count": len(rows),
        "both_ok_count": sum(1 for row in rows if row["on_status"] == "ok" and row["off_status"] == "ok"),
        "on_success_count": sum(1 for row in rows if row["on_status"] == "ok"),
        "off_success_count": sum(1 for row in rows if row["off_status"] == "ok"),
        "success_rate_delta": None
        if not rows
        else (
            sum(1 for row in rows if row["on_status"] == "ok")
            - sum(1 for row in rows if row["off_status"] == "ok")
        )
        / len(rows),
        "mean_samples_per_second_delta": _mean_optional([row["samples_per_second_delta"] for row in rows]),
        "mean_measured_step_us_delta": _mean_optional([row["measured_step_us_delta"] for row in rows]),
        "mean_measured_peak_bytes_delta": _mean_optional([row["measured_peak_bytes_delta"] for row in rows]),
        "mean_candidate_count_delta": _mean_optional([row["candidate_count_delta"] for row in rows]),
        "mean_measured_candidate_count_delta": _mean_optional(
            [row["measured_candidate_count_delta"] for row in rows]
        ),
        "mean_repair_success_count_delta": _mean_optional([row["repair_success_count_delta"] for row in rows]),
        "mean_diagnostic_hint_count_delta": _mean_optional([row["diagnostic_hint_count_delta"] for row in rows]),
        "mean_diagnostic_hint_candidate_match_count_delta": _mean_optional(
            [row["diagnostic_hint_candidate_match_count_delta"] for row in rows]
        ),
        "mean_diagnostic_hint_order_delta_count_delta": _mean_optional(
            [row["diagnostic_hint_order_delta_count_delta"] for row in rows]
        ),
        "conclusion_counts": conclusion_counts,
        "improved_pair_count": improved_count,
        "regressed_pair_count": regressed_count,
        "changed_search_order_pair_count": changed_order_count,
        "neutral_pair_count": conclusion_counts.get("neutral", 0),
        "inconclusive_pair_count": conclusion_counts.get("inconclusive", 0),
        "verdict": _hint_ablation_verdict(conclusion_counts),
        "rows": rows,
    }


def summarize_cache_reuse(records: tuple[ExperimentRecord, ...]) -> dict[str, Any]:
    pass_rows = []
    for pass_index in sorted({record.matrix_pass_index for record in records}):
        subset = [record for record in records if record.matrix_pass_index == pass_index]
        hits = sum(record.cache_total_hits for record in subset)
        misses = sum(record.cache_total_misses for record in subset)
        total = hits + misses
        layer_hits = _sum_layer_counts(subset, "cache_layer_hits")
        layer_misses = _sum_layer_counts(subset, "cache_layer_misses")
        capture_attempts = layer_hits.get("capture", 0) + layer_misses.get("capture", 0)
        pass_rows.append(
            {
                "matrix_pass_index": pass_index,
                "record_count": len(subset),
                "ok_count": sum(1 for record in subset if record.status == "ok"),
                "failed_count": sum(1 for record in subset if record.status != "ok"),
                "total_cache_hits": hits,
                "total_cache_misses": misses,
                "cache_hit_rate": None if total == 0 else hits / total,
                "cache_layer_hits": layer_hits,
                "cache_layer_misses": layer_misses,
                "cache_layer_hit_rates": _layer_hit_rates(layer_hits, layer_misses),
                "capture_cache_attempt_count": capture_attempts,
                "capture_cache_hit_rate": None
                if capture_attempts == 0
                else layer_hits.get("capture", 0) / capture_attempts,
                "actual_joint_capture_count": sum(record.actual_joint_capture_count for record in subset),
            }
        )
    warm_rows = [row for row in pass_rows if int(row["matrix_pass_index"]) > 0]
    return {
        "matrix_pass_count": len(pass_rows),
        "record_count": len(records),
        "warm_pass_count": len(warm_rows),
        "cold_cache_hit_rate": None if not pass_rows else pass_rows[0]["cache_hit_rate"],
        "mean_warm_cache_hit_rate": _mean_optional([row["cache_hit_rate"] for row in warm_rows]),
        "cold_capture_cache_hit_rate": None if not pass_rows else pass_rows[0]["capture_cache_hit_rate"],
        "mean_warm_capture_cache_hit_rate": _mean_optional([row["capture_cache_hit_rate"] for row in warm_rows]),
        "cold_actual_joint_capture_count": None if not pass_rows else pass_rows[0]["actual_joint_capture_count"],
        "warm_actual_joint_capture_count": sum(int(row["actual_joint_capture_count"]) for row in warm_rows),
        "pass_rows": pass_rows,
    }


def _baseline_group(plan_id: str) -> str:
    if plan_id.startswith("greedy_drop_"):
        return "greedy"
    return plan_id


def _sac_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if "rows" in payload and isinstance(payload["rows"], list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    return [payload]


def _sac_key(row: dict[str, Any]) -> tuple[str, int, str] | None:
    try:
        return str(row["task_name"]), int(row["microbatch_size"]), str(row.get("device", "cpu"))
    except (KeyError, TypeError, ValueError):
        return None


def summarize_baseline_comparisons(
    records: tuple[ExperimentRecord, ...],
    sac_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sac_rows = _sac_rows(sac_baseline)
    usable_sac_by_key = {}
    usable_sac_row_count = 0
    for row in sac_rows:
        key = _sac_key(row)
        if key is not None and row.get("status") == "ok" and row.get("performance_result_usable"):
            usable_sac_row_count += 1
            usable_sac_by_key[key] = row
    for record in records:
        if record.status != "ok" or record.measured_peak_bytes is None or record.measured_step_us is None:
            continue
        selected_peak = float(record.measured_peak_bytes)
        selected_step = float(record.measured_step_us)
        for measured in record.measured_plan_results:
            plan_id = str(measured["plan_id"])
            peak = measured.get("measured_peak_bytes")
            step = measured.get("measured_step_us")
            if peak is None or step is None:
                continue
            peak = float(peak)
            step = float(step)
            row = {
                "variant_name": record.variant_name,
                "task_name": record.task_name,
                "microbatch_size": record.microbatch_size,
                "budget_bytes": record.budget_bytes,
                "plan_id": plan_id,
                "baseline_group": _baseline_group(plan_id),
                "selected_plan_id": record.selected_plan_id,
                "measured_peak_bytes": int(peak),
                "measured_step_us": step,
                "measured_feasible": bool(peak <= record.budget_bytes),
                "selected_measured_peak_bytes": int(selected_peak),
                "selected_measured_step_us": selected_step,
                "peak_reduction_vs_plan_bytes": int(peak - selected_peak),
                "step_time_delta_vs_plan_us": step - selected_step,
                "samples_per_second_speedup_vs_plan": step / max(selected_step, 1.0),
            }
            rows.append(row)
        sac_row = usable_sac_by_key.get(
            (
                record.task_name,
                record.microbatch_size,
                str(record.config_fingerprint.get("device", "cpu")),
            )
        )
        if sac_row is not None:
            peak = float(sac_row["sac_overall_peak_bytes"])
            step = float(sac_row["sac_step_us"])
            rows.append(
                {
                    "variant_name": record.variant_name,
                    "task_name": record.task_name,
                    "microbatch_size": record.microbatch_size,
                    "budget_bytes": record.budget_bytes,
                    "plan_id": str(sac_row.get("baseline_id", "pytorch_sac_prefer_recompute")),
                    "baseline_group": "pytorch_sac",
                    "selected_plan_id": record.selected_plan_id,
                    "measured_peak_bytes": int(peak),
                    "measured_step_us": step,
                    "measured_feasible": bool(peak <= record.budget_bytes),
                    "selected_measured_peak_bytes": int(selected_peak),
                    "selected_measured_step_us": selected_step,
                    "peak_reduction_vs_plan_bytes": int(peak - selected_peak),
                    "step_time_delta_vs_plan_us": step - selected_step,
                    "samples_per_second_speedup_vs_plan": step / max(selected_step, 1.0),
                    "external_baseline": True,
                    "correctness_passed": bool(sac_row.get("correctness_passed")),
                }
            )
    summaries: dict[str, dict[str, Any]] = {}
    for group in sorted({row["baseline_group"] for row in rows}):
        group_rows = [row for row in rows if row["baseline_group"] == group]
        peak_reductions = [row["peak_reduction_vs_plan_bytes"] for row in group_rows]
        step_deltas = [row["step_time_delta_vs_plan_us"] for row in group_rows]
        speedups = [row["samples_per_second_speedup_vs_plan"] for row in group_rows]
        summaries[group] = {
            "measured_count": len(group_rows),
            "budget_violation_count": sum(1 for row in group_rows if not row["measured_feasible"]),
            "budget_violation_rate": sum(1 for row in group_rows if not row["measured_feasible"]) / len(group_rows),
            "selected_peak_win_count": sum(1 for value in peak_reductions if value > 0),
            "selected_step_win_count": sum(1 for value in step_deltas if value > 0),
            "mean_peak_reduction_vs_plan_bytes": _mean(peak_reductions),
            "mean_step_time_delta_vs_plan_us": _mean(step_deltas),
            "mean_samples_per_second_speedup_vs_plan": _mean(speedups),
        }
    return {
        "baseline_groups": summaries,
        "row_count": len(rows),
        "external_sac_rows_seen": len(sac_rows),
        "external_sac_rows_usable": usable_sac_row_count,
        "external_sac_keys_matched": len(usable_sac_by_key),
        "rows": rows,
    }


def summarize_layered_simulation_accuracy(records: tuple[ExperimentRecord, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status != "ok" or record.measured_peak_bytes is None:
            continue
        measured_peak = int(record.measured_peak_bytes)
        measured_phase = record.measured_peak_phase
        for item in record.diagnostic_counterfactuals:
            peak_bytes = item.get("candidate_peak_bytes")
            if peak_bytes is None:
                rows.append(
                    {
                        "variant_name": record.variant_name,
                        "task_name": record.task_name,
                        "microbatch_size": record.microbatch_size,
                        "budget_bytes": record.budget_bytes,
                        "level": item.get("level"),
                        "status": item.get("status"),
                        "unavailable_reason": item.get("unavailable_reason"),
                        "candidate_peak_bytes": None,
                        "measured_peak_bytes": measured_peak,
                        "error_bytes": None,
                        "relative_error": None,
                        "candidate_peak_phase": item.get("candidate_peak_phase"),
                        "measured_peak_phase": measured_phase,
                        "phase_match": None,
                    }
                )
                continue
            error = measured_peak - int(peak_bytes)
            candidate_phase = item.get("candidate_peak_phase")
            rows.append(
                {
                    "variant_name": record.variant_name,
                    "task_name": record.task_name,
                    "microbatch_size": record.microbatch_size,
                    "budget_bytes": record.budget_bytes,
                    "level": item.get("level"),
                    "status": item.get("status"),
                    "unavailable_reason": item.get("unavailable_reason"),
                    "candidate_peak_bytes": int(peak_bytes),
                    "measured_peak_bytes": measured_peak,
                    "error_bytes": error,
                    "relative_error": None if int(peak_bytes) == 0 else error / int(peak_bytes),
                    "candidate_peak_phase": candidate_phase,
                    "measured_peak_phase": measured_phase,
                    "phase_match": None if measured_phase is None else candidate_phase == measured_phase,
                }
            )
    level_summaries: dict[str, dict[str, Any]] = {}
    for level in sorted({str(row["level"]) for row in rows if row.get("level") is not None}):
        level_rows = [row for row in rows if row.get("level") == level]
        available = [row for row in level_rows if row.get("error_bytes") is not None]
        abs_errors = [abs(int(row["error_bytes"])) for row in available]
        rel_errors = [abs(float(row["relative_error"])) for row in available if row.get("relative_error") is not None]
        phase_matches = [bool(row["phase_match"]) for row in available if row.get("phase_match") is not None]
        level_summaries[level] = {
            "row_count": len(level_rows),
            "available_count": len(available),
            "mean_absolute_error_bytes": _mean(abs_errors),
            "max_absolute_error_bytes": None if not abs_errors else max(abs_errors),
            "mean_absolute_relative_error": _mean(rel_errors),
            "within_10_percent_rate": None
            if not rel_errors
            else sum(1 for error in rel_errors if error <= 0.10) / len(rel_errors),
            "phase_classification_count": len(phase_matches),
            "phase_classification_accuracy": None
            if not phase_matches
            else sum(1 for matched in phase_matches if matched) / len(phase_matches),
        }
    return {
        "level_summaries": level_summaries,
        "row_count": len(rows),
        "rows": rows,
    }


def _counterfactual_by_level(record: ExperimentRecord) -> dict[str, dict[str, Any]]:
    return {
        str(item["level"]): item
        for item in record.diagnostic_counterfactuals
        if item.get("level") is not None
    }


def _counterfactual_peak(levels: dict[str, dict[str, Any]], level: str) -> int | None:
    item = levels.get(level)
    if item is None or item.get("candidate_peak_bytes") is None:
        return None
    return int(item["candidate_peak_bytes"])


def _counterfactual_phase(levels: dict[str, dict[str, Any]], level: str) -> str | None:
    item = levels.get(level)
    if item is None or item.get("candidate_peak_phase") is None:
        return None
    return str(item["candidate_peak_phase"])


def _simulation_error_sources(record: ExperimentRecord, d3_to_d5_delta: int | None) -> tuple[str, ...]:
    sources: list[str] = []
    if record.measured_peak_phase == "optimizer":
        sources.append("optimizer_fixed_frontier_offset")
    if (
        record.selected_peak_phase is not None
        and record.measured_peak_phase is not None
        and record.selected_peak_phase != record.measured_peak_phase
    ):
        sources.append("peak_phase_mismatch")
    raw_error = record.selected_prediction_error_bytes
    if raw_error is not None and d3_to_d5_delta is not None and abs(d3_to_d5_delta) >= 0.5 * abs(raw_error):
        sources.append("compiler_runtime_offset")
    if record.diagnostic_primary_cause:
        sources.append(f"diagnostic:{record.diagnostic_primary_cause}")
    return tuple(dict.fromkeys(sources or ["unclassified"]))


def summarize_simulation_error_root_causes(
    records: tuple[ExperimentRecord, ...],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status != "ok" or record.selected_prediction_error_bytes is None:
            continue
        levels = _counterfactual_by_level(record)
        d0_peak = _counterfactual_peak(levels, "D0")
        d3_peak = _counterfactual_peak(levels, "D3")
        d5_peak = _counterfactual_peak(levels, "D5")
        d0_to_d3_delta = None if d0_peak is None or d3_peak is None else d3_peak - d0_peak
        d3_to_d5_delta = None if d3_peak is None or d5_peak is None else d5_peak - d3_peak
        calibrated_error = record.selected_calibrated_prediction_error_bytes
        row = {
            "variant_name": record.variant_name,
            "task_name": record.task_name,
            "microbatch_size": record.microbatch_size,
            "budget_bytes": record.budget_bytes,
            "selected_plan_id": record.selected_plan_id,
            "selected_estimated_peak_bytes": record.selected_estimated_peak_bytes,
            "measured_peak_bytes": record.measured_peak_bytes,
            "selected_prediction_error_bytes": record.selected_prediction_error_bytes,
            "selected_prediction_absolute_relative_error": None
            if record.selected_prediction_relative_error is None
            else abs(record.selected_prediction_relative_error),
            "selected_calibrated_prediction_error_bytes": calibrated_error,
            "selected_calibrated_absolute_relative_error": None
            if record.selected_calibrated_prediction_relative_error is None
            else abs(record.selected_calibrated_prediction_relative_error),
            "d0_candidate_peak_bytes": d0_peak,
            "d3_candidate_peak_bytes": d3_peak,
            "d5_candidate_peak_bytes": d5_peak,
            "d0_to_d3_delta_bytes": d0_to_d3_delta,
            "d3_to_d5_delta_bytes": d3_to_d5_delta,
            "selected_peak_phase": record.selected_peak_phase,
            "d3_peak_phase": _counterfactual_phase(levels, "D3"),
            "d5_peak_phase": _counterfactual_phase(levels, "D5"),
            "measured_peak_phase": record.measured_peak_phase,
            "phase_mismatch": (
                record.selected_peak_phase is not None
                and record.measured_peak_phase is not None
                and record.selected_peak_phase != record.measured_peak_phase
            ),
            "diagnostic_primary_cause": record.diagnostic_primary_cause,
            "error_sources": _simulation_error_sources(record, d3_to_d5_delta),
        }
        rows.append(row)

    task_summaries: dict[str, dict[str, Any]] = {}
    for task_name in sorted({row["task_name"] for row in rows}):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        raw_rel = [
            row["selected_prediction_absolute_relative_error"]
            for row in task_rows
            if row["selected_prediction_absolute_relative_error"] is not None
        ]
        calibrated_rel = [
            row["selected_calibrated_absolute_relative_error"]
            for row in task_rows
            if row["selected_calibrated_absolute_relative_error"] is not None
        ]
        d0_to_d3 = [
            abs(row["d0_to_d3_delta_bytes"])
            for row in task_rows
            if row["d0_to_d3_delta_bytes"] is not None
        ]
        d3_to_d5 = [
            abs(row["d3_to_d5_delta_bytes"])
            for row in task_rows
            if row["d3_to_d5_delta_bytes"] is not None
        ]
        task_summaries[task_name] = {
            "row_count": len(task_rows),
            "mean_raw_absolute_relative_error": _mean(raw_rel),
            "max_raw_absolute_relative_error": None if not raw_rel else max(raw_rel),
            "mean_calibrated_absolute_relative_error": _mean(calibrated_rel),
            "max_calibrated_absolute_relative_error": None if not calibrated_rel else max(calibrated_rel),
            "phase_mismatch_count": sum(1 for row in task_rows if row["phase_mismatch"]),
            "measured_peak_phase_counts": _counts([row["measured_peak_phase"] for row in task_rows]),
            "diagnostic_primary_cause_counts": _counts([row["diagnostic_primary_cause"] for row in task_rows]),
            "error_source_counts": _tuple_counts([tuple(row["error_sources"]) for row in task_rows]),
            "mean_abs_d0_to_d3_delta_bytes": _mean(d0_to_d3),
            "mean_abs_d3_to_d5_delta_bytes": _mean(d3_to_d5),
        }

    outliers = sorted(
        rows,
        key=lambda row: (
            row["selected_prediction_absolute_relative_error"] is None,
            -(row["selected_prediction_absolute_relative_error"] or 0.0),
            row["task_name"],
            row["budget_bytes"],
        ),
    )[:top_k]
    return {
        "row_count": len(rows),
        "task_summaries": task_summaries,
        "top_outliers": outliers,
    }


def _record_compile_backend(record: ExperimentRecord) -> str:
    backend = record.config_fingerprint.get("compile_backend")
    if backend is not None:
        return str(backend)
    if record.config_fingerprint.get("enable_inductor"):
        return "inductor"
    if record.config_fingerprint.get("enable_compile"):
        return "aot_eager"
    return "eager"


def summarize_steady_state_phases(records: tuple[ExperimentRecord, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status != "ok":
            continue
        rows.append(
            {
                "variant_name": record.variant_name,
                "task_name": record.task_name,
                "microbatch_size": record.microbatch_size,
                "budget_bytes": record.budget_bytes,
                "compile_backend": _record_compile_backend(record),
                "device": record.config_fingerprint.get("device"),
                "measurement_repeats": record.measurement_repeats,
                "measurement_warmup_steps": record.measurement_warmup_steps,
                "selected_plan_id": record.selected_plan_id,
                "measured_peak_phase": record.measured_peak_phase,
                "measured_peak_bytes": record.measured_peak_bytes,
                "measured_step_us": record.measured_step_us,
                "fw_us": record.measured_fw_us,
                "bw_us": record.measured_bw_us,
                "optimizer_us": record.measured_optimizer_us,
                "fw_peak_bytes": record.measured_fw_peak_bytes,
                "bw_peak_bytes": record.measured_bw_peak_bytes,
                "optimizer_peak_bytes": record.measured_optimizer_peak_bytes,
                "has_phase_timing": all(
                    value is not None
                    for value in (record.measured_fw_us, record.measured_bw_us, record.measured_optimizer_us)
                ),
                "has_phase_peaks": all(
                    value is not None
                    for value in (
                        record.measured_fw_peak_bytes,
                        record.measured_bw_peak_bytes,
                        record.measured_optimizer_peak_bytes,
                    )
                ),
            }
        )

    backend_summaries: dict[str, dict[str, Any]] = {}
    for backend in sorted({row["compile_backend"] for row in rows}):
        backend_rows = [row for row in rows if row["compile_backend"] == backend]
        repeats = [row["measurement_repeats"] for row in backend_rows if row["measurement_repeats"] is not None]
        warmups = [
            row["measurement_warmup_steps"]
            for row in backend_rows
            if row["measurement_warmup_steps"] is not None
        ]
        backend_summaries[backend] = {
            "row_count": len(backend_rows),
            "device_counts": _counts([None if row["device"] is None else str(row["device"]) for row in backend_rows]),
            "min_measurement_repeats": None if not repeats else min(repeats),
            "min_measurement_warmup_steps": None if not warmups else min(warmups),
            "steady_state_record_count": sum(
                1
                for row in backend_rows
                if (row["measurement_repeats"] or 0) >= 2 and (row["measurement_warmup_steps"] or 0) >= 1
            ),
            "phase_timing_record_count": sum(1 for row in backend_rows if row["has_phase_timing"]),
            "phase_peak_record_count": sum(1 for row in backend_rows if row["has_phase_peaks"]),
            "measured_peak_phase_counts": _counts([row["measured_peak_phase"] for row in backend_rows]),
            "mean_step_us": _mean([row["measured_step_us"] for row in backend_rows if row["measured_step_us"] is not None]),
            "mean_fw_us": _mean([row["fw_us"] for row in backend_rows if row["fw_us"] is not None]),
            "mean_bw_us": _mean([row["bw_us"] for row in backend_rows if row["bw_us"] is not None]),
            "mean_optimizer_us": _mean(
                [row["optimizer_us"] for row in backend_rows if row["optimizer_us"] is not None]
            ),
            "optimizer_phase_peak_count": sum(
                1 for row in backend_rows if row["measured_peak_phase"] == "optimizer"
            ),
        }

    return {
        "row_count": len(rows),
        "backend_summaries": backend_summaries,
        "rows": rows,
    }


def _ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_experiment_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_summary_json(summary: ExperimentSummary, path: str | Path) -> None:
    text = json.dumps(experiment_summary_to_dict(summary), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_variant_summary_json(
    summaries: dict[str, ExperimentSummary],
    path: str | Path,
) -> None:
    text = json.dumps(experiment_variant_summaries_to_dict(summaries), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_hint_ablation_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(summarize_hint_ablation(records), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_cache_reuse_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(summarize_cache_reuse(records), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_baseline_comparison_json(
    records: tuple[ExperimentRecord, ...],
    path: str | Path,
    sac_baseline: dict[str, Any] | None = None,
) -> None:
    text = json.dumps(summarize_baseline_comparisons(records, sac_baseline), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_layered_accuracy_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(summarize_layered_simulation_accuracy(records), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_simulation_error_json(
    records: tuple[ExperimentRecord, ...],
    path: str | Path,
    *,
    top_k: int = 10,
) -> None:
    text = json.dumps(summarize_simulation_error_root_causes(records, top_k=top_k), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_steady_state_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(summarize_steady_state_phases(records), indent=2, sort_keys=True)
    _ensure_parent_dir(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_csv(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    rows = experiment_records_to_dicts(records)
    fieldnames = tuple(ExperimentRecord.__dataclass_fields__)
    with _ensure_parent_dir(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
