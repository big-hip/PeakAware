from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from peakaware.api import optimize_training
from peakaware.config import PeakAwareConfig
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import summarize_result
from peakaware.search.exact import solve_exact_small_graph
from peakaware.search.plan import plan_identity_key


@dataclass(frozen=True)
class ExperimentCase:
    variant_name: str
    task_name: str
    microbatch_size: int
    budget_bytes: int


@dataclass(frozen=True)
class ExperimentRecord:
    variant_name: str
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
    measured_step_us: float | None
    samples_per_second: float | None
    feasibility_status: str | None
    baseline_peak_phase: str | None
    selected_peak_phase: str | None
    diagnostic_primary_cause: str | None
    diagnostic_normalized_saved_reduction_bytes: int | None
    diagnostic_realization_gap_bytes: int | None
    diagnostic_total_expectation_gap_bytes: int | None
    measured_candidate_count: int
    selected_prediction_error_bytes: int | None
    selected_prediction_relative_error: float | None
    simulation_accuracy_candidate_count: int
    simulation_accuracy_mean_absolute_error_bytes: float | None
    simulation_accuracy_max_absolute_error_bytes: int | None
    simulation_accuracy_mean_absolute_relative_error: float | None
    simulation_accuracy_within_10_percent_rate: float | None
    cache_total_hits: int
    cache_total_misses: int
    cache_hit_rate: float | None
    candidate_count: int
    fallback_plan_ids: tuple[str, ...]
    diagnostic_hints_enabled: bool | None = None
    diagnostic_hint_count: int = 0
    diagnostic_hint_kinds: tuple[str, ...] = ()
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


@dataclass(frozen=True)
class ExperimentSummary:
    total_records: int
    ok_records: int
    failed_records: int
    success_rate: float | None
    variant_counts: dict[str, int]
    budget_violation_count: int
    budget_violation_rate: float | None
    max_feasible_microbatch: int | None
    mean_samples_per_second: float | None
    mean_measured_step_us: float | None
    mean_measured_peak_bytes: float | None
    mean_measured_budget_headroom_bytes: float | None
    mean_estimated_peak_reduction_bytes: float | None
    aggregate_cache_hit_rate: float | None
    total_cache_hits: int
    total_cache_misses: int
    selected_prediction_count: int
    mean_selected_prediction_absolute_error_bytes: float | None
    max_selected_prediction_absolute_error_bytes: int | None
    mean_selected_prediction_absolute_relative_error: float | None
    simulation_accuracy_candidate_count: int
    mean_simulation_accuracy_absolute_error_bytes: float | None
    max_simulation_accuracy_absolute_error_bytes: int | None
    mean_simulation_accuracy_absolute_relative_error: float | None
    mean_simulation_accuracy_within_10_percent_rate: float | None
    root_cause_counts: dict[str, int]
    selected_peak_phase_counts: dict[str, int]
    diagnostic_hints_enabled_count: int
    diagnostic_hint_count: int
    diagnostic_hint_kind_counts: dict[str, int]
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


def _record_success(
    case: ExperimentCase,
    summary: dict[str, Any],
    exact: dict[str, Any] | None = None,
) -> ExperimentRecord:
    measured = summary["measured"]
    diagnostic = summary.get("diagnostic")
    expectation = {} if diagnostic is None else diagnostic.get("expectation", {})
    selected_correction = summary.get("topk_correction", {}).get("selected")
    simulation_accuracy = summary.get("topk_correction", {}).get("simulation_accuracy", {})
    search_diagnostics = summary.get("search_diagnostics") or {}
    baseline = _plan_by_id(summary, "all_save")
    selected_plan = _plan_by_id(summary, summary["selected_plan_id"])
    baseline_peak = None if baseline is None else baseline.get("peak_snapshot")
    selected_peak = None if selected_plan is None else selected_plan.get("peak_snapshot")
    baseline_peak_bytes = None if baseline is None else int(baseline["estimated_peak_bytes"])
    selected_peak_bytes = int(summary["estimated_peak_bytes"])
    cache = summary.get("cache", {})
    step_us = float(measured["step_us"])
    exact = exact or {}
    return ExperimentRecord(
        task_name=case.task_name,
        variant_name=case.variant_name,
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
        measured_peak_bytes=int(measured["peak_bytes"]),
        measured_peak_reserved_bytes=int(measured.get("reserved_peak_bytes", 0)),
        measured_budget_headroom_bytes=case.budget_bytes - int(measured["peak_bytes"]),
        measured_step_us=step_us,
        samples_per_second=case.microbatch_size * 1_000_000.0 / max(step_us, 1.0),
        feasibility_status=summary["feasibility"]["status"],
        baseline_peak_phase=None if baseline_peak is None else baseline_peak["phase"],
        selected_peak_phase=None if selected_peak is None else selected_peak["phase"],
        diagnostic_primary_cause=None if diagnostic is None else diagnostic["primary_cause"],
        diagnostic_normalized_saved_reduction_bytes=expectation.get("normalized_saved_reduction"),
        diagnostic_realization_gap_bytes=expectation.get("realization_gap"),
        diagnostic_total_expectation_gap_bytes=expectation.get("total_expectation_gap"),
        measured_candidate_count=len(summary["measured_candidates"]),
        selected_prediction_error_bytes=None if selected_correction is None else selected_correction["error_bytes"],
        selected_prediction_relative_error=None if selected_correction is None else selected_correction["relative_error"],
        simulation_accuracy_candidate_count=int(simulation_accuracy.get("candidate_count", 0)),
        simulation_accuracy_mean_absolute_error_bytes=simulation_accuracy.get("mean_absolute_error_bytes"),
        simulation_accuracy_max_absolute_error_bytes=simulation_accuracy.get("max_absolute_error_bytes"),
        simulation_accuracy_mean_absolute_relative_error=simulation_accuracy.get("mean_absolute_relative_error"),
        simulation_accuracy_within_10_percent_rate=simulation_accuracy.get("within_10_percent_rate"),
        cache_total_hits=int(cache.get("total_hits", 0)),
        cache_total_misses=int(cache.get("total_misses", 0)),
        cache_hit_rate=cache.get("hit_rate"),
        candidate_count=len(summary["plans"]),
        fallback_plan_ids=tuple(summary["fallback_plan_ids"]),
        diagnostic_hints_enabled=search_diagnostics.get("diagnostic_hints_enabled"),
        diagnostic_hint_count=int(search_diagnostics.get("diagnostic_hint_count", 0)),
        diagnostic_hint_kinds=tuple(search_diagnostics.get("diagnostic_hint_kinds", ())),
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
    )


def _record_failure(case: ExperimentCase, exc: Exception) -> ExperimentRecord:
    return ExperimentRecord(
        task_name=case.task_name,
        variant_name=case.variant_name,
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
        measured_step_us=None,
        samples_per_second=None,
        feasibility_status=None,
        baseline_peak_phase=None,
        selected_peak_phase=None,
        diagnostic_primary_cause=None,
        diagnostic_normalized_saved_reduction_bytes=None,
        diagnostic_realization_gap_bytes=None,
        diagnostic_total_expectation_gap_bytes=None,
        measured_candidate_count=0,
        selected_prediction_error_bytes=None,
        selected_prediction_relative_error=None,
        simulation_accuracy_candidate_count=0,
        simulation_accuracy_mean_absolute_error_bytes=None,
        simulation_accuracy_max_absolute_error_bytes=None,
        simulation_accuracy_mean_absolute_relative_error=None,
        simulation_accuracy_within_10_percent_rate=None,
        cache_total_hits=0,
        cache_total_misses=0,
        cache_hit_rate=None,
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
    registry = registry or TrainingTaskRegistry.with_defaults()
    config = config or PeakAwareConfig()
    records: list[ExperimentRecord] = []
    for task_name in task_names:
        task = registry.get(task_name)
        for microbatch_size in microbatch_sizes:
            for budget in budget_bytes:
                case = ExperimentCase(variant_name, task_name, microbatch_size, budget)
                model = task.build_model()
                optimizer = task.build_optimizer(model)
                args, kwargs = task.build_batch(microbatch_size)
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


def _mean(values: list[float | int]) -> float | None:
    return None if not values else sum(values) / len(values)


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
        variant_counts=_counts([record.variant_name for record in records]),
        budget_violation_count=len(violations),
        budget_violation_rate=None if not ok else len(violations) / len(ok),
        max_feasible_microbatch=max(feasible_microbatches) if feasible_microbatches else None,
        mean_samples_per_second=_mean(samples),
        mean_measured_step_us=_mean(steps),
        mean_measured_peak_bytes=_mean(peaks),
        mean_measured_budget_headroom_bytes=_mean(headrooms),
        mean_estimated_peak_reduction_bytes=_mean(reductions),
        aggregate_cache_hit_rate=None if total_hits + total_misses == 0 else total_hits / (total_hits + total_misses),
        total_cache_hits=total_hits,
        total_cache_misses=total_misses,
        selected_prediction_count=len(selected_abs_errors),
        mean_selected_prediction_absolute_error_bytes=_mean(selected_abs_errors),
        max_selected_prediction_absolute_error_bytes=None if not selected_abs_errors else max(selected_abs_errors),
        mean_selected_prediction_absolute_relative_error=_mean(selected_abs_relative_errors),
        simulation_accuracy_candidate_count=sum(simulation_accuracy_counts),
        mean_simulation_accuracy_absolute_error_bytes=_mean(simulation_accuracy_mean_abs_errors),
        max_simulation_accuracy_absolute_error_bytes=None
        if not simulation_accuracy_max_abs_errors
        else max(simulation_accuracy_max_abs_errors),
        mean_simulation_accuracy_absolute_relative_error=_mean(simulation_accuracy_mean_relative_errors),
        mean_simulation_accuracy_within_10_percent_rate=_mean(simulation_accuracy_within_10),
        root_cause_counts=_counts([record.diagnostic_primary_cause for record in ok]),
        selected_peak_phase_counts=_counts([record.selected_peak_phase for record in ok]),
        diagnostic_hints_enabled_count=sum(1 for record in ok if record.diagnostic_hints_enabled),
        diagnostic_hint_count=sum(record.diagnostic_hint_count for record in ok),
        diagnostic_hint_kind_counts=_tuple_counts([record.diagnostic_hint_kinds for record in ok]),
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


def write_experiment_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_summary_json(summary: ExperimentSummary, path: str | Path) -> None:
    text = json.dumps(experiment_summary_to_dict(summary), indent=2, sort_keys=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_csv(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    rows = experiment_records_to_dicts(records)
    fieldnames = tuple(ExperimentRecord.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
