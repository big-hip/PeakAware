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
    task_name: str
    microbatch_size: int
    budget_bytes: int


@dataclass(frozen=True)
class ExperimentRecord:
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
    measured_peak_bytes: int | None
    measured_peak_reserved_bytes: int | None
    measured_step_us: float | None
    samples_per_second: float | None
    feasibility_status: str | None
    diagnostic_primary_cause: str | None
    measured_candidate_count: int
    selected_prediction_error_bytes: int | None
    selected_prediction_relative_error: float | None
    cache_total_hits: int
    cache_total_misses: int
    cache_hit_rate: float | None
    candidate_count: int
    fallback_plan_ids: tuple[str, ...]
    exact_plan_id: str | None = None
    exact_plan_key: str | None = None
    exact_estimated_peak_bytes: int | None = None
    exact_estimated_step_us: float | None = None
    selected_exact_peak_gap_bytes: int | None = None
    exact_error_type: str | None = None
    exact_error_message: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def _record_success(
    case: ExperimentCase,
    summary: dict[str, Any],
    exact: dict[str, Any] | None = None,
) -> ExperimentRecord:
    measured = summary["measured"]
    diagnostic = summary.get("diagnostic")
    selected_correction = summary.get("topk_correction", {}).get("selected")
    cache = summary.get("cache", {})
    step_us = float(measured["step_us"])
    exact = exact or {}
    return ExperimentRecord(
        task_name=case.task_name,
        microbatch_size=case.microbatch_size,
        budget_bytes=case.budget_bytes,
        status="ok",
        selected_plan_id=summary["selected_plan_id"],
        selected_plan_key=summary["selected_plan_key"],
        graph_key=summary["graph_key"],
        selected_saved_value_ids=tuple(summary["selected_saved_value_ids"]),
        selected_effective_saved_value_ids=tuple(summary["selected_effective_saved_value_ids"]),
        selected_estimated_peak_bytes=int(summary["estimated_peak_bytes"]),
        measured_peak_bytes=int(measured["peak_bytes"]),
        measured_peak_reserved_bytes=int(measured.get("reserved_peak_bytes", 0)),
        measured_step_us=step_us,
        samples_per_second=case.microbatch_size * 1_000_000.0 / max(step_us, 1.0),
        feasibility_status=summary["feasibility"]["status"],
        diagnostic_primary_cause=None if diagnostic is None else diagnostic["primary_cause"],
        measured_candidate_count=len(summary["measured_candidates"]),
        selected_prediction_error_bytes=None if selected_correction is None else selected_correction["error_bytes"],
        selected_prediction_relative_error=None if selected_correction is None else selected_correction["relative_error"],
        cache_total_hits=int(cache.get("total_hits", 0)),
        cache_total_misses=int(cache.get("total_misses", 0)),
        cache_hit_rate=cache.get("hit_rate"),
        candidate_count=len(summary["plans"]),
        fallback_plan_ids=tuple(summary["fallback_plan_ids"]),
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
        microbatch_size=case.microbatch_size,
        budget_bytes=case.budget_bytes,
        status="failed",
        selected_plan_id=None,
        selected_plan_key=None,
        graph_key=None,
        selected_saved_value_ids=(),
        selected_effective_saved_value_ids=(),
        selected_estimated_peak_bytes=None,
        measured_peak_bytes=None,
        measured_peak_reserved_bytes=None,
        measured_step_us=None,
        samples_per_second=None,
        feasibility_status=None,
        diagnostic_primary_cause=None,
        measured_candidate_count=0,
        selected_prediction_error_bytes=None,
        selected_prediction_relative_error=None,
        cache_total_hits=0,
        cache_total_misses=0,
        cache_hit_rate=None,
        candidate_count=0,
        fallback_plan_ids=(),
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
                case = ExperimentCase(task_name, microbatch_size, budget)
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


def write_experiment_json(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    text = json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_experiment_csv(records: tuple[ExperimentRecord, ...], path: str | Path) -> None:
    rows = experiment_records_to_dicts(records)
    fieldnames = tuple(ExperimentRecord.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
