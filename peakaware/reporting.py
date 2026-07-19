from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peakaware.contracts import (
    EvaluatedPlan,
    FailureRecord,
    FeasibilityReport,
    MeasuredExecutable,
    OptimizedTrainingResult,
    PeakSnapshot,
)
from peakaware.diagnostics import PlanDiagnosticReport, diagnose_plan, render_diagnostic_text
from peakaware.search.plan import plan_identity_key


def _effective_saved_value_ids(plan: EvaluatedPlan) -> frozenset[int]:
    return plan.plan.saved_value_ids | plan.plan.mandatory_value_ids


def _selected_effective_saved_value_ids(result: OptimizedTrainingResult) -> frozenset[int]:
    return result.selected_plan.saved_value_ids | result.selected_plan.mandatory_value_ids


def _ranking_provenance(plan: EvaluatedPlan | None = None) -> dict[str, Any]:
    cost_sources = () if plan is None else plan.plan.cost_sources
    return {
        "risk_score": {
            "range": "[0, 1]",
            "direction": "lower_is_better",
            "source": "memory.simulator",
            "components": (
                "dropped_storage_recompute_wave",
                "cost_confidence",
                "workspace_uncertainty",
            ),
            "cost_sources": cost_sources,
        },
        "confidence": {
            "range": "[0, 1]",
            "direction": "higher_is_better",
            "source": "memory.simulator",
            "components": (
                "closure_validity",
                "storage_alias_confidence",
                "cost_provider_confidence",
            ),
            "cost_sources": cost_sources,
        },
        "stable_tie_break": (
            "feasible_first",
            "risk_score",
            "negative_confidence",
            "estimated_peak_bytes",
            "estimated_step_us",
            "plan_id",
        ),
    }


def _peak_snapshot_row(snapshot: PeakSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "phase": snapshot.phase,
        "op_id": snapshot.op_id,
        "live_storage_ids": tuple(sorted(snapshot.live_storage_ids)),
        "live_bytes": snapshot.live_bytes,
        "parameter_bytes": snapshot.parameter_bytes,
        "gradient_bytes": snapshot.gradient_bytes,
        "optimizer_bytes": snapshot.optimizer_bytes,
        "saved_activation_bytes": snapshot.saved_activation_bytes,
        "recomputed_bytes": snapshot.recomputed_bytes,
        "workspace_bytes": snapshot.workspace_bytes,
    }


def _plan_row(plan: EvaluatedPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan.plan_id,
        "plan_key": plan_identity_key(plan.plan.graph_key, _effective_saved_value_ids(plan), plan.plan.budget_bytes),
        "effective_saved_value_ids": tuple(sorted(_effective_saved_value_ids(plan))),
        "feasible": plan.feasible,
        "rejection_reason": plan.rejection_reason,
        "estimated_peak_bytes": plan.simulation.estimated_peak_bytes,
        "estimated_step_us": plan.simulation.estimated_step_us,
        "after_fw_retained_bytes": plan.simulation.after_fw_retained_bytes,
        "bw_peak_bytes": plan.simulation.bw_peak_bytes,
        "max_recompute_live_bytes": plan.simulation.max_recompute_live_bytes,
        "risk_score": plan.plan.risk_score,
        "confidence": plan.plan.confidence,
        "ranking_provenance": _ranking_provenance(plan),
        "peak_snapshot": _peak_snapshot_row(plan.simulation.peak_snapshot),
    }


def _measured_peak_phase(phase_metrics: dict[str, Any]) -> str | None:
    peaks = {
        "fw": int(phase_metrics.get("fw_peak_bytes", 0)),
        "bw": int(phase_metrics.get("bw_peak_bytes", 0)),
        "optimizer": int(phase_metrics.get("optimizer_peak_bytes", 0)),
    }
    if not any(peaks.values()):
        return None
    return max(peaks, key=lambda phase: (peaks[phase], {"fw": 0, "bw": 1, "optimizer": 2}[phase]))


def _prediction_error_row(
    plan: EvaluatedPlan | None,
    peak_bytes: int,
    measured_phase: str | None = None,
) -> dict[str, Any] | None:
    if plan is None:
        return None
    estimated = int(plan.simulation.estimated_peak_bytes)
    error = int(peak_bytes) - estimated
    relative = None if estimated == 0 else error / estimated
    estimated_phase = plan.simulation.peak_snapshot.phase
    estimated_feasible = bool(plan.feasible)
    measured_feasible = int(peak_bytes) <= int(plan.plan.budget_bytes)
    return {
        "plan_id": plan.plan.plan_id,
        "estimated_peak_bytes": estimated,
        "measured_peak_bytes": int(peak_bytes),
        "error_bytes": error,
        "relative_error": relative,
        "estimated_peak_phase": estimated_phase,
        "measured_peak_phase": measured_phase,
        "phase_match": None if measured_phase is None else estimated_phase == measured_phase,
        "estimated_feasible": estimated_feasible,
        "measured_feasible": measured_feasible,
        "feasibility_match": estimated_feasible == measured_feasible,
    }


def _simulation_accuracy_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "candidate_count": 0,
            "mean_absolute_error_bytes": None,
            "max_absolute_error_bytes": None,
            "mean_absolute_relative_error": None,
            "within_10_percent_rate": None,
            "phase_classification_count": 0,
            "phase_classification_accuracy": None,
            "feasible_classification_count": 0,
            "feasible_classification_accuracy": None,
        }
    absolute_errors = [abs(int(row["error_bytes"])) for row in rows]
    relative_errors = [
        abs(float(row["relative_error"]))
        for row in rows
        if row["relative_error"] is not None
    ]
    phase_matches = [bool(row["phase_match"]) for row in rows if row.get("phase_match") is not None]
    feasible_matches = [
        bool(row["feasibility_match"])
        for row in rows
        if row.get("feasibility_match") is not None
    ]
    return {
        "candidate_count": len(rows),
        "mean_absolute_error_bytes": sum(absolute_errors) / len(absolute_errors),
        "max_absolute_error_bytes": max(absolute_errors),
        "mean_absolute_relative_error": None
        if not relative_errors
        else sum(relative_errors) / len(relative_errors),
        "within_10_percent_rate": None
        if not relative_errors
        else sum(1 for error in relative_errors if error <= 0.10) / len(relative_errors),
        "phase_classification_count": len(phase_matches),
        "phase_classification_accuracy": None
        if not phase_matches
        else sum(1 for matched in phase_matches if matched) / len(phase_matches),
        "feasible_classification_count": len(feasible_matches),
        "feasible_classification_accuracy": None
        if not feasible_matches
        else sum(1 for matched in feasible_matches if matched) / len(feasible_matches),
    }


def _early_stop_row(result: OptimizedTrainingResult) -> dict[str, Any] | None:
    if result.analysis is None or result.analysis.early_stop is None:
        return None
    report = result.analysis.early_stop
    return {
        "reason": report.reason,
        "best_plan_id": report.best_plan_id,
        "evidence": {
            "evaluated_plan_count": report.evidence.evaluated_plan_count,
            "feasible_plan_count": report.evidence.feasible_plan_count,
            "best_plan_id": report.evidence.best_plan_id,
            "best_estimated_peak_bytes": report.evidence.best_estimated_peak_bytes,
            "best_estimated_step_us": report.evidence.best_estimated_step_us,
            "best_risk_score": report.evidence.best_risk_score,
            "best_confidence": report.evidence.best_confidence,
            "fixed_peak_lower_bound_bytes": report.evidence.fixed_peak_lower_bound_bytes,
            "budget_bytes": report.evidence.budget_bytes,
        },
    }


def _search_diagnostics_row(result: OptimizedTrainingResult) -> dict[str, Any] | None:
    if result.analysis is None or result.analysis.search_diagnostics is None:
        return None
    diagnostics = result.analysis.search_diagnostics
    return {
        "diagnostic_hints_enabled": diagnostics.diagnostic_hints_enabled,
        "manual_hint_count": diagnostics.manual_hint_count,
        "diagnostic_hint_count": diagnostics.diagnostic_hint_count,
        "diagnostic_hint_kinds": diagnostics.diagnostic_hint_kinds,
        "greedy_plan_count": diagnostics.greedy_plan_count,
        "feasible_before_repair_count": diagnostics.feasible_before_repair_count,
        "repaired_candidate_count": diagnostics.repaired_candidate_count,
        "repair_success_count": diagnostics.repair_success_count,
        "feasible_after_repair_count": diagnostics.feasible_after_repair_count,
        "repaired_plan_ids": diagnostics.repaired_plan_ids,
    }


def _failure_row(record: FailureRecord) -> dict[str, Any]:
    return {
        "stage": record.stage,
        "error_type": record.error_type,
        "message": record.message,
        "recovered": record.recovered,
        "next_fallback": record.next_fallback,
        "applied_adapters": record.applied_adapters,
        "applied_plugins": record.applied_plugins,
    }


def _counterfactual_rows(report: PlanDiagnosticReport) -> list[dict[str, Any]]:
    return [
        {
            "level": item.level,
            "status": item.status,
            "peak_gain_bytes": item.peak_gain_bytes,
            "confidence": item.confidence,
            "unavailable_reason": item.unavailable_reason,
            "baseline_peak": _peak_snapshot_row(item.baseline_peak),
            "candidate_peak": _peak_snapshot_row(item.candidate_peak),
        }
        for item in report.counterfactuals
    ]


def _repair_hint_rows(report: PlanDiagnosticReport) -> list[dict[str, Any]]:
    return [
        {
            "kind": hint.kind,
            "target_ids": hint.target_ids,
            "priority": hint.priority,
            "reason": hint.reason,
        }
        for hint in report.repair_hints
    ]


def _evidence_rows(report: PlanDiagnosticReport) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "root_cause": item.root_cause,
            "metric": item.metric,
            "value": item.value,
            "threshold": item.threshold,
            "direction": item.direction,
            "description": item.description,
        }
        for item in report.evidence
    ]


def _expectation_row(report: PlanDiagnosticReport) -> dict[str, Any]:
    return {
        "strategy_status": report.strategy_expectation_status,
        "strategy_provenance": report.strategy_expectation_provenance,
        "strategy_expected_saved_reduction": report.strategy_expected_saved_reduction,
        "normalized_saved_reduction": report.normalized_saved_reduction,
        "strategy_estimation_gap": report.strategy_estimation_gap,
        "realization_gap": report.realization_gap,
        "total_expectation_gap": report.total_expectation_gap,
    }


def _diagnostic_row(report: PlanDiagnosticReport) -> dict[str, Any]:
    return {
        "plan_id": report.plan_id,
        "text": render_diagnostic_text(report),
        "expected_saved_reduction": report.expected_saved_reduction,
        "expectation": _expectation_row(report),
        "after_fw_retained_reduction": report.after_fw_retained_reduction,
        "bw_recompute_transient_change": report.bw_recompute_transient_change,
        "fixed_frontier_overlap": report.fixed_frontier_overlap,
        "compiler_workspace_allocator_change": report.compiler_workspace_allocator_change,
        "actual_overall_peak_reduction": report.actual_overall_peak_reduction,
        "primary_cause": report.primary_cause.name,
        "secondary_causes": [cause.name for cause in report.secondary_causes],
        "root_causes": list(report.root_causes),
        "confidence": report.confidence,
        "evidence": _evidence_rows(report),
        "repair_hints": _repair_hint_rows(report),
        "counterfactuals": _counterfactual_rows(report),
    }


def _plan_diagnostic_rows(
    plans: tuple[EvaluatedPlan, ...],
    baseline: EvaluatedPlan | None,
    selected_measurement: MeasuredExecutable,
    feasibility: FeasibilityReport,
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    rows = []
    for plan in plans:
        measured = selected_measurement if plan.plan.plan_id == selected_measurement.plan_id else None
        rows.append(
            _diagnostic_row(
                diagnose_plan(
                    baseline,
                    plan,
                    feasibility=feasibility,
                    measured=measured,
                )
            )
        )
    return rows


def summarize_result(result: OptimizedTrainingResult) -> dict[str, Any]:
    plans = result.analysis.baseline_results if result.analysis is not None else ()
    plans_by_id = {plan.plan.plan_id: plan for plan in plans}
    baseline = next((plan for plan in plans if plan.plan.plan_id == "all_save"), plans[0] if plans else None)
    selected_evaluated = plans_by_id.get(result.selected_plan.plan_id)
    diagnostic = None
    diagnostic_text = None
    if baseline is not None and selected_evaluated is not None:
        diagnostic = diagnose_plan(
            baseline,
            selected_evaluated,
            feasibility=result.feasibility,
            measured=result.executable,
        )
        diagnostic_text = render_diagnostic_text(diagnostic)
    measured_candidate_rows = [
        {
            "plan_id": candidate.plan_id,
            "peak_bytes": candidate.measured_peak_bytes,
            "peak_phase": _measured_peak_phase(candidate.phase_metrics),
            "step_us": candidate.measured_step_us,
            "correctness_passed": candidate.correctness_passed,
            "prediction_error": _prediction_error_row(
                plans_by_id.get(candidate.plan_id),
                candidate.measured_peak_bytes,
                _measured_peak_phase(candidate.phase_metrics),
            ),
        }
        for candidate in result.measured_candidates
    ]
    correction_rows = [
        row
        for row in (
            _prediction_error_row(
                plans_by_id.get(candidate.plan_id),
                candidate.measured_peak_bytes,
                _measured_peak_phase(candidate.phase_metrics),
            )
            for candidate in result.measured_candidates
        )
        if row is not None
    ]
    return {
        "selected_plan_id": result.selected_plan.plan_id,
        "selected_plan_key": plan_identity_key(
            result.selected_plan.graph_key,
            _selected_effective_saved_value_ids(result),
            result.selected_plan.budget_bytes,
        ),
        "graph_key": result.selected_plan.graph_key,
        "selected_saved_value_ids": tuple(sorted(result.selected_plan.saved_value_ids)),
        "selected_effective_saved_value_ids": tuple(sorted(_selected_effective_saved_value_ids(result))),
        "estimated_peak_bytes": result.selected_plan.estimated_peak_bytes,
        "estimated_step_us": result.selected_plan.estimated_step_us,
        "selection_objective": getattr(result.executor, "selection_objective", "unknown"),
        "fallback_plan_ids": result.fallback_plan_ids,
        "feasibility": {
            "status": result.feasibility.status,
            "dominant_phase": result.feasibility.dominant_phase,
            "fixed_peak_lower_bound": result.feasibility.fixed_peak_lower_bound,
            "activation_budget_bytes": result.feasibility.activation_budget_bytes,
            "explanations": result.feasibility.explanations,
        },
        "measured": {
            "peak_bytes": result.executable.measured_peak_bytes,
            "peak_phase": _measured_peak_phase(result.executable.phase_metrics),
            "reserved_peak_bytes": int(result.executable.phase_metrics.get("overall_reserved_peak_bytes", 0)),
            "step_us": result.executable.measured_step_us,
            "correctness_passed": result.executable.correctness_passed,
            "phase_metrics": result.executable.phase_metrics,
        },
        "measured_candidates": measured_candidate_rows,
        "topk_correction": {
            "selected": _prediction_error_row(
                selected_evaluated,
                result.executable.measured_peak_bytes,
                _measured_peak_phase(result.executable.phase_metrics),
            ),
            "candidates": correction_rows,
            "simulation_accuracy": _simulation_accuracy_row(correction_rows),
        },
        "cache": {
            "layer_hits": result.cache_stats.layer_hits,
            "layer_misses": result.cache_stats.layer_misses,
            "total_hits": result.cache_stats.total_hits,
            "total_misses": result.cache_stats.total_misses,
            "hit_rate": result.cache_stats.hit_rate,
        },
        "plans": [_plan_row(plan) for plan in plans],
        "capture_failures": []
        if result.analysis is None
        else [_failure_row(record) for record in result.analysis.capture_failures],
        "plan_diagnostics": _plan_diagnostic_rows(plans, baseline, result.executable, result.feasibility),
        "search_diagnostics": _search_diagnostics_row(result),
        "early_stop": _early_stop_row(result),
        "diagnostic": None
        if diagnostic is None
        else {
            "text": diagnostic_text,
            "primary_cause": diagnostic.primary_cause.name,
            "secondary_causes": [cause.name for cause in diagnostic.secondary_causes],
            "confidence": diagnostic.confidence,
            "expectation": _expectation_row(diagnostic),
            "evidence": _evidence_rows(diagnostic),
            "repair_hints": _repair_hint_rows(diagnostic),
            "counterfactuals": _counterfactual_rows(diagnostic),
        },
    }


def summarize_plan_artifact(result: OptimizedTrainingResult) -> dict[str, Any]:
    dry_run = result.dry_run
    plans = result.analysis.baseline_results if result.analysis is not None else ()
    selected_evaluated = next((plan for plan in plans if plan.plan.plan_id == result.selected_plan.plan_id), None)
    return {
        "plan_id": result.selected_plan.plan_id,
        "plan_key": plan_identity_key(
            result.selected_plan.graph_key,
            _selected_effective_saved_value_ids(result),
            result.selected_plan.budget_bytes,
        ),
        "graph_key": result.selected_plan.graph_key,
        "saved_value_ids": tuple(sorted(result.selected_plan.saved_value_ids)),
        "effective_saved_value_ids": tuple(sorted(_selected_effective_saved_value_ids(result))),
        "budget_bytes": result.selected_plan.budget_bytes,
        "safety_margin_bytes": result.selected_plan.safety_margin_bytes,
        "estimated_peak_bytes": result.selected_plan.estimated_peak_bytes,
        "estimated_step_us": result.selected_plan.estimated_step_us,
        "max_recompute_live_bytes": result.selected_plan.max_recompute_live_bytes,
        "recompute_span_ops": result.selected_plan.recompute_span_ops,
        "risk_score": result.selected_plan.risk_score,
        "confidence": result.selected_plan.confidence,
        "ranking_provenance": _ranking_provenance(selected_evaluated),
        "peak_snapshot": None
        if selected_evaluated is None
        else _peak_snapshot_row(selected_evaluated.simulation.peak_snapshot),
        "measured_peak_bytes": result.executable.measured_peak_bytes,
        "measured_step_us": result.executable.measured_step_us,
        "correctness": None
        if dry_run is None
        else {
            "abi_valid": dry_run.abi_valid,
            "outputs_match": dry_run.outputs_match,
            "gradients_match": dry_run.gradients_match,
            "rng_match": dry_run.rng_match,
            "failure_reason": dry_run.failure_reason,
        },
        "fallback_plan_ids": result.fallback_plan_ids,
    }


def render_text_report(result: OptimizedTrainingResult) -> str:
    summary = summarize_result(result)
    lines = [
        f"Selected plan: {summary['selected_plan_id']}",
        f"Feasibility: {summary['feasibility']['status']} ({summary['feasibility']['dominant_phase']})",
        f"Measured peak: {summary['measured']['peak_bytes']} bytes",
        f"Measured step: {summary['measured']['step_us']:.3f} us",
    ]
    if summary["diagnostic"] is not None:
        lines.append(f"Diagnostic: {summary['diagnostic']['text']}")
    return "\n".join(lines)


def export_result_json(result: OptimizedTrainingResult, path: str | Path | None = None) -> str:
    text = json.dumps(summarize_result(result), indent=2, sort_keys=True, default=str)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text


def export_plan_artifact_json(result: OptimizedTrainingResult, path: str | Path | None = None) -> str:
    text = json.dumps(summarize_plan_artifact(result), indent=2, sort_keys=True, default=str)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text
