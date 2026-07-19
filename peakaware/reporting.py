from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peakaware.contracts import EvaluatedPlan, OptimizedTrainingResult
from peakaware.diagnostics import diagnose_plan, render_diagnostic_text


def _plan_row(plan: EvaluatedPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan.plan_id,
        "feasible": plan.feasible,
        "rejection_reason": plan.rejection_reason,
        "estimated_peak_bytes": plan.simulation.estimated_peak_bytes,
        "estimated_step_us": plan.simulation.estimated_step_us,
        "after_fw_retained_bytes": plan.simulation.after_fw_retained_bytes,
        "bw_peak_bytes": plan.simulation.bw_peak_bytes,
        "max_recompute_live_bytes": plan.simulation.max_recompute_live_bytes,
        "risk_score": plan.plan.risk_score,
        "confidence": plan.plan.confidence,
    }


def _prediction_error_row(plan: EvaluatedPlan | None, peak_bytes: int) -> dict[str, Any] | None:
    if plan is None:
        return None
    estimated = int(plan.simulation.estimated_peak_bytes)
    error = int(peak_bytes) - estimated
    relative = None if estimated == 0 else error / estimated
    return {
        "plan_id": plan.plan.plan_id,
        "estimated_peak_bytes": estimated,
        "measured_peak_bytes": int(peak_bytes),
        "error_bytes": error,
        "relative_error": relative,
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
            "step_us": candidate.measured_step_us,
            "correctness_passed": candidate.correctness_passed,
            "prediction_error": _prediction_error_row(
                plans_by_id.get(candidate.plan_id),
                candidate.measured_peak_bytes,
            ),
        }
        for candidate in result.measured_candidates
    ]
    correction_rows = [
        row
        for row in (
            _prediction_error_row(plans_by_id.get(candidate.plan_id), candidate.measured_peak_bytes)
            for candidate in result.measured_candidates
        )
        if row is not None
    ]
    return {
        "selected_plan_id": result.selected_plan.plan_id,
        "graph_key": result.selected_plan.graph_key,
        "selected_saved_value_ids": tuple(sorted(result.selected_plan.saved_value_ids)),
        "estimated_peak_bytes": result.selected_plan.estimated_peak_bytes,
        "estimated_step_us": result.selected_plan.estimated_step_us,
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
            "reserved_peak_bytes": int(result.executable.phase_metrics.get("overall_reserved_peak_bytes", 0)),
            "step_us": result.executable.measured_step_us,
            "correctness_passed": result.executable.correctness_passed,
            "phase_metrics": result.executable.phase_metrics,
        },
        "measured_candidates": measured_candidate_rows,
        "topk_correction": {
            "selected": _prediction_error_row(selected_evaluated, result.executable.measured_peak_bytes),
            "candidates": correction_rows,
        },
        "cache": {
            "layer_hits": result.cache_stats.layer_hits,
            "layer_misses": result.cache_stats.layer_misses,
            "total_hits": result.cache_stats.total_hits,
            "total_misses": result.cache_stats.total_misses,
            "hit_rate": result.cache_stats.hit_rate,
        },
        "plans": [_plan_row(plan) for plan in plans],
        "early_stop": _early_stop_row(result),
        "diagnostic": None
        if diagnostic is None
        else {
            "text": diagnostic_text,
            "primary_cause": diagnostic.primary_cause.name,
            "secondary_causes": [cause.name for cause in diagnostic.secondary_causes],
            "counterfactuals": [
                {
                    "level": item.level,
                    "status": item.status,
                    "peak_gain_bytes": item.peak_gain_bytes,
                    "confidence": item.confidence,
                    "unavailable_reason": item.unavailable_reason,
                }
                for item in diagnostic.counterfactuals
            ],
        },
    }


def summarize_plan_artifact(result: OptimizedTrainingResult) -> dict[str, Any]:
    dry_run = result.dry_run
    return {
        "plan_id": result.selected_plan.plan_id,
        "graph_key": result.selected_plan.graph_key,
        "saved_value_ids": tuple(sorted(result.selected_plan.saved_value_ids)),
        "budget_bytes": result.selected_plan.budget_bytes,
        "safety_margin_bytes": result.selected_plan.safety_margin_bytes,
        "estimated_peak_bytes": result.selected_plan.estimated_peak_bytes,
        "estimated_step_us": result.selected_plan.estimated_step_us,
        "max_recompute_live_bytes": result.selected_plan.max_recompute_live_bytes,
        "recompute_span_ops": result.selected_plan.recompute_span_ops,
        "risk_score": result.selected_plan.risk_score,
        "confidence": result.selected_plan.confidence,
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
