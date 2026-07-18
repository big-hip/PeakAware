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


def summarize_result(result: OptimizedTrainingResult) -> dict[str, Any]:
    plans = result.analysis.baseline_results if result.analysis is not None else ()
    baseline = next((plan for plan in plans if plan.plan.plan_id == "all_save"), plans[0] if plans else None)
    selected_evaluated = next((plan for plan in plans if plan.plan.plan_id == result.selected_plan.plan_id), None)
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
    return {
        "selected_plan_id": result.selected_plan.plan_id,
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
            "step_us": result.executable.measured_step_us,
            "correctness_passed": result.executable.correctness_passed,
            "phase_metrics": result.executable.phase_metrics,
        },
        "plans": [_plan_row(plan) for plan in plans],
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
