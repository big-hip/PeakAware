from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from peakaware.contracts import (
    EvaluatedPlan,
    FeasibilityReport,
    MeasuredExecutable,
    PeakSnapshot,
    RecomputePlan,
    SimulationResult,
    StorageEffect,
)
from peakaware.diagnostics import (
    PlanDiagnosticReport,
    RootCause,
    RootCauseGroundTruth,
    RootCausePrediction,
    diagnose_plan,
    evaluate_root_cause_predictions,
)


@dataclass(frozen=True)
class SyntheticRootCauseCase:
    name: str
    label: RootCauseGroundTruth
    report: PlanDiagnosticReport


def _snapshot(
    *,
    phase: str,
    live_bytes: int,
    saved_activation_bytes: int,
    recomputed_bytes: int = 0,
) -> PeakSnapshot:
    return PeakSnapshot(
        phase=phase,
        op_id=1,
        live_storage_ids=frozenset({1, 2}),
        live_bytes=live_bytes,
        parameter_bytes=256,
        gradient_bytes=256,
        optimizer_bytes=512,
        saved_activation_bytes=saved_activation_bytes,
        recomputed_bytes=recomputed_bytes,
        workspace_bytes=0,
    )


def _effect(*, pinned: bool = False) -> StorageEffect:
    return StorageEffect(
        storage_id=1,
        decision="DROP",
        decision_value_ids=(1,),
        alias_value_ids=(1, 2) if pinned else (1,),
        released_at_peak_bytes=0 if pinned else 128,
        retained_after_fw_bytes=128 if pinned else 0,
        pinning_value_ids=(2,) if pinned else (),
        confidence=0.9,
    )


def _evaluated_plan(
    *,
    plan_id: str,
    after_fw_retained_bytes: int,
    estimated_peak_bytes: int,
    phase: str,
    max_recompute_live_bytes: int = 0,
    estimated_step_us: float = 100.0,
    effects: tuple[StorageEffect, ...] = (),
) -> EvaluatedPlan:
    simulation = SimulationResult(
        plan_id=plan_id,
        estimated_peak_bytes=estimated_peak_bytes,
        estimated_step_us=estimated_step_us,
        peak_snapshot=_snapshot(
            phase=phase,
            live_bytes=estimated_peak_bytes,
            saved_activation_bytes=after_fw_retained_bytes,
            recomputed_bytes=max_recompute_live_bytes,
        ),
        after_fw_retained_bytes=after_fw_retained_bytes,
        fw_peak_bytes=estimated_peak_bytes if phase == "fw" else after_fw_retained_bytes,
        bw_peak_bytes=estimated_peak_bytes if phase == "bw" else max_recompute_live_bytes,
        optimizer_peak_bytes=estimated_peak_bytes if phase == "optimizer" else 1024,
        max_recompute_live_bytes=max_recompute_live_bytes,
        recompute_span_ops=2 if max_recompute_live_bytes else 0,
        recompute_before_first_bw_op_bytes=max_recompute_live_bytes,
        risk_score=0.1 if max_recompute_live_bytes == 0 else 0.6,
        confidence=0.9,
    )
    plan = RecomputePlan(
        graph_key="synthetic-root-cause",
        budget_bytes=4096,
        storage_effects=effects,
        saved_value_ids=frozenset({1, 2}),
        mandatory_value_ids=frozenset(),
        estimated_peak_bytes=estimated_peak_bytes,
        estimated_step_us=estimated_step_us,
        max_recompute_live_bytes=max_recompute_live_bytes,
        recompute_span_ops=simulation.recompute_span_ops,
        recompute_before_first_bw_op_bytes=max_recompute_live_bytes,
        risk_score=simulation.risk_score,
        confidence=simulation.confidence,
        safety_margin_bytes=0,
        cost_sources=("synthetic",),
        plan_id=plan_id,
    )
    return EvaluatedPlan(plan=plan, simulation=simulation, feasible=True, rejection_reason=None)


def _feasibility(status: str) -> FeasibilityReport:
    return FeasibilityReport(
        user_budget_bytes=4096,
        fixed_peak_lower_bound=2200,
        activation_budget_bytes=1896,
        dominant_phase="bw",
        status=status,
        explanations=(f"synthetic {status.lower()} fixed frontier",),
    )


def _measured(plan_id: str, *, peak_bytes: int, step_us: float) -> MeasuredExecutable:
    return MeasuredExecutable(
        plan_id=plan_id,
        forward_backward=lambda x: x,
        measured_peak_bytes=peak_bytes,
        measured_step_us=step_us,
        correctness_passed=True,
    )


def _label(plan_id: str, primary: RootCause, root_causes: tuple[RootCause, ...] | None = None) -> RootCauseGroundTruth:
    return RootCauseGroundTruth(
        plan_id=plan_id,
        primary_cause=primary,
        root_causes=root_causes or (primary,),
    )


def _synthetic_cases() -> tuple[SyntheticRootCauseCase, ...]:
    baseline = _evaluated_plan(
        plan_id="all_save",
        after_fw_retained_bytes=1000,
        estimated_peak_bytes=2000,
        phase="fw",
    )
    specs: list[tuple[str, EvaluatedPlan, RootCauseGroundTruth, FeasibilityReport | None, MeasuredExecutable | None]] = [
        (
            "alias pinning",
            _evaluated_plan(
                plan_id="alias_pinning",
                after_fw_retained_bytes=1000,
                estimated_peak_bytes=2000,
                phase="fw",
                effects=(_effect(pinned=True),),
            ),
            _label("alias_pinning", RootCause.ALIAS_OR_VIEW_PINNING),
            None,
            None,
        ),
        (
            "rematerialization wave",
            _evaluated_plan(
                plan_id="rematerialization_wave",
                after_fw_retained_bytes=700,
                estimated_peak_bytes=1800,
                phase="bw",
                max_recompute_live_bytes=512,
                effects=(_effect(),),
            ),
            _label("rematerialization_wave", RootCause.REMATERIALIZATION_WAVE),
            None,
            None,
        ),
        (
            "fixed backward frontier",
            _evaluated_plan(
                plan_id="fixed_backward_frontier",
                after_fw_retained_bytes=600,
                estimated_peak_bytes=2000,
                phase="fw",
                effects=(_effect(),),
            ),
            _label("fixed_backward_frontier", RootCause.FIXED_BACKWARD_FRONTIER),
            _feasibility("LIMITED_ACTIVATION_HEADROOM"),
            None,
        ),
        (
            "peak phase migration",
            _evaluated_plan(
                plan_id="peak_phase_migration",
                after_fw_retained_bytes=600,
                estimated_peak_bytes=2100,
                phase="bw",
                effects=(_effect(),),
            ),
            _label("peak_phase_migration", RootCause.PEAK_PHASE_MIGRATION),
            None,
            None,
        ),
        (
            "workspace growth and cost misrank",
            _evaluated_plan(
                plan_id="workspace_growth_cost_misrank",
                after_fw_retained_bytes=1000,
                estimated_peak_bytes=2000,
                phase="fw",
                estimated_step_us=100.0,
            ),
            _label(
                "workspace_growth_cost_misrank",
                RootCause.WORKSPACE_GROWTH,
                (RootCause.WORKSPACE_GROWTH, RootCause.COST_MODEL_MISRANK),
            ),
            None,
            _measured("workspace_growth_cost_misrank", peak_bytes=2000 + (4 << 20), step_us=5000.0),
        ),
    ]
    return tuple(
        SyntheticRootCauseCase(
            name=name,
            label=label,
            report=diagnose_plan(baseline, candidate, feasibility=feasibility, measured=measured),
        )
        for name, candidate, label, feasibility, measured in specs
    )


def run_synthetic_root_cause_benchmark() -> dict[str, Any]:
    cases = _synthetic_cases()
    predictions = tuple(
        RootCausePrediction(
            plan_id=case.report.plan_id,
            primary_cause=case.report.primary_cause,
            root_causes=tuple(case.report.root_causes),
        )
        for case in cases
    )
    labels = tuple(case.label for case in cases)
    evaluation = evaluate_root_cause_predictions(predictions, labels)
    rows = []
    for case in cases:
        predicted = set(case.report.root_causes) - {RootCause.UNKNOWN.name}
        expected = {cause.name if isinstance(cause, RootCause) else str(cause) for cause in case.label.root_causes}
        rows.append(
            {
                "name": case.name,
                "plan_id": case.report.plan_id,
                "expected_primary_cause": case.label.primary_cause.name
                if isinstance(case.label.primary_cause, RootCause)
                else str(case.label.primary_cause),
                "predicted_primary_cause": case.report.primary_cause.name,
                "expected_root_causes": sorted(expected),
                "predicted_root_causes": sorted(predicted),
                "matched_root_causes": sorted(predicted & expected),
                "missed_root_causes": sorted(expected - predicted),
                "extra_root_causes": sorted(predicted - expected),
                "confidence": case.report.confidence,
                "evidence_ids": tuple(item.evidence_id for item in case.report.evidence),
            }
        )
    return {
        "evaluation": asdict(evaluation),
        "case_count": len(cases),
        "rows": rows,
    }
