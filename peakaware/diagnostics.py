from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from peakaware.contracts import EvaluatedPlan, FeasibilityReport, MeasuredExecutable, PeakSnapshot, RepairHint


class RootCause(Enum):
    UNKNOWN = auto()
    SAVED_ESTIMATE_MISMATCH = auto()
    ALIAS_OR_VIEW_PINNING = auto()
    MANDATORY_SAVE_DOMINANCE = auto()
    REMATERIALIZATION_WAVE = auto()
    SHARED_RECOMPUTE_EXPANSION = auto()
    FIXED_BACKWARD_FRONTIER = auto()
    PEAK_PHASE_MIGRATION = auto()
    COMPILER_MATERIALIZATION = auto()
    WORKSPACE_GROWTH = auto()
    ALLOCATOR_FRAGMENTATION = auto()
    COST_MODEL_MISRANK = auto()
    DYNAMIC_SHAPE_DRIFT = auto()
    MEASUREMENT_NOISE = auto()


@dataclass(frozen=True)
class CounterfactualResult:
    level: str
    status: str
    unavailable_reason: str | None
    baseline_peak: PeakSnapshot | None
    candidate_peak: PeakSnapshot | None
    peak_gain_bytes: int | None
    confidence: float


@dataclass(frozen=True)
class PlanDiagnosticReport:
    plan_id: str
    expected_saved_reduction: int
    after_fw_retained_reduction: int
    bw_recompute_transient_change: int
    fixed_frontier_overlap: int
    compiler_workspace_allocator_change: int
    actual_overall_peak_reduction: int
    primary_cause: RootCause
    secondary_causes: tuple[RootCause, ...]
    root_causes: tuple[str, ...]
    repair_hints: tuple[RepairHint, ...]
    counterfactuals: tuple[CounterfactualResult, ...] = ()


def _snapshot_with_bytes(source: PeakSnapshot, total_bytes: int, confidence: float) -> PeakSnapshot:
    del confidence
    return PeakSnapshot(
        phase=source.phase,
        op_id=source.op_id,
        live_storage_ids=source.live_storage_ids,
        live_bytes=total_bytes,
        parameter_bytes=source.parameter_bytes,
        gradient_bytes=source.gradient_bytes,
        optimizer_bytes=source.optimizer_bytes,
        saved_activation_bytes=source.saved_activation_bytes,
        recomputed_bytes=source.recomputed_bytes,
        workspace_bytes=source.workspace_bytes,
    )


def _counterfactual(
    level: str,
    status: str,
    baseline_peak: PeakSnapshot | None,
    candidate_peak: PeakSnapshot | None,
    confidence: float,
    unavailable_reason: str | None = None,
) -> CounterfactualResult:
    gain = None
    if baseline_peak is not None and candidate_peak is not None:
        gain = baseline_peak.live_bytes - candidate_peak.live_bytes
    return CounterfactualResult(level, status, unavailable_reason, baseline_peak, candidate_peak, gain, confidence)


def build_counterfactual_ladder(
    baseline: EvaluatedPlan,
    candidate: EvaluatedPlan,
    *,
    feasibility: FeasibilityReport | None = None,
    measured: MeasuredExecutable | None = None,
) -> tuple[CounterfactualResult, ...]:
    base = baseline.simulation
    cand = candidate.simulation
    d0_baseline = _snapshot_with_bytes(base.peak_snapshot, base.after_fw_retained_bytes, base.confidence)
    d0_candidate = _snapshot_with_bytes(cand.peak_snapshot, cand.after_fw_retained_bytes, cand.confidence)
    d1_baseline = d0_baseline
    d1_candidate = d0_candidate
    d2_baseline = _snapshot_with_bytes(base.peak_snapshot, base.after_fw_retained_bytes + base.max_recompute_live_bytes, base.confidence)
    d2_candidate = _snapshot_with_bytes(cand.peak_snapshot, cand.after_fw_retained_bytes + cand.max_recompute_live_bytes, cand.confidence)
    fixed = feasibility.fixed_peak_lower_bound if feasibility is not None else 0
    d3_baseline = _snapshot_with_bytes(base.peak_snapshot, max(base.estimated_peak_bytes, fixed), base.confidence)
    d3_candidate = _snapshot_with_bytes(cand.peak_snapshot, max(cand.estimated_peak_bytes, fixed), cand.confidence)
    ladder = [
        _counterfactual("D0", "available", d0_baseline, d0_candidate, min(base.confidence, cand.confidence)),
        _counterfactual("D1", "available", d1_baseline, d1_candidate, min(base.confidence, cand.confidence)),
        _counterfactual("D2", "available", d2_baseline, d2_candidate, min(base.confidence, cand.confidence)),
        _counterfactual("D3", "available", d3_baseline, d3_candidate, min(base.confidence, cand.confidence)),
    ]
    if measured is None:
        ladder.append(_counterfactual("D4", "unavailable", None, None, 0.0, "candidate was not compiler-corrected"))
        ladder.append(_counterfactual("D5", "unavailable", None, None, 0.0, "candidate was not runtime-measured"))
    else:
        measured_snapshot = _snapshot_with_bytes(
            cand.peak_snapshot,
            measured.measured_peak_bytes,
            0.9 if measured.correctness_passed else 0.2,
        )
        ladder.append(_counterfactual("D4", "estimated", d3_baseline, measured_snapshot, 0.6))
        ladder.append(_counterfactual("D5", "available", d3_baseline, measured_snapshot, 0.9 if measured.correctness_passed else 0.2))
    return tuple(ladder)


def rank_root_causes(
    *,
    expected_saved: int,
    transient: int,
    peak_reduction: int,
    baseline: EvaluatedPlan,
    candidate: EvaluatedPlan,
    feasibility: FeasibilityReport | None,
) -> tuple[RootCause, tuple[RootCause, ...]]:
    causes: list[RootCause] = []
    if any(effect.pinning_value_ids for effect in candidate.plan.storage_effects):
        causes.append(RootCause.ALIAS_OR_VIEW_PINNING)
    if transient > 0:
        causes.append(RootCause.REMATERIALIZATION_WAVE)
    if expected_saved > 0 and peak_reduction <= 0:
        if baseline.simulation.peak_snapshot.phase != candidate.simulation.peak_snapshot.phase:
            causes.append(RootCause.PEAK_PHASE_MIGRATION)
        else:
            causes.append(RootCause.FIXED_BACKWARD_FRONTIER)
    if feasibility is not None and feasibility.status != "FEASIBLE":
        causes.append(RootCause.FIXED_BACKWARD_FRONTIER)
    if not causes:
        causes.append(RootCause.UNKNOWN)
    deduped = tuple(dict.fromkeys(causes))
    return deduped[0], deduped[1:]


def diagnose_plan(
    baseline: EvaluatedPlan,
    candidate: EvaluatedPlan,
    feasibility: FeasibilityReport | None = None,
    measured: MeasuredExecutable | None = None,
) -> PlanDiagnosticReport:
    expected_saved = (
        baseline.simulation.after_fw_retained_bytes
        - candidate.simulation.after_fw_retained_bytes
    )
    transient = candidate.simulation.max_recompute_live_bytes - baseline.simulation.max_recompute_live_bytes
    peak_reduction = baseline.simulation.estimated_peak_bytes - candidate.simulation.estimated_peak_bytes
    hints: list[RepairHint] = []
    if transient > 0:
        hints.append(
            RepairHint(
                kind="SAVE_PEAK_STORAGE",
                target_ids=tuple(sorted(candidate.simulation.peak_snapshot.live_storage_ids)),
                priority=0.8,
                reason="recompute transient increased at peak",
            )
        )
    fixed_overlap = 0
    if feasibility is not None:
        fixed_overlap = max(0, feasibility.fixed_peak_lower_bound - candidate.simulation.estimated_peak_bytes)
    primary, secondary = rank_root_causes(
        expected_saved=expected_saved,
        transient=transient,
        peak_reduction=peak_reduction,
        baseline=baseline,
        candidate=candidate,
        feasibility=feasibility,
    )
    causes = tuple(cause.name for cause in (primary,) + secondary)
    return PlanDiagnosticReport(
        plan_id=candidate.plan.plan_id,
        expected_saved_reduction=expected_saved,
        after_fw_retained_reduction=expected_saved,
        bw_recompute_transient_change=transient,
        fixed_frontier_overlap=fixed_overlap,
        compiler_workspace_allocator_change=0,
        actual_overall_peak_reduction=peak_reduction,
        primary_cause=primary,
        secondary_causes=secondary,
        root_causes=causes,
        repair_hints=tuple(hints),
        counterfactuals=build_counterfactual_ladder(
            baseline,
            candidate,
            feasibility=feasibility,
            measured=measured,
        ),
    )


def render_diagnostic_text(report: PlanDiagnosticReport) -> str:
    causes = ", ".join(report.root_causes) if report.root_causes else RootCause.UNKNOWN.name
    return (
        f"{report.plan_id}: peak_reduction={report.actual_overall_peak_reduction} bytes, "
        f"after_fw_reduction={report.after_fw_retained_reduction} bytes, "
        f"primary={report.primary_cause.name}, causes={causes}"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported diagnostic JSON value: {type(value).__name__}")


def export_diagnostic_json(report: PlanDiagnosticReport, path: str | Path | None = None) -> str:
    text = json.dumps(asdict(report), default=_json_default, indent=2, sort_keys=True)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text
