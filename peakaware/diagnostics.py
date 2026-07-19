from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
class DiagnosticEvidence:
    evidence_id: str
    root_cause: str
    metric: str
    value: int | float | str | None
    threshold: int | float | str | None
    direction: str
    description: str


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
    evidence: tuple[DiagnosticEvidence, ...]
    repair_hints: tuple[RepairHint, ...]
    counterfactuals: tuple[CounterfactualResult, ...] = ()
    strategy_expectation_status: str = "unavailable"
    strategy_expectation_provenance: dict[str, Any] = field(default_factory=dict)
    strategy_expected_saved_reduction: int | None = None
    normalized_saved_reduction: int = 0
    strategy_estimation_gap: int | None = None
    realization_gap: int = 0
    total_expectation_gap: int | None = None


@dataclass(frozen=True)
class RootCauseGroundTruth:
    plan_id: str
    primary_cause: RootCause | str
    root_causes: tuple[RootCause | str, ...]


@dataclass(frozen=True)
class RootCauseEvaluation:
    case_count: int
    matched_case_count: int
    missing_prediction_count: int
    unknown_prediction_count: int
    primary_accuracy: float | None
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None


def _normalize_cause_name(cause: RootCause | str) -> str:
    return cause.name if isinstance(cause, RootCause) else str(cause)


def evaluate_root_cause_predictions(
    reports: tuple[PlanDiagnosticReport, ...],
    labels: tuple[RootCauseGroundTruth, ...],
) -> RootCauseEvaluation:
    by_plan = {report.plan_id: report for report in reports}
    matched = 0
    missing = 0
    unknown = 0
    primary_correct = 0
    true_positive = 0
    predicted_total = 0
    expected_total = 0
    for label in labels:
        report = by_plan.get(label.plan_id)
        if report is None:
            missing += 1
            expected_total += len(set(label.root_causes))
            continue
        matched += 1
        expected_primary = _normalize_cause_name(label.primary_cause)
        predicted_primary = report.primary_cause.name
        if predicted_primary == expected_primary:
            primary_correct += 1
        predicted = set(report.root_causes) - {RootCause.UNKNOWN.name}
        expected = {_normalize_cause_name(cause) for cause in label.root_causes} - {RootCause.UNKNOWN.name}
        if not predicted and RootCause.UNKNOWN.name in report.root_causes:
            unknown += 1
        true_positive += len(predicted & expected)
        predicted_total += len(predicted)
        expected_total += len(expected)
    precision = None if predicted_total == 0 else true_positive / predicted_total
    recall = None if expected_total == 0 else true_positive / expected_total
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return RootCauseEvaluation(
        case_count=len(labels),
        matched_case_count=matched,
        missing_prediction_count=missing,
        unknown_prediction_count=unknown,
        primary_accuracy=None if matched == 0 else primary_correct / matched,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
    )


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


def _measured_root_causes(
    candidate: EvaluatedPlan,
    measured: MeasuredExecutable | None,
) -> tuple[tuple[RootCause, ...], int]:
    if measured is None:
        return (), 0
    estimated_peak = max(int(candidate.simulation.estimated_peak_bytes), 1)
    peak_residual = int(measured.measured_peak_bytes) - estimated_peak
    noise_threshold = max(1 << 20, int(estimated_peak * 0.01))
    causes: list[RootCause] = []
    workspace_change = 0
    if abs(peak_residual) <= noise_threshold:
        causes.append(RootCause.MEASUREMENT_NOISE)
    elif peak_residual > 0:
        workspace_change = peak_residual
        causes.append(RootCause.WORKSPACE_GROWTH)

    estimated_step_us = max(float(candidate.simulation.estimated_step_us), 1.0)
    measured_step_us = float(measured.measured_step_us)
    if measured_step_us > estimated_step_us * 2.0 and measured_step_us - estimated_step_us > 100.0:
        causes.append(RootCause.COST_MODEL_MISRANK)
    return tuple(dict.fromkeys(causes)), workspace_change


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
    evidence: list[DiagnosticEvidence] = [
        DiagnosticEvidence(
            evidence_id=f"{candidate.plan.plan_id}:normalized_saved_reduction",
            root_cause="EXPECTATION_GAP",
            metric="normalized_saved_reduction_bytes",
            value=expected_saved,
            threshold=0,
            direction="greater_than",
            description="PeakAware storage-normalized after-FW retained reduction versus all-save baseline.",
        ),
        DiagnosticEvidence(
            evidence_id=f"{candidate.plan.plan_id}:overall_peak_reduction",
            root_cause="EXPECTATION_GAP",
            metric="estimated_overall_peak_reduction_bytes",
            value=peak_reduction,
            threshold=expected_saved,
            direction="compare_to_saved_reduction",
            description="Estimated end-to-end peak reduction compared with normalized saved reduction.",
        ),
    ]
    if any(effect.pinning_value_ids for effect in candidate.plan.storage_effects):
        pinned_count = sum(1 for effect in candidate.plan.storage_effects if effect.pinning_value_ids)
        evidence.append(
            DiagnosticEvidence(
                evidence_id=f"{candidate.plan.plan_id}:alias_pinning",
                root_cause=RootCause.ALIAS_OR_VIEW_PINNING.name,
                metric="pinned_storage_effect_count",
                value=pinned_count,
                threshold=0,
                direction="greater_than",
                description="Storage effects include values whose base storage remains pinned by aliases or outputs.",
            )
        )
    if transient > 0:
        evidence.append(
            DiagnosticEvidence(
                evidence_id=f"{candidate.plan.plan_id}:recompute_transient",
                root_cause=RootCause.REMATERIALIZATION_WAVE.name,
                metric="bw_recompute_transient_change_bytes",
                value=transient,
                threshold=0,
                direction="greater_than",
                description="Candidate increases backward recompute live bytes relative to baseline.",
            )
        )
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
        if feasibility.status != "FEASIBLE":
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:fixed_frontier_status",
                    root_cause=RootCause.FIXED_BACKWARD_FRONTIER.name,
                    metric="feasibility_status",
                    value=feasibility.status,
                    threshold="FEASIBLE",
                    direction="not_equal",
                    description="Fixed frontier analyzer reported limited or impossible activation headroom.",
                )
            )
    if expected_saved > 0 and peak_reduction <= 0:
        if baseline.simulation.peak_snapshot.phase != candidate.simulation.peak_snapshot.phase:
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:peak_phase_migration",
                    root_cause=RootCause.PEAK_PHASE_MIGRATION.name,
                    metric="peak_phase_transition",
                    value=f"{baseline.simulation.peak_snapshot.phase}->{candidate.simulation.peak_snapshot.phase}",
                    threshold="same_phase",
                    direction="not_equal",
                    description="The candidate moves the overall peak to a different phase after reducing saved activations.",
                )
            )
        else:
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:fixed_frontier_no_gain",
                    root_cause=RootCause.FIXED_BACKWARD_FRONTIER.name,
                    metric="estimated_overall_peak_reduction_bytes",
                    value=peak_reduction,
                    threshold=1,
                    direction="less_than",
                    description="Saved activations decrease but estimated end-to-end peak does not improve.",
                )
            )
    primary, secondary = rank_root_causes(
        expected_saved=expected_saved,
        transient=transient,
        peak_reduction=peak_reduction,
        baseline=baseline,
        candidate=candidate,
        feasibility=feasibility,
    )
    measured_causes, compiler_workspace_allocator_change = _measured_root_causes(candidate, measured)
    if measured is not None:
        estimated_peak = max(int(candidate.simulation.estimated_peak_bytes), 1)
        peak_residual = int(measured.measured_peak_bytes) - estimated_peak
        noise_threshold = max(1 << 20, int(estimated_peak * 0.01))
        if abs(peak_residual) <= noise_threshold:
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:measurement_noise",
                    root_cause=RootCause.MEASUREMENT_NOISE.name,
                    metric="measured_minus_estimated_peak_bytes",
                    value=peak_residual,
                    threshold=noise_threshold,
                    direction="absolute_less_or_equal",
                    description="Measured peak differs from prediction within the configured noise tolerance.",
                )
            )
        elif peak_residual > 0:
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:workspace_growth",
                    root_cause=RootCause.WORKSPACE_GROWTH.name,
                    metric="measured_minus_estimated_peak_bytes",
                    value=peak_residual,
                    threshold=noise_threshold,
                    direction="greater_than",
                    description="Runtime peak exceeds the simulated peak beyond the noise tolerance.",
                )
            )
        estimated_step_us = max(float(candidate.simulation.estimated_step_us), 1.0)
        measured_step_us = float(measured.measured_step_us)
        if measured_step_us > estimated_step_us * 2.0 and measured_step_us - estimated_step_us > 100.0:
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=f"{candidate.plan.plan_id}:cost_model_misrank",
                    root_cause=RootCause.COST_MODEL_MISRANK.name,
                    metric="measured_to_estimated_step_ratio",
                    value=measured_step_us / estimated_step_us,
                    threshold=2.0,
                    direction="greater_than",
                    description="Runtime step time is much higher than the simulated recompute cost.",
                )
            )
    all_causes = tuple(dict.fromkeys((primary,) + secondary + measured_causes))
    causes = tuple(cause.name for cause in all_causes)
    return PlanDiagnosticReport(
        plan_id=candidate.plan.plan_id,
        expected_saved_reduction=expected_saved,
        after_fw_retained_reduction=expected_saved,
        bw_recompute_transient_change=transient,
        fixed_frontier_overlap=fixed_overlap,
        compiler_workspace_allocator_change=compiler_workspace_allocator_change,
        actual_overall_peak_reduction=peak_reduction,
        primary_cause=primary,
        secondary_causes=all_causes[1:],
        root_causes=causes,
        evidence=tuple(evidence),
        repair_hints=tuple(hints),
        counterfactuals=build_counterfactual_ladder(
            baseline,
            candidate,
            feasibility=feasibility,
            measured=measured,
        ),
        strategy_expectation_status="unavailable",
        strategy_expectation_provenance={
            "source": "none",
            "reason": "strategy did not provide a public saved-bytes expectation model",
        },
        strategy_expected_saved_reduction=None,
        normalized_saved_reduction=expected_saved,
        strategy_estimation_gap=None,
        realization_gap=expected_saved - peak_reduction,
        total_expectation_gap=None,
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
