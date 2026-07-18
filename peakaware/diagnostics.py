from __future__ import annotations

from dataclasses import dataclass

from peakaware.contracts import EvaluatedPlan, FeasibilityReport, RepairHint


@dataclass(frozen=True)
class PlanDiagnosticReport:
    plan_id: str
    expected_saved_reduction: int
    after_fw_retained_reduction: int
    bw_recompute_transient_change: int
    fixed_frontier_overlap: int
    compiler_workspace_allocator_change: int
    actual_overall_peak_reduction: int
    root_causes: tuple[str, ...]
    repair_hints: tuple[RepairHint, ...]


def diagnose_plan(
    baseline: EvaluatedPlan,
    candidate: EvaluatedPlan,
    feasibility: FeasibilityReport | None = None,
) -> PlanDiagnosticReport:
    expected_saved = (
        baseline.simulation.after_fw_retained_bytes
        - candidate.simulation.after_fw_retained_bytes
    )
    transient = candidate.simulation.max_recompute_live_bytes - baseline.simulation.max_recompute_live_bytes
    peak_reduction = baseline.simulation.estimated_peak_bytes - candidate.simulation.estimated_peak_bytes
    causes: list[str] = []
    hints: list[RepairHint] = []
    if expected_saved > 0 and peak_reduction <= 0:
        causes.append("peak_migration_or_fixed_frontier")
    if transient > 0:
        causes.append("rematerialization_wave")
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
            causes.append("fixed_frontier")
    return PlanDiagnosticReport(
        plan_id=candidate.plan.plan_id,
        expected_saved_reduction=expected_saved,
        after_fw_retained_reduction=expected_saved,
        bw_recompute_transient_change=transient,
        fixed_frontier_overlap=fixed_overlap,
        compiler_workspace_allocator_change=0,
        actual_overall_peak_reduction=peak_reduction,
        root_causes=tuple(dict.fromkeys(causes)),
        repair_hints=tuple(hints),
    )


def render_diagnostic_text(report: PlanDiagnosticReport) -> str:
    causes = ", ".join(report.root_causes) if report.root_causes else "none"
    return (
        f"{report.plan_id}: peak_reduction={report.actual_overall_peak_reduction} bytes, "
        f"after_fw_reduction={report.after_fw_retained_reduction} bytes, causes={causes}"
    )
