from __future__ import annotations

from peakaware.contracts import (
    FixedTimeline,
    JointTrainingIR,
    PeakSnapshot,
    RecomputePlan,
    SimulationResult,
)


def _storage_bytes(ir: JointTrainingIR) -> dict[int, int]:
    return {storage.id: storage.physical_nbytes for storage in ir.storages}


def _activation_storage_ids(ir: JointTrainingIR) -> frozenset[int]:
    external_ids = {storage.id for storage in ir.storages if storage.is_external}
    return frozenset(storage.id for storage in ir.storages if storage.id not in external_ids)


def _plan_saved_storage_ids(ir: JointTrainingIR, plan: RecomputePlan) -> frozenset[int]:
    value_by_id = {value.id: value for value in ir.values}
    saved = set()
    for value_id in plan.saved_value_ids | plan.mandatory_value_ids:
        value = value_by_id.get(value_id)
        if value is not None and value.storage_id is not None:
            saved.add(value.storage_id)
    return frozenset(saved)


def compute_peak_snapshot(
    phase: str,
    op_id: int | None,
    live_storage_ids: frozenset[int],
    live_activation_bytes: int,
    fixed: FixedTimeline,
    recomputed_bytes: int = 0,
    workspace_bytes: int = 0,
) -> PeakSnapshot:
    return PeakSnapshot(
        phase=phase,
        op_id=op_id,
        live_storage_ids=live_storage_ids,
        live_bytes=fixed.steady_bytes + live_activation_bytes + recomputed_bytes + workspace_bytes,
        parameter_bytes=fixed.parameter_bytes,
        gradient_bytes=fixed.gradient_bytes if phase in {"bw", "optimizer"} else 0,
        optimizer_bytes=fixed.optimizer_state_bytes + (fixed.optimizer_temporary_bytes if phase == "optimizer" else 0),
        saved_activation_bytes=live_activation_bytes,
        recomputed_bytes=recomputed_bytes,
        workspace_bytes=workspace_bytes,
    )


def apply_event(current: int, delta: int) -> int:
    return max(0, current + delta)


def simulate_plan(
    ir: JointTrainingIR,
    plan: RecomputePlan,
    fixed_timeline: FixedTimeline,
) -> SimulationResult:
    bytes_by_storage = _storage_bytes(ir)
    activation_ids = _activation_storage_ids(ir)
    saved_storage_ids = _plan_saved_storage_ids(ir, plan) & activation_ids
    dropped_storage_ids = activation_ids - saved_storage_ids

    saved_bytes = sum(bytes_by_storage[sid] for sid in saved_storage_ids)
    dropped_bytes = sum(bytes_by_storage[sid] for sid in dropped_storage_ids)
    recompute_bytes = dropped_bytes
    fw_peak = fixed_timeline.parameter_bytes + fixed_timeline.buffer_bytes + saved_bytes + dropped_bytes
    after_fw = fixed_timeline.parameter_bytes + fixed_timeline.buffer_bytes + saved_bytes
    bw_peak = fixed_timeline.steady_bytes + saved_bytes + recompute_bytes
    optimizer_peak = fixed_timeline.peak_lower_bound_bytes + saved_bytes
    overall = max(fw_peak, bw_peak, optimizer_peak)
    if overall == bw_peak:
        snapshot = compute_peak_snapshot(
            "bw",
            None,
            frozenset(saved_storage_ids | dropped_storage_ids),
            saved_bytes,
            fixed_timeline,
            recomputed_bytes=recompute_bytes,
        )
    elif overall == optimizer_peak:
        snapshot = compute_peak_snapshot("optimizer", None, frozenset(saved_storage_ids), saved_bytes, fixed_timeline)
    else:
        snapshot = compute_peak_snapshot("fw", None, frozenset(activation_ids), saved_bytes + dropped_bytes, fixed_timeline)

    recompute_span_ops = len([op for op in ir.ops if op.phase == "fw" and op.output_value_ids]) if dropped_storage_ids else 0
    estimated_step_us = float(len(ir.ops) * 10 + len(dropped_storage_ids) * 5)
    return SimulationResult(
        plan_id=plan.plan_id,
        estimated_peak_bytes=overall,
        estimated_step_us=estimated_step_us,
        peak_snapshot=snapshot,
        after_fw_retained_bytes=after_fw,
        fw_peak_bytes=fw_peak,
        bw_peak_bytes=bw_peak,
        optimizer_peak_bytes=optimizer_peak,
        max_recompute_live_bytes=recompute_bytes,
        recompute_span_ops=recompute_span_ops,
        recompute_before_first_bw_op_bytes=recompute_bytes,
        risk_score=0.2 if dropped_storage_ids else 0.05,
        confidence=0.65 if dropped_storage_ids else 0.9,
    )


def compare_plan_delta(baseline: SimulationResult, candidate: SimulationResult) -> dict[str, int]:
    return {
        "estimated_peak_delta_bytes": candidate.estimated_peak_bytes - baseline.estimated_peak_bytes,
        "after_fw_retained_delta_bytes": candidate.after_fw_retained_bytes - baseline.after_fw_retained_bytes,
        "bw_peak_delta_bytes": candidate.bw_peak_bytes - baseline.bw_peak_bytes,
    }
