from __future__ import annotations

from peakaware.contracts import JointTrainingIR, RecomputePlan


def last_storage_consumer(ir: JointTrainingIR) -> dict[int, int]:
    last: dict[int, int] = {}
    for value in ir.values:
        if value.storage_id is None:
            continue
        if value.consumer_ids:
            last[value.storage_id] = max(last.get(value.storage_id, -1), max(value.consumer_ids))
    return last


def pin_graph_outputs(ir: JointTrainingIR) -> frozenset[int]:
    output_value_ids = {
        input_id
        for op in ir.ops
        if op.phase == "output"
        for input_id in op.input_value_ids
    }
    return frozenset(
        value.storage_id
        for value in ir.values
        if value.id in output_value_ids and value.storage_id is not None
    )


def build_baseline_events(ir: JointTrainingIR) -> tuple[tuple[str, int, int], ...]:
    events: list[tuple[str, int, int]] = []
    for op in ir.ops:
        for value_id in op.output_value_ids:
            value = ir.values[value_id]
            if value.storage_id is not None:
                events.append(("alloc", op.id, value.storage_id))
    return tuple(events)


def build_plan_events(ir: JointTrainingIR, plan: RecomputePlan) -> tuple[tuple[str, int, int], ...]:
    dropped = {
        effect.storage_id
        for effect in plan.storage_effects
        if effect.decision == "DROP"
    }
    events: list[tuple[str, int, int]] = []
    for op in ir.ops:
        for value_id in op.output_value_ids:
            value = ir.values[value_id]
            if value.storage_id is None:
                continue
            event = "recompute" if value.storage_id in dropped and op.phase == "fw" else "alloc"
            events.append((event, op.id, value.storage_id))
    return tuple(events)
