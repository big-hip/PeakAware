from __future__ import annotations

from dataclasses import dataclass

from peakaware.contracts import JointTrainingIR


@dataclass(frozen=True)
class RecomputeClosure:
    recomputed_op_ids: frozenset[int]
    recomputed_value_ids: frozenset[int]
    barrier_value_ids: frozenset[int]


def derive_recompute_closure(ir: JointTrainingIR, saved_value_ids: frozenset[int]) -> RecomputeClosure:
    mandatory = frozenset(v.id for v in ir.values if v.mandatory_save_reason)
    saved = saved_value_ids | mandatory
    recomputed_values = frozenset(
        v.id
        for v in ir.values
        if v.phase == "fw" and v.producer_id is not None and v.recomputable and v.id not in saved
    )
    recomputed_ops = frozenset(
        v.producer_id
        for v in ir.values
        if v.id in recomputed_values and v.producer_id is not None
    )
    barriers = frozenset(
        v.id
        for v in ir.values
        if v.phase == "fw" and v.producer_id is not None and not v.recomputable and v.id not in saved
    )
    return RecomputeClosure(recomputed_ops, recomputed_values, barriers)


def find_saved_ancestors(ir: JointTrainingIR, value_id: int, saved_value_ids: frozenset[int]) -> frozenset[int]:
    del ir
    return frozenset({value_id} & saved_value_ids)


def deduplicate_shared_ancestors(value_ids: frozenset[int]) -> frozenset[int]:
    return frozenset(value_ids)


def validate_closure(closure: RecomputeClosure) -> tuple[bool, str | None]:
    if closure.barrier_value_ids:
        return False, f"plan drops illegal values: {sorted(closure.barrier_value_ids)}"
    return True, None
