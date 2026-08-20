from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from peakaware.contracts import JointTrainingIR, OpInfo, ValueInfo


@dataclass(frozen=True)
class RecomputeClosure:
    recomputed_op_ids: frozenset[int]
    recomputed_value_ids: frozenset[int]
    barrier_value_ids: frozenset[int]


@dataclass(frozen=True)
class RecomputeGraphCache:
    ir: JointTrainingIR
    value_by_id: Mapping[int, ValueInfo]
    op_by_id: Mapping[int, OpInfo]
    external_storage_ids: frozenset[int]
    mandatory_value_ids: frozenset[int]
    target_value_ids: frozenset[int]

    def validate_for(self, ir: JointTrainingIR) -> None:
        if self.ir is not ir:
            raise ValueError("recompute graph cache belongs to a different IR instance")


def build_recompute_graph_cache(ir: JointTrainingIR) -> RecomputeGraphCache:
    value_by_id = {value.id: value for value in ir.values}
    return RecomputeGraphCache(
        ir=ir,
        value_by_id=value_by_id,
        op_by_id={op.id: op for op in ir.ops},
        external_storage_ids=frozenset(
            storage.id for storage in ir.storages if storage.is_external
        ),
        mandatory_value_ids=frozenset(
            value.id for value in ir.values if value.mandatory_save_reason
        ),
        target_value_ids=frozenset(
            value.id
            for value in ir.values
            if value.phase == "fw"
            and value.crosses_fw_bw
            and value.producer_id is not None
        ),
    )


def derive_recompute_closure(
    ir: JointTrainingIR,
    saved_value_ids: frozenset[int],
    *,
    graph_cache: RecomputeGraphCache | None = None,
) -> RecomputeClosure:
    if graph_cache is None:
        graph_cache = build_recompute_graph_cache(ir)
    else:
        graph_cache.validate_for(ir)
    value_by_id = graph_cache.value_by_id
    op_by_id = graph_cache.op_by_id
    external_storage_ids = graph_cache.external_storage_ids
    mandatory = graph_cache.mandatory_value_ids
    saved = saved_value_ids | mandatory
    saved_storage_ids = {
        value_by_id[value_id].storage_id
        for value_id in saved
        if value_id in value_by_id and value_by_id[value_id].storage_id is not None
    }
    targets = {
        value_id
        for value_id in graph_cache.target_value_ids
        for value in (value_by_id[value_id],)
        if (
            value.storage_id not in saved_storage_ids
            and value.storage_id not in external_storage_ids
        )
    }

    recomputed_ops: set[int] = set()
    recomputed_values: set[int] = set()
    barriers: set[int] = set()
    pending = list(targets)
    visited: set[int] = set()
    while pending:
        value_id = pending.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        value = value_by_id[value_id]
        if value.storage_id in saved_storage_ids or value.producer_id is None:
            continue
        producer = op_by_id.get(value.producer_id)
        if value.phase != "fw" or not value.recomputable or producer is None or not producer.recomputable:
            barriers.add(value_id)
            continue
        recomputed_ops.add(producer.id)
        recomputed_values.update(
            output_id
            for output_id in producer.output_value_ids
            if output_id in value_by_id and value_by_id[output_id].phase == "fw"
        )
        for input_id in producer.input_value_ids:
            input_value = value_by_id.get(input_id)
            if input_value is None or input_value.producer_id is None:
                continue
            if input_value.storage_id in saved_storage_ids:
                continue
            pending.append(input_id)
    return RecomputeClosure(
        frozenset(recomputed_ops),
        frozenset(recomputed_values),
        frozenset(barriers),
    )


def find_saved_ancestors(
    ir: JointTrainingIR,
    value_id: int,
    saved_value_ids: frozenset[int],
) -> frozenset[int]:
    value_by_id = {value.id: value for value in ir.values}
    op_by_id = {op.id: op for op in ir.ops}
    saved_storage_ids = {
        value_by_id[saved_id].storage_id
        for saved_id in saved_value_ids
        if saved_id in value_by_id and value_by_id[saved_id].storage_id is not None
    }
    ancestors: set[int] = set()
    pending = [value_id]
    visited: set[int] = set()
    while pending:
        current_id = pending.pop()
        if current_id in visited or current_id not in value_by_id:
            continue
        visited.add(current_id)
        value = value_by_id[current_id]
        if current_id in saved_value_ids or value.storage_id in saved_storage_ids:
            ancestors.add(current_id)
            continue
        if value.producer_id is None or value.producer_id not in op_by_id:
            continue
        pending.extend(op_by_id[value.producer_id].input_value_ids)
    return frozenset(ancestors)


def deduplicate_shared_ancestors(value_ids: frozenset[int]) -> frozenset[int]:
    return frozenset(value_ids)


def validate_closure(closure: RecomputeClosure) -> tuple[bool, str | None]:
    if closure.barrier_value_ids:
        return False, f"plan drops illegal values: {sorted(closure.barrier_value_ids)}"
    return True, None
