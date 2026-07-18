from __future__ import annotations

from dataclasses import dataclass

from peakaware.contracts import JointTrainingIR, RepairHint, StorageEffect
from peakaware.cost.base import CostProvider, StaticCostProvider, signature_for_op


@dataclass(frozen=True)
class SaveCandidate:
    storage_id: int
    value_ids: frozenset[int]
    bytes_at_peak: int
    estimated_recompute_us: float
    score: float
    reason: str
    confidence: float


def group_by_storage(ir: JointTrainingIR) -> dict[int, frozenset[int]]:
    return {
        storage.id: frozenset(storage.value_ids)
        for storage in ir.storages
        if not storage.is_external
    }


def compute_storage_effect(ir: JointTrainingIR, storage_id: int, decision: str) -> StorageEffect:
    storage = next(s for s in ir.storages if s.id == storage_id)
    mandatory = {v.id for v in ir.values if v.mandatory_save_reason}
    pinning = tuple(v for v in storage.value_ids if v in mandatory and decision == "DROP")
    return StorageEffect(
        storage_id=storage.id,
        decision=decision,
        decision_value_ids=tuple(storage.value_ids),
        alias_value_ids=tuple(storage.value_ids),
        released_at_peak_bytes=0 if decision == "SAVE" or pinning else storage.physical_nbytes,
        retained_after_fw_bytes=storage.physical_nbytes if decision == "SAVE" else 0,
        pinning_value_ids=pinning,
        confidence=0.9 if not pinning else 0.4,
    )


def reject_alias_pinned_gain(effect: StorageEffect) -> bool:
    return effect.decision == "DROP" and effect.released_at_peak_bytes <= 0


def _cost_for_storage(ir: JointTrainingIR, storage_value_ids: frozenset[int], provider: CostProvider) -> tuple[float, float, str]:
    producer_ids = {
        value.producer_id
        for value in ir.values
        if value.id in storage_value_ids and value.producer_id is not None
    }
    total_us = 0.0
    confidence = 1.0
    sources: list[str] = []
    for op in ir.ops:
        if op.id not in producer_ids:
            continue
        signature = signature_for_op(ir, op)
        cost = provider.estimate(signature) if provider.supports(signature) else None
        if cost is None:
            total_us += 10.0
            confidence = min(confidence, 0.2)
            sources.append("unknown")
        else:
            total_us += cost.estimated_us
            confidence = min(confidence, cost.confidence)
            sources.append(cost.source)
    return max(total_us, 1.0), confidence, ",".join(sorted(set(sources))) or "static"


def select_save_candidates(
    ir: JointTrainingIR,
    *,
    cost_provider: CostProvider | None = None,
    hints: tuple[RepairHint, ...] = (),
    min_bytes: int = 1,
) -> tuple[SaveCandidate, ...]:
    provider = cost_provider or StaticCostProvider()
    hinted_storage_ids = {target for hint in hints for target in hint.target_ids}
    candidates: list[SaveCandidate] = []
    storage_values = group_by_storage(ir)
    storage_by_id = {storage.id: storage for storage in ir.storages}
    for storage_id, value_ids in storage_values.items():
        storage = storage_by_id[storage_id]
        if storage.physical_nbytes < min_bytes:
            continue
        values = [value for value in ir.values if value.id in value_ids]
        if any(value.mandatory_save_reason for value in values):
            continue
        if not any(value.recomputable for value in values):
            continue
        recompute_us, confidence, source = _cost_for_storage(ir, value_ids, provider)
        priority_boost = 2.0 if storage_id in hinted_storage_ids else 1.0
        score = priority_boost * storage.physical_nbytes / recompute_us
        candidates.append(
            SaveCandidate(
                storage_id=storage_id,
                value_ids=value_ids,
                bytes_at_peak=storage.physical_nbytes,
                estimated_recompute_us=recompute_us,
                score=score,
                reason=f"bytes_per_cost:{source}",
                confidence=confidence,
            )
        )
    candidates.sort(key=lambda c: (-c.score, -c.bytes_at_peak, c.storage_id))
    return tuple(candidates)


def build_region_candidates(ir: JointTrainingIR) -> tuple[SaveCandidate, ...]:
    return select_save_candidates(ir)


def expand_peak_region(ir: JointTrainingIR, peak_storage_ids: frozenset[int]) -> tuple[int, ...]:
    return tuple(sorted(storage_id for storage_id in peak_storage_ids if any(s.id == storage_id for s in ir.storages)))
