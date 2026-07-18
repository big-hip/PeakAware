from __future__ import annotations

import hashlib
from dataclasses import dataclass

from peakaware.contracts import JointTrainingIR, RecomputePlan, StorageEffect
from peakaware.errors import PlanValidationError


@dataclass(frozen=True)
class SearchMove:
    value_ids: frozenset[int]
    decision: str
    reason: str


@dataclass(frozen=True)
class SearchResult:
    evaluated_plans: tuple[object, ...]
    selected_plan_id: str
    early_stop_reason: str | None


def _plan_id(graph_key: str, saved_value_ids: frozenset[int], budget: int) -> str:
    h = hashlib.sha256()
    h.update(graph_key.encode("utf-8"))
    h.update(str(sorted(saved_value_ids)).encode("utf-8"))
    h.update(str(budget).encode("utf-8"))
    return h.hexdigest()[:12]


def _storage_effects(ir: JointTrainingIR, saved_value_ids: frozenset[int]) -> tuple[StorageEffect, ...]:
    saved_with_mandatory = saved_value_ids | frozenset(v.id for v in ir.values if v.mandatory_save_reason)
    value_by_id = {v.id: v for v in ir.values}
    effects: list[StorageEffect] = []
    for storage in ir.storages:
        if storage.is_external:
            continue
        saved_aliases = tuple(v for v in storage.value_ids if v in saved_with_mandatory)
        decision = "SAVE" if saved_aliases else "DROP"
        pinning = tuple(
            v
            for v in storage.value_ids
            if value_by_id[v].mandatory_save_reason and v not in saved_aliases
        )
        effects.append(
            StorageEffect(
                storage_id=storage.id,
                decision=decision,
                decision_value_ids=saved_aliases or tuple(storage.value_ids[:1]),
                alias_value_ids=tuple(storage.value_ids),
                released_at_peak_bytes=0 if decision == "SAVE" or pinning else storage.physical_nbytes,
                retained_after_fw_bytes=storage.physical_nbytes if decision == "SAVE" else 0,
                pinning_value_ids=pinning,
                confidence=0.9 if not pinning else 0.5,
            )
        )
    return tuple(effects)


def validate_plan_identity(ir: JointTrainingIR, saved_value_ids: frozenset[int]) -> None:
    valid = {v.id for v in ir.values}
    unknown = saved_value_ids - valid
    if unknown:
        raise PlanValidationError(f"saved values are not in IR: {sorted(unknown)}")


def build_recompute_plan(
    ir: JointTrainingIR,
    *,
    budget_bytes: int,
    saved_value_ids: frozenset[int],
    safety_margin_bytes: int = 0,
    label: str | None = None,
) -> RecomputePlan:
    validate_plan_identity(ir, saved_value_ids)
    mandatory = frozenset(v.id for v in ir.values if v.mandatory_save_reason)
    plan_id = label or _plan_id(ir.graph_key, saved_value_ids | mandatory, budget_bytes)
    return RecomputePlan(
        graph_key=ir.graph_key,
        budget_bytes=budget_bytes,
        storage_effects=_storage_effects(ir, saved_value_ids),
        saved_value_ids=frozenset(saved_value_ids),
        mandatory_value_ids=mandatory,
        estimated_peak_bytes=0,
        estimated_step_us=0.0,
        max_recompute_live_bytes=0,
        recompute_span_ops=0,
        recompute_before_first_bw_op_bytes=0,
        risk_score=0.0,
        confidence=0.0,
        safety_margin_bytes=safety_margin_bytes,
        cost_sources=("m0_static",),
        plan_id=plan_id,
    )
