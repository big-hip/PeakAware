from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from peakaware.contracts import (
    JointTrainingIR,
    RecomputePlan,
    StorageEffect,
    StorageInfo,
    ValueInfo,
)
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


@dataclass(frozen=True)
class PlanBuildCache:
    ir: JointTrainingIR
    valid_value_ids: frozenset[int]
    mandatory_value_ids: frozenset[int]
    value_by_id: Mapping[int, ValueInfo]
    non_external_storages: tuple[StorageInfo, ...]
    fw_logical_nbytes_by_value_id: Mapping[int, int]
    storage_effect_cache: dict[tuple[int, tuple[int, ...]], StorageEffect]
    plan_identity_cache: dict[tuple[frozenset[int], int], str]
    recompute_plan_cache: dict[
        tuple[int, frozenset[int], int, str | None, str | None],
        RecomputePlan,
    ]

    def validate_for(self, ir: JointTrainingIR) -> None:
        if self.ir is not ir:
            raise ValueError("plan build cache belongs to a different IR instance")


def build_plan_build_cache(ir: JointTrainingIR) -> PlanBuildCache:
    return PlanBuildCache(
        ir=ir,
        valid_value_ids=frozenset(value.id for value in ir.values),
        mandatory_value_ids=frozenset(
            value.id for value in ir.values if value.mandatory_save_reason
        ),
        value_by_id={value.id: value for value in ir.values},
        non_external_storages=tuple(
            storage for storage in ir.storages if not storage.is_external
        ),
        fw_logical_nbytes_by_value_id={
            value.id: value.logical_nbytes
            for value in ir.values
            if value.phase == "fw"
        },
        storage_effect_cache={},
        plan_identity_cache={},
        recompute_plan_cache={},
    )


def plan_identity_key(
    graph_key: str,
    saved_value_ids: frozenset[int],
    budget_bytes: int,
    *,
    plan_cache: PlanBuildCache | None = None,
) -> str:
    cache_key = (saved_value_ids, budget_bytes)
    if plan_cache is not None:
        if plan_cache.ir.graph_key != graph_key:
            raise ValueError("plan build cache belongs to a different graph")
        cached = plan_cache.plan_identity_cache.get(cache_key)
        if cached is not None:
            return cached
    h = hashlib.sha256()
    h.update(graph_key.encode("utf-8"))
    h.update(str(sorted(saved_value_ids)).encode("utf-8"))
    h.update(str(budget_bytes).encode("utf-8"))
    identity = h.hexdigest()[:12]
    if plan_cache is not None:
        plan_cache.plan_identity_cache[cache_key] = identity
    return identity


def _storage_effects(
    ir: JointTrainingIR,
    saved_value_ids: frozenset[int],
    *,
    plan_cache: PlanBuildCache | None = None,
) -> tuple[StorageEffect, ...]:
    if plan_cache is None:
        plan_cache = build_plan_build_cache(ir)
    else:
        plan_cache.validate_for(ir)
    saved_with_mandatory = saved_value_ids | plan_cache.mandatory_value_ids
    value_by_id = plan_cache.value_by_id
    effects: list[StorageEffect] = []
    for storage in plan_cache.non_external_storages:
        saved_aliases = tuple(v for v in storage.value_ids if v in saved_with_mandatory)
        cache_key = (storage.id, saved_aliases)
        cached_effect = plan_cache.storage_effect_cache.get(cache_key)
        if cached_effect is not None:
            effects.append(cached_effect)
            continue
        decision = "SAVE" if saved_aliases else "DROP"
        pinning = tuple(
            v
            for v in storage.value_ids
            if value_by_id[v].mandatory_save_reason and v not in saved_aliases
        )
        effect = StorageEffect(
            storage_id=storage.id,
            decision=decision,
            decision_value_ids=saved_aliases or tuple(storage.value_ids[:1]),
            alias_value_ids=tuple(storage.value_ids),
            released_at_peak_bytes=(
                0 if decision == "SAVE" or pinning else storage.physical_nbytes
            ),
            retained_after_fw_bytes=(
                storage.physical_nbytes if decision == "SAVE" else 0
            ),
            pinning_value_ids=pinning,
            confidence=0.9 if not pinning else 0.5,
        )
        plan_cache.storage_effect_cache[cache_key] = effect
        effects.append(effect)
    return tuple(effects)


def validate_plan_identity(
    ir: JointTrainingIR,
    saved_value_ids: frozenset[int],
    *,
    plan_cache: PlanBuildCache | None = None,
) -> None:
    if plan_cache is None:
        valid = frozenset(value.id for value in ir.values)
    else:
        plan_cache.validate_for(ir)
        valid = plan_cache.valid_value_ids
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
    strategy_expectation_source: str | None = None,
    plan_cache: PlanBuildCache | None = None,
) -> RecomputePlan:
    if plan_cache is None:
        plan_cache = build_plan_build_cache(ir)
    else:
        plan_cache.validate_for(ir)
    validate_plan_identity(ir, saved_value_ids, plan_cache=plan_cache)
    mandatory = plan_cache.mandatory_value_ids
    effective_saved_value_ids = saved_value_ids | mandatory
    cache_key = (
        budget_bytes,
        saved_value_ids,
        safety_margin_bytes,
        label,
        strategy_expectation_source,
    )
    cached_plan = plan_cache.recompute_plan_cache.get(cache_key)
    if cached_plan is not None:
        return cached_plan
    plan_id = label or plan_identity_key(
        ir.graph_key,
        effective_saved_value_ids,
        budget_bytes,
        plan_cache=plan_cache,
    )
    strategy_saved_bytes = None
    strategy_provenance = {}
    if strategy_expectation_source is not None:
        strategy_saved_bytes = sum(
            plan_cache.fw_logical_nbytes_by_value_id.get(value_id, 0)
            for value_id in effective_saved_value_ids
        )
        strategy_provenance = {
            "source": strategy_expectation_source,
            "metric": "logical_fw_saved_value_bytes",
            "saved_value_count": len(effective_saved_value_ids),
            "mandatory_value_count": len(mandatory),
        }
    plan = RecomputePlan(
        graph_key=ir.graph_key,
        budget_bytes=budget_bytes,
        storage_effects=_storage_effects(ir, saved_value_ids, plan_cache=plan_cache),
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
        strategy_saved_bytes=strategy_saved_bytes,
        strategy_expectation_provenance=strategy_provenance,
    )
    plan_cache.recompute_plan_cache[cache_key] = plan
    return plan
