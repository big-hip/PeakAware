from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from peakaware.contracts import (
    FixedTimeline,
    JointTrainingIR,
    PeakSnapshot,
    RecomputePlan,
    SimulationResult,
)

from peakaware.cost.base import CostProvider, OpSignature, signature_for_op
from peakaware.recompute import (
    RecomputeClosure,
    RecomputeGraphCache,
    build_recompute_graph_cache,
    derive_recompute_closure,
)


ProviderCost = tuple[float, int, float, str]
FullGraphCost = tuple[float, float, tuple[str, ...], dict[str, Any]]
OptimizerCost = tuple[float, float, str]


@dataclass(frozen=True)
class SimulationCostCache:
    ir: JointTrainingIR
    fixed_timeline: FixedTimeline
    cost_provider: CostProvider | None
    op_costs: Mapping[int, ProviderCost]
    full_graph_step_cost: FullGraphCost
    optimizer_step_cost: OptimizerCost
    recompute_graph_cache: RecomputeGraphCache
    bytes_by_storage: Mapping[int, int]
    activation_storage_ids: frozenset[int]
    value_storage_by_id: Mapping[int, int | None]
    fw_ops: tuple[Any, ...]
    fw_produced_storage_ids_by_op: Mapping[int, frozenset[int]]
    fw_storage_last_use: Mapping[int, int]

    def validate_for(
        self,
        ir: JointTrainingIR,
        fixed_timeline: FixedTimeline,
        cost_provider: CostProvider | None,
    ) -> None:
        if self.ir is not ir:
            raise ValueError("simulation cost cache belongs to a different IR instance")
        if self.fixed_timeline is not fixed_timeline:
            raise ValueError("simulation cost cache belongs to a different fixed timeline")
        if self.cost_provider is not cost_provider:
            raise ValueError("simulation cost cache belongs to a different cost provider")


def _storage_bytes(ir: JointTrainingIR) -> dict[int, int]:
    return {storage.id: storage.physical_nbytes for storage in ir.storages}


def _activation_storage_ids(ir: JointTrainingIR) -> frozenset[int]:
    external_ids = {storage.id for storage in ir.storages if storage.is_external}
    crossing_ids = {
        value.storage_id
        for value in ir.values
        if value.phase == "fw" and value.crosses_fw_bw and value.storage_id is not None
    }
    if crossing_ids:
        return frozenset(crossing_ids - external_ids)
    return frozenset(storage.id for storage in ir.storages if storage.id not in external_ids)


def _forward_liveness_peak(
    ir: JointTrainingIR,
    saved_storage_ids: frozenset[int],
    dropped_storage_ids: frozenset[int],
    bytes_by_storage: dict[int, int],
    *,
    cached_fw_ops: tuple[Any, ...] | None = None,
    cached_produced_storage_ids_by_op: Mapping[int, frozenset[int]] | None = None,
    cached_storage_last_use: Mapping[int, int] | None = None,
    disable_lifetime: bool = False,
) -> int:
    fw_ops = cached_fw_ops or tuple(op for op in ir.ops if op.phase == "fw")
    if not fw_ops:
        return sum(bytes_by_storage[sid] for sid in saved_storage_ids | dropped_storage_ids)

    activation_storage_ids = saved_storage_ids | dropped_storage_ids
    if cached_storage_last_use is None:
        op_position = {op.id: index for index, op in enumerate(fw_ops)}
        storage_last_fw_use: Mapping[int, int] = {}
        mutable_last_use: dict[int, int] = {}
        for value in ir.values:
            if value.storage_id not in activation_storage_ids:
                continue
            consumers = [
                op_position[consumer]
                for consumer in value.consumer_ids
                if consumer in op_position
            ]
            if consumers:
                mutable_last_use[value.storage_id] = max(
                    mutable_last_use.get(value.storage_id, -1),
                    max(consumers),
                )
        storage_last_fw_use = mutable_last_use
    else:
        storage_last_fw_use = cached_storage_last_use

    value_by_id = None
    if cached_produced_storage_ids_by_op is None:
        value_by_id = {value.id: value for value in ir.values}

    live_storage_ids: set[int] = set()
    live_bytes = 0
    peak_bytes = 0
    for index, op in enumerate(fw_ops):
        produced_storage_ids = (
            cached_produced_storage_ids_by_op.get(op.id, frozenset())
            if cached_produced_storage_ids_by_op is not None
            else {
                value_by_id[value_id].storage_id
                for value_id in op.output_value_ids
                if (
                    value_id in value_by_id
                    and value_by_id[value_id].storage_id is not None
                    and value_by_id[value_id].storage_id in activation_storage_ids
                )
            }
        )
        for storage_id in sorted(produced_storage_ids):
            if storage_id not in live_storage_ids:
                live_storage_ids.add(storage_id)
                live_bytes += bytes_by_storage[storage_id]
        peak_bytes = max(peak_bytes, live_bytes)

        if disable_lifetime:
            # Lifetime accounting OFF: no last-use release; every produced
            # activation stays alive until the end of the forward phase.
            continue
        releasable = [
            storage_id
            for storage_id in live_storage_ids
            if storage_id not in saved_storage_ids and storage_last_fw_use.get(storage_id, index) <= index
        ]
        for storage_id in releasable:
            live_storage_ids.remove(storage_id)
            live_bytes = apply_event(live_bytes, -bytes_by_storage[storage_id])
    return peak_bytes


def _simulated_memory_event_trace(
    ir: JointTrainingIR,
    saved_storage_ids: frozenset[int],
    dropped_storage_ids: frozenset[int],
    bytes_by_storage: dict[int, int],
    fixed_timeline: FixedTimeline,
    *,
    cost_provider: CostProvider | None,
    cached_op_costs: Mapping[int, ProviderCost] | None,
    optimizer_step_us: float,
    recompute_peak_bytes: int,
    recompute_workspace_bytes: int,
    recompute_step_us: float,
    expected_bw_peak_bytes: int,
) -> tuple[dict[str, Any], ...]:
    value_by_id = {value.id: value for value in ir.values}
    activation_storage_ids = saved_storage_ids | dropped_storage_ids
    rows: list[dict[str, Any]] = []
    time_us = 0.0

    def append(
        phase: str,
        event: str,
        memory_bytes: int,
        *,
        fixed_bytes: int = 0,
        payload_bytes: int = 0,
        optimizer_bytes: int = 0,
        gradient_bytes: int = 0,
        parameter_bytes: int = 0,
        buffer_bytes: int = 0,
        runtime_replica_bytes: int = 0,
        workspace_bytes: int = 0,
        live_storage_count: int = 0,
        **extra: Any,
    ) -> None:
        rows.append(
            {
                "phase": phase,
                "event": event,
                "time_us": time_us,
                "bytes": max(0, int(memory_bytes)),
                "fixed_bytes": max(0, int(fixed_bytes)),
                "payload_bytes": max(0, int(payload_bytes)),
                "optimizer_bytes": max(0, int(optimizer_bytes)),
                "gradient_bytes": max(0, int(gradient_bytes)),
                "parameter_bytes": max(0, int(parameter_bytes)),
                "buffer_bytes": max(0, int(buffer_bytes)),
                "runtime_replica_bytes": max(0, int(runtime_replica_bytes)),
                "workspace_bytes": max(0, int(workspace_bytes)),
                "live_storage_count": max(0, int(live_storage_count)),
                **extra,
            }
        )

    append("start", "step_start", 0)

    fw_ops = tuple(op for op in ir.ops if op.phase == "fw")
    fw_position = {op.id: index for index, op in enumerate(fw_ops)}
    storage_last_fw_use: dict[int, int] = {}
    for value in ir.values:
        if value.storage_id not in activation_storage_ids:
            continue
        consumers = [fw_position[consumer] for consumer in value.consumer_ids if consumer in fw_position]
        if consumers:
            storage_last_fw_use[value.storage_id] = max(
                storage_last_fw_use.get(value.storage_id, -1),
                max(consumers),
            )

    live_storage_ids: set[int] = set()
    live_activation_bytes = 0
    for index, op in enumerate(fw_ops):
        op_us, workspace_bytes, _confidence, source = _provider_cost(
            cost_provider,
            ir,
            op.id,
            cached_op_costs=cached_op_costs,
        )
        time_us += op_us
        produced_storage_ids = {
            value_by_id[value_id].storage_id
            for value_id in op.output_value_ids
            if (
                value_id in value_by_id
                and value_by_id[value_id].storage_id is not None
                and value_by_id[value_id].storage_id in activation_storage_ids
            )
        }
        for storage_id in sorted(produced_storage_ids):
            if storage_id not in live_storage_ids:
                live_storage_ids.add(storage_id)
                live_activation_bytes += bytes_by_storage[storage_id]
        append(
            "fw",
            "op_end",
            fixed_timeline.forward_resident_bytes + live_activation_bytes + workspace_bytes,
            fixed_bytes=fixed_timeline.forward_resident_bytes,
            payload_bytes=live_activation_bytes,
            optimizer_bytes=fixed_timeline.optimizer_state_bytes,
            parameter_bytes=fixed_timeline.parameter_bytes,
            buffer_bytes=fixed_timeline.buffer_bytes,
            runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
            workspace_bytes=workspace_bytes,
            live_storage_count=len(live_storage_ids),
            op_id=op.id,
            op_name=op.name,
            source=source,
        )
        releasable = [
            storage_id
            for storage_id in live_storage_ids
            if storage_id not in saved_storage_ids and storage_last_fw_use.get(storage_id, index) <= index
        ]
        for storage_id in releasable:
            live_storage_ids.remove(storage_id)
            live_activation_bytes = apply_event(live_activation_bytes, -bytes_by_storage[storage_id])

    append(
        "after_fw",
        "phase_boundary",
        fixed_timeline.forward_resident_bytes
        + sum(bytes_by_storage[sid] for sid in saved_storage_ids),
        fixed_bytes=fixed_timeline.forward_resident_bytes,
        payload_bytes=sum(bytes_by_storage[sid] for sid in saved_storage_ids),
        optimizer_bytes=fixed_timeline.optimizer_state_bytes,
        parameter_bytes=fixed_timeline.parameter_bytes,
        buffer_bytes=fixed_timeline.buffer_bytes,
        runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
        live_storage_count=len(saved_storage_ids),
    )

    bw_ops = tuple(op for op in ir.ops if op.phase == "bw")
    bw_position = {op.id: index for index, op in enumerate(bw_ops)}
    storage_last_bw_use: dict[int, int] = {}
    for value in ir.values:
        if value.storage_id not in saved_storage_ids:
            continue
        consumers = [bw_position[consumer] for consumer in value.consumer_ids if consumer in bw_position]
        if consumers:
            storage_last_bw_use[value.storage_id] = max(
                storage_last_bw_use.get(value.storage_id, -1),
                max(consumers),
            )
    live_saved_storage_ids = set(saved_storage_ids)
    live_saved_bytes = sum(bytes_by_storage[sid] for sid in live_saved_storage_ids)
    time_us += max(float(recompute_step_us), 0.0)
    append(
        "bw",
        "phase_peak_bound",
        expected_bw_peak_bytes,
        fixed_bytes=fixed_timeline.resident_bytes,
        payload_bytes=live_saved_bytes + recompute_peak_bytes,
        optimizer_bytes=fixed_timeline.optimizer_state_bytes,
        gradient_bytes=fixed_timeline.gradient_bytes,
        parameter_bytes=fixed_timeline.parameter_bytes,
        buffer_bytes=fixed_timeline.buffer_bytes,
        runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
        workspace_bytes=recompute_workspace_bytes,
        live_storage_count=len(live_saved_storage_ids),
        recomputed_bytes=recompute_peak_bytes,
        bound_kind="max_saved_plus_phase_workspace_and_recompute",
    )
    for index, op in enumerate(bw_ops):
        op_us, workspace_bytes, _confidence, source = _provider_cost(
            cost_provider,
            ir,
            op.id,
            cached_op_costs=cached_op_costs,
        )
        time_us += op_us
        append(
            "bw",
            "op_end",
            fixed_timeline.resident_bytes + live_saved_bytes + workspace_bytes,
            fixed_bytes=fixed_timeline.resident_bytes,
            payload_bytes=live_saved_bytes,
            optimizer_bytes=fixed_timeline.optimizer_state_bytes,
            gradient_bytes=fixed_timeline.gradient_bytes,
            parameter_bytes=fixed_timeline.parameter_bytes,
            buffer_bytes=fixed_timeline.buffer_bytes,
            runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
            workspace_bytes=workspace_bytes,
            live_storage_count=len(live_saved_storage_ids),
            op_id=op.id,
            op_name=op.name,
            source=source,
        )
        releasable = [
            storage_id
            for storage_id in live_saved_storage_ids
            if storage_id in storage_last_bw_use and storage_last_bw_use[storage_id] <= index
        ]
        for storage_id in releasable:
            live_saved_storage_ids.remove(storage_id)
            live_saved_bytes = apply_event(live_saved_bytes, -bytes_by_storage[storage_id])
    append(
        "optimizer",
        "phase_start",
        fixed_timeline.resident_bytes,
        fixed_bytes=fixed_timeline.resident_bytes,
        optimizer_bytes=fixed_timeline.optimizer_state_bytes,
        gradient_bytes=fixed_timeline.gradient_bytes,
        parameter_bytes=fixed_timeline.parameter_bytes,
        buffer_bytes=fixed_timeline.buffer_bytes,
        runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
    )
    time_us += max(float(optimizer_step_us), 0.0)
    append(
        "optimizer",
        "optimizer_peak",
        fixed_timeline.peak_lower_bound_bytes,
        fixed_bytes=fixed_timeline.resident_bytes,
        optimizer_bytes=fixed_timeline.optimizer_state_bytes + fixed_timeline.optimizer_temporary_bytes,
        gradient_bytes=fixed_timeline.gradient_bytes,
        parameter_bytes=fixed_timeline.parameter_bytes,
        buffer_bytes=fixed_timeline.buffer_bytes,
        runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
        workspace_bytes=fixed_timeline.mandatory_workspace_bytes,
    )
    append(
        "overall",
        "step_end",
        fixed_timeline.peak_lower_bound_bytes,
        fixed_bytes=fixed_timeline.resident_bytes,
        optimizer_bytes=fixed_timeline.optimizer_state_bytes + fixed_timeline.optimizer_temporary_bytes,
        gradient_bytes=fixed_timeline.gradient_bytes,
        parameter_bytes=fixed_timeline.parameter_bytes,
        buffer_bytes=fixed_timeline.buffer_bytes,
        runtime_replica_bytes=fixed_timeline.runtime_replica_bytes,
        workspace_bytes=fixed_timeline.mandatory_workspace_bytes,
    )
    return tuple(rows)


def _plan_saved_storage_ids(ir: JointTrainingIR, plan: RecomputePlan) -> frozenset[int]:
    value_by_id = {value.id: value for value in ir.values}
    saved = set()
    for value_id in plan.saved_value_ids | plan.mandatory_value_ids:
        value = value_by_id.get(value_id)
        if value is not None and value.storage_id is not None:
            saved.add(value.storage_id)
    return frozenset(saved)


def _plan_saved_storage_ids_cached(
    plan: RecomputePlan,
    value_storage_by_id: Mapping[int, int | None],
) -> frozenset[int]:
    return frozenset(
        storage_id
        for value_id in plan.saved_value_ids | plan.mandatory_value_ids
        if (storage_id := value_storage_by_id.get(value_id)) is not None
    )


def compute_peak_snapshot(
    phase: str,
    op_id: int | None,
    live_storage_ids: frozenset[int],
    live_activation_bytes: int,
    fixed: FixedTimeline,
    recomputed_bytes: int = 0,
    workspace_bytes: int = 0,
) -> PeakSnapshot:
    fixed_bytes = fixed.forward_resident_bytes if phase == "fw" else fixed.resident_bytes
    return PeakSnapshot(
        phase=phase,
        op_id=op_id,
        live_storage_ids=live_storage_ids,
        live_bytes=fixed_bytes + live_activation_bytes + recomputed_bytes + workspace_bytes,
        parameter_bytes=fixed.parameter_bytes,
        gradient_bytes=fixed.gradient_bytes if phase in {"bw", "optimizer"} else 0,
        optimizer_bytes=fixed.optimizer_state_bytes + (fixed.optimizer_temporary_bytes if phase == "optimizer" else 0),
        saved_activation_bytes=live_activation_bytes,
        recomputed_bytes=recomputed_bytes,
        workspace_bytes=workspace_bytes,
        runtime_replica_bytes=fixed.runtime_replica_bytes,
    )


def apply_event(current: int, delta: int) -> int:
    return max(0, current + delta)


def _provider_cost(
    provider: CostProvider | None,
    ir: JointTrainingIR,
    op_id: int,
    *,
    cached_op_costs: Mapping[int, ProviderCost] | None = None,
) -> ProviderCost:
    if cached_op_costs is not None and op_id in cached_op_costs:
        return cached_op_costs[op_id]
    if provider is None:
        return 10.0, 0, 0.65, "m0_static"
    op = next((item for item in ir.ops if item.id == op_id), None)
    if op is None:
        return 10.0, 0, 0.2, "missing_op"
    signature = signature_for_op(ir, op)
    cost = provider.estimate(signature) if provider.supports(signature) else None
    if cost is None:
        return 10.0, 0, 0.2, "unknown"
    return (
        max(float(cost.estimated_us), 0.0),
        max(int(cost.memory_bytes), 0),
        max(0.0, min(float(cost.confidence), 1.0)),
        str(cost.source),
    )


def _is_real_cost_source(source: str) -> bool:
    return (
        source.startswith("legacy_adapter:atencost_analytical")
        or source.startswith("profile_db")
    )


def _full_graph_step_cost(
    ir: JointTrainingIR,
    provider: CostProvider | None,
    *,
    cached_op_costs: Mapping[int, ProviderCost] | None = None,
) -> FullGraphCost:
    if provider is None:
        return float(len(ir.ops) * 10), 0.65, ("m0_static",), {
            "op_count": len(ir.ops),
            "real_cost_op_count": 0,
            "source_counts": {"m0_static": len(ir.ops)},
            "fallback_ops": (),
        }
    total_us = 0.0
    confidence = 1.0
    sources: set[str] = set()
    source_counts: dict[str, int] = {}
    fallback_ops: list[dict[str, Any]] = []
    workspace_peak_by_phase = {"fw": 0, "bw": 0, "optimizer": 0}
    op_count = 0
    real_cost_op_count = 0
    used_real_cost = False
    for op in ir.ops:
        if op.phase in {"input", "output"}:
            continue
        op_count += 1
        op_us, workspace_bytes, op_confidence, source = _provider_cost(
            provider,
            ir,
            op.id,
            cached_op_costs=cached_op_costs,
        )
        if op.phase in workspace_peak_by_phase:
            workspace_peak_by_phase[op.phase] = max(workspace_peak_by_phase[op.phase], workspace_bytes)
        sources.add(source)
        source_counts[source] = source_counts.get(source, 0) + 1
        confidence = min(confidence, op_confidence)
        if _is_real_cost_source(source):
            used_real_cost = True
            real_cost_op_count += 1
            total_us += op_us
        else:
            fallback_ops.append(
                {
                    "op_id": op.id,
                    "name": op.name,
                    "target": op.target,
                    "phase": op.phase,
                    "fallback_source": source,
                }
            )
            total_us += 10.0
    breakdown = {
        "op_count": op_count,
        "real_cost_op_count": real_cost_op_count,
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_ops": tuple(fallback_ops[:50]),
        "fallback_op_count": len(fallback_ops),
        "workspace_peak_by_phase_bytes": dict(workspace_peak_by_phase),
    }
    if not used_real_cost:
        return float(len(ir.ops) * 10), 1.0, tuple(sorted(sources)) or ("m0_static",), breakdown
    return total_us, confidence, tuple(sorted(sources)), breakdown


def _optimizer_step_cost(fixed_timeline: FixedTimeline, provider: CostProvider | None) -> tuple[float, float, str]:
    if provider is None or fixed_timeline.parameter_bytes <= 0:
        return 0.0, 0.65, "m0_static"
    signature = OpSignature(
        op_name="optimizer_step",
        target="AdamOptimizerStep",
        input_bytes=fixed_timeline.parameter_bytes,
        output_bytes=0,
        dtype="torch.float32",
    )
    cost = provider.estimate(signature) if provider.supports(signature) else None
    if cost is None or not _is_real_cost_source(str(cost.source)):
        return 0.0, 1.0, "optimizer_static"
    return max(float(cost.estimated_us), 0.0), max(0.0, min(float(cost.confidence), 1.0)), str(cost.source)


def build_simulation_cost_cache(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    cost_provider: CostProvider | None,
) -> SimulationCostCache:
    recompute_graph_cache = build_recompute_graph_cache(ir)
    bytes_by_storage = _storage_bytes(ir)
    activation_storage_ids = _activation_storage_ids(ir)
    fw_ops = tuple(op for op in ir.ops if op.phase == "fw")
    fw_position = {op.id: index for index, op in enumerate(fw_ops)}
    fw_storage_last_use: dict[int, int] = {}
    for value in ir.values:
        if value.storage_id not in activation_storage_ids:
            continue
        consumers = [
            fw_position[consumer]
            for consumer in value.consumer_ids
            if consumer in fw_position
        ]
        if consumers:
            fw_storage_last_use[value.storage_id] = max(
                fw_storage_last_use.get(value.storage_id, -1),
                max(consumers),
            )
    value_by_id = recompute_graph_cache.value_by_id
    fw_produced_storage_ids_by_op = {
        op.id: frozenset(
            value_by_id[value_id].storage_id
            for value_id in op.output_value_ids
            if value_id in value_by_id
            and value_by_id[value_id].storage_id is not None
            and value_by_id[value_id].storage_id in activation_storage_ids
        )
        for op in fw_ops
    }
    op_costs = {
        op.id: _provider_cost(cost_provider, ir, op.id)
        for op in ir.ops
        if op.phase not in {"input", "output"}
    }
    return SimulationCostCache(
        ir=ir,
        fixed_timeline=fixed_timeline,
        cost_provider=cost_provider,
        op_costs=op_costs,
        full_graph_step_cost=_full_graph_step_cost(
            ir,
            cost_provider,
            cached_op_costs=op_costs,
        ),
        optimizer_step_cost=_optimizer_step_cost(fixed_timeline, cost_provider),
        recompute_graph_cache=recompute_graph_cache,
        bytes_by_storage=bytes_by_storage,
        activation_storage_ids=activation_storage_ids,
        value_storage_by_id={
            value.id: value.storage_id for value in ir.values
        },
        fw_ops=fw_ops,
        fw_produced_storage_ids_by_op=fw_produced_storage_ids_by_op,
        fw_storage_last_use=fw_storage_last_use,
    )


def _recompute_liveness_peak(
    ir: JointTrainingIR,
    closure: RecomputeClosure,
    saved_storage_ids: frozenset[int],
    bytes_by_storage: dict[int, int],
    *,
    cost_provider: CostProvider | None,
    cached_op_costs: Mapping[int, ProviderCost] | None,
    cached_value_by_id: Mapping[int, Any] | None = None,
    cached_fw_ops: tuple[Any, ...] | None = None,
    disable_lifetime: bool = False,
    disable_replay: bool = False,
) -> tuple[int, int, int, int, float, float, tuple[str, ...]]:
    """Estimate transient BW recompute memory with a small live-range simulator.

    The previous M0 model treated every dropped activation storage as live at
    the same time during backward.  In practice rematerialization is usually a
    wave: intermediate recompute outputs die once their replay consumers have
    executed, and terminal outputs are consumed by the next backward slice.  The
    event simulation mirrors the live-range accounting style used by the
    top-level toolkit estimator while staying on PeakAware's storage-aware IR.

    Ablation switches: ``disable_replay`` falls back to the M0 all-simultaneous
    model (every dropped storage counted at once); ``disable_lifetime`` keeps
    every recompute temporary alive until the end of the replay.
    """

    value_by_id = cached_value_by_id or {value.id: value for value in ir.values}
    recomputed_values = set(closure.recomputed_value_ids)
    fw_ops = cached_fw_ops or tuple(op for op in ir.ops if op.phase == "fw")
    recompute_ops = tuple(op for op in fw_ops if op.id in closure.recomputed_op_ids)
    if not recompute_ops:
        return 0, 0, 0, 0, 0.0, 0.9, ("m0_static",)

    op_position = {op.id: index for index, op in enumerate(recompute_ops)}
    storage_last_use: dict[int, int] = {}
    terminal_storage_ids: set[int] = set()
    for value_id in recomputed_values:
        value = value_by_id[value_id]
        if value.storage_id is None or value.storage_id in saved_storage_ids:
            continue
        consumers = [op_position[consumer] for consumer in value.consumer_ids if consumer in op_position]
        if consumers:
            storage_last_use[value.storage_id] = max(storage_last_use.get(value.storage_id, -1), max(consumers))
        else:
            terminal_storage_ids.add(value.storage_id)

    if disable_replay:
        # M0 model: every dropped activation storage is simultaneously live
        # during the backward replay, regardless of consumption order.
        peak_payload_bytes = sum(
            bytes_by_storage[sid]
            for sid in (
                {v.storage_id for v in value_by_id.values() if v.id in recomputed_values}
                - saved_storage_ids
            )
            if sid is not None and sid in bytes_by_storage
        )
        workspace_peak_bytes = 0
        total_step_us = 0.0
        confidence = 0.9
        return (
            peak_payload_bytes,
            workspace_peak_bytes,
            len(recompute_ops),
            peak_payload_bytes,
            total_step_us,
            confidence,
            ("m0_static",),
        )

    live_storage_ids: set[int] = set()
    live_bytes = 0
    peak_payload_bytes = 0
    workspace_peak_bytes = 0
    terminal_peak_bytes = 0
    total_step_us = 0.0
    confidence = 1.0
    sources: set[str] = set()

    for index, op in enumerate(recompute_ops):
        op_us, workspace_bytes, op_confidence, source = _provider_cost(
            cost_provider,
            ir,
            op.id,
            cached_op_costs=cached_op_costs,
        )
        total_step_us += op_us
        workspace_peak_bytes = max(workspace_peak_bytes, workspace_bytes)
        confidence = min(confidence, op_confidence)
        sources.add(source)

        produced_storage_ids = {
            value_by_id[value_id].storage_id
            for value_id in op.output_value_ids
            if (
                value_id in value_by_id
                and value_by_id[value_id].storage_id is not None
                and value_by_id[value_id].storage_id not in saved_storage_ids
                and value_by_id[value_id].storage_id in bytes_by_storage
            )
        }
        for storage_id in sorted(produced_storage_ids):
            if storage_id in terminal_storage_ids:
                terminal_peak_bytes = max(terminal_peak_bytes, bytes_by_storage[storage_id])
                continue
            if storage_id not in live_storage_ids:
                live_storage_ids.add(storage_id)
                live_bytes += bytes_by_storage[storage_id]

        peak_payload_bytes = max(peak_payload_bytes, live_bytes + terminal_peak_bytes)

        if disable_lifetime:
            continue
        releasable = [
            storage_id
            for storage_id in live_storage_ids
            if storage_last_use.get(storage_id, -1) <= index and storage_id not in saved_storage_ids
        ]
        for storage_id in releasable:
            live_storage_ids.remove(storage_id)
            live_bytes = apply_event(live_bytes, -bytes_by_storage[storage_id])

    recompute_before_first_bw = peak_payload_bytes
    return (
        peak_payload_bytes,
        workspace_peak_bytes,
        len(recompute_ops),
        recompute_before_first_bw,
        total_step_us,
        confidence,
        tuple(sorted(sources)) or ("m0_static",),
    )


def simulate_plan(
    ir: JointTrainingIR,
    plan: RecomputePlan,
    fixed_timeline: FixedTimeline,
    *,
    cost_provider: CostProvider | None = None,
    cost_cache: SimulationCostCache | None = None,
    materialize_event_trace: bool = True,
    recompute_closure: RecomputeClosure | None = None,
    components: Mapping[str, bool] | None = None,
) -> SimulationResult:
    """Full training-step memory simulation for one recomputation candidate.

    ``components`` (ablation only; default ``None`` = full model) maps a
    mechanism name to whether it is *enabled*. A ``False`` value disables that
    mechanism: ``opt`` (optimizer-state + optimizer-temporary accounting),
    ``workspace`` (per-op workspace bytes), ``materialization`` (compiler
    mandatory-workspace + runtime-replica residency), ``lifetime`` (last-use
    release in forward liveness and replay liveness), ``replay`` (fall back to
    the all-simultaneous M0 recompute model), and ``alias`` (view/alias storage
    sharing; ``False`` counts each aliased value's logical bytes separately).
    """
    if components is None:
        components = {}
    disable = lambda name: components.get(name, True) is False
    ft = fixed_timeline
    if disable("opt"):
        ft = replace(ft, optimizer_state_bytes=0, optimizer_temporary_bytes=0)
    if disable("materialization"):
        ft = replace(
            ft,
            mandatory_workspace_bytes=0,
            runtime_replica_bytes=0,
        )

    if cost_cache is not None:
        cost_cache.validate_for(ir, fixed_timeline, cost_provider)
    bytes_by_storage = (
        _storage_bytes(ir) if cost_cache is None else cost_cache.bytes_by_storage
    )
    if disable("alias"):
        # No view/alias storage sharing: each value's logical bytes counted
        # separately, so aliased views multiply their footprint.
        value_by_id = {value.id: value for value in ir.values}
        no_alias: dict[int, int] = {}
        for storage in ir.storages:
            total = sum(
                value.logical_nbytes
                for value in ir.values
                if value.storage_id == storage.id
            )
            no_alias[storage.id] = max(storage.physical_nbytes, total)
        bytes_by_storage = no_alias
    activation_ids = (
        _activation_storage_ids(ir)
        if cost_cache is None
        else cost_cache.activation_storage_ids
    )
    saved_storage_ids = (
        _plan_saved_storage_ids(ir, plan)
        if cost_cache is None
        else _plan_saved_storage_ids_cached(plan, cost_cache.value_storage_by_id)
    ) & activation_ids
    dropped_storage_ids = activation_ids - saved_storage_ids
    closure = recompute_closure or derive_recompute_closure(
        ir,
        plan.saved_value_ids | plan.mandatory_value_ids,
        graph_cache=None if cost_cache is None else cost_cache.recompute_graph_cache,
    )

    saved_bytes = sum(bytes_by_storage[sid] for sid in saved_storage_ids)
    (
        recompute_bytes,
        workspace_bytes,
        recompute_span_ops,
        recompute_before_first_bw,
        recompute_step_us,
        cost_confidence,
        _cost_sources,
    ) = _recompute_liveness_peak(
        ir,
        closure,
        saved_storage_ids,
        bytes_by_storage,
        cost_provider=cost_provider,
        cached_op_costs=None if cost_cache is None else cost_cache.op_costs,
        cached_value_by_id=(
            None if cost_cache is None else cost_cache.recompute_graph_cache.value_by_id
        ),
        cached_fw_ops=None if cost_cache is None else cost_cache.fw_ops,
        disable_lifetime=disable("lifetime"),
        disable_replay=disable("replay"),
    )
    if cost_cache is None:
        full_graph_cost = _full_graph_step_cost(ir, cost_provider)
        optimizer_cost = _optimizer_step_cost(ft, cost_provider)
    else:
        full_graph_cost = cost_cache.full_graph_step_cost
        optimizer_cost = cost_cache.optimizer_step_cost
    full_graph_step_us, full_graph_confidence, full_graph_sources, cost_breakdown = full_graph_cost
    optimizer_step_us, optimizer_confidence, optimizer_source = optimizer_cost
    cost_breakdown = {
        **cost_breakdown,
        "full_graph_step_us": full_graph_step_us,
        "recompute_step_us": recompute_step_us,
        "recompute_closure_op_count": len(closure.recomputed_op_ids),
        "recompute_closure_value_count": len(closure.recomputed_value_ids),
        "recompute_barrier_value_ids": tuple(sorted(closure.barrier_value_ids)),
        "optimizer_step_us": optimizer_step_us,
        "optimizer_source": optimizer_source,
        "cost_sources": tuple(sorted({*full_graph_sources, optimizer_source, *_cost_sources})),
    }
    fw_live_peak = _forward_liveness_peak(
        ir,
        saved_storage_ids,
        dropped_storage_ids,
        bytes_by_storage,
        cached_fw_ops=None if cost_cache is None else cost_cache.fw_ops,
        cached_produced_storage_ids_by_op=(
            None if cost_cache is None else cost_cache.fw_produced_storage_ids_by_op
        ),
        cached_storage_last_use=(
            None if cost_cache is None else cost_cache.fw_storage_last_use
        ),
        disable_lifetime=disable("lifetime"),
    )
    full_graph_workspace_by_phase = dict(cost_breakdown.get("workspace_peak_by_phase_bytes") or {})
    fw_workspace_bytes = int(full_graph_workspace_by_phase.get("fw", 0))
    bw_workspace_bytes = max(workspace_bytes, int(full_graph_workspace_by_phase.get("bw", 0)))
    if disable("workspace"):
        fw_workspace_bytes = 0
        bw_workspace_bytes = 0
    fw_peak = (
        ft.forward_resident_bytes
        + fw_live_peak
        + fw_workspace_bytes
    )
    after_fw = (
        ft.forward_resident_bytes
        + saved_bytes
    )
    bw_peak = ft.resident_bytes + saved_bytes + recompute_bytes + bw_workspace_bytes
    optimizer_peak = ft.peak_lower_bound_bytes
    cost_breakdown["memory_components"] = {
        "parameter_bytes": ft.parameter_bytes,
        "buffer_bytes": ft.buffer_bytes,
        "gradient_bytes": ft.gradient_bytes,
        "optimizer_state_bytes": ft.optimizer_state_bytes,
        "optimizer_temporary_bytes": ft.optimizer_temporary_bytes,
        "mandatory_workspace_bytes": ft.mandatory_workspace_bytes,
        "runtime_replica_bytes": ft.runtime_replica_bytes,
        "runtime_replica_source": (
            "aot_compiled_candidate_residency"
            if ft.runtime_replica_bytes
            else "none"
        ),
        "forward_fixed_resident_bytes": ft.forward_resident_bytes,
        "saved_activation_bytes": saved_bytes,
        "forward_activation_live_peak_bytes": fw_live_peak,
        "recompute_live_peak_bytes": recompute_bytes,
        "workspace_peak_bytes": bw_workspace_bytes,
        "recompute_workspace_peak_bytes": workspace_bytes,
        "full_graph_fw_workspace_peak_bytes": fw_workspace_bytes,
        "full_graph_bw_workspace_peak_bytes": int(full_graph_workspace_by_phase.get("bw", 0)),
        "fw_peak_bytes": fw_peak,
        "after_fw_retained_bytes": after_fw,
        "bw_peak_bytes": bw_peak,
        "optimizer_peak_bytes": optimizer_peak,
    }
    overall = max(fw_peak, bw_peak, optimizer_peak)
    simulated_event_trace = ()
    if materialize_event_trace:
        simulated_event_trace = _simulated_memory_event_trace(
            ir,
            saved_storage_ids,
            dropped_storage_ids,
            bytes_by_storage,
            fixed_timeline,
            cost_provider=cost_provider,
            cached_op_costs=None if cost_cache is None else cost_cache.op_costs,
            optimizer_step_us=optimizer_step_us,
            recompute_peak_bytes=recompute_bytes,
            recompute_workspace_bytes=bw_workspace_bytes,
            recompute_step_us=recompute_step_us,
            expected_bw_peak_bytes=bw_peak,
        )
    if overall == bw_peak:
        snapshot = compute_peak_snapshot(
            "bw",
            None,
            frozenset(saved_storage_ids | dropped_storage_ids),
            saved_bytes,
            ft,
            recomputed_bytes=recompute_bytes,
            workspace_bytes=bw_workspace_bytes,
        )
    elif overall == optimizer_peak:
        snapshot = compute_peak_snapshot("optimizer", None, frozenset(), 0, ft)
    else:
        snapshot = compute_peak_snapshot(
            "fw",
            None,
            frozenset(activation_ids),
            fw_live_peak,
            ft,
            workspace_bytes=fw_workspace_bytes,
        )

    estimated_step_us = float(full_graph_step_us + recompute_step_us + optimizer_step_us)
    confidence = min(
        0.9 if not dropped_storage_ids else 0.75,
        cost_confidence,
        full_graph_confidence,
        optimizer_confidence,
    )
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
        recompute_before_first_bw_op_bytes=recompute_before_first_bw,
        risk_score=0.2 if dropped_storage_ids else 0.05,
        confidence=confidence,
        cost_breakdown=cost_breakdown,
        simulated_memory_event_trace=simulated_event_trace,
    )


def compare_plan_delta(baseline: SimulationResult, candidate: SimulationResult) -> dict[str, int]:
    return {
        "estimated_peak_delta_bytes": candidate.estimated_peak_bytes - baseline.estimated_peak_bytes,
        "after_fw_retained_delta_bytes": candidate.after_fw_retained_bytes - baseline.after_fw_retained_bytes,
        "bw_peak_delta_bytes": candidate.bw_peak_bytes - baseline.bw_peak_bytes,
    }
