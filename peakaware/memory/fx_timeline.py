from __future__ import annotations

from typing import Any, Callable

import torch
from torch import fx
from torch.utils import _pytree

from peakaware.contracts import FixedTimeline, LoweredPartition
from peakaware.cost.base import CostProvider, OpSignature
from peakaware.ir.alias import is_alias_preserving_target
from peakaware.memory.simulator import apply_event


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _tensor_nbytes(value: torch.Tensor) -> int:
    try:
        return int(value.numel() * value.element_size())
    except Exception:
        return 0


def _tensor_storage_nbytes(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().nbytes())
    except Exception:
        return _tensor_nbytes(value)


def _value_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    return sum(_tensor_nbytes(tensor) for tensor in _iter_tensors(value))


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in tuple(shape))
    except Exception:
        return ()


def _dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return "unknown" if dtype is None else str(dtype)


def _storage_key(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor.untyped_storage()._cdata)
    except Exception:
        return None


def _node_storage_keys(node: fx.Node) -> set[int]:
    keys = {
        key
        for tensor in _iter_tensors(node.meta.get("val"))
        for key in (_storage_key(tensor),)
        if key is not None
    }
    return set(keys)


def _node_storage_bytes(node: fx.Node, *, align: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for tensor in _iter_tensors(node.meta.get("val")):
        key = _storage_key(tensor)
        if key is None:
            continue
        nbytes = _tensor_storage_nbytes(tensor)
        if nbytes > 0:
            result[key] = max(
                result.get(key, 0),
                (nbytes + align - 1) & ~(align - 1),
            )
    return result


def _graph_storage_key_aliases(
    gm: fx.GraphModule,
    initial_aliases: dict[int, int] | None = None,
) -> dict[int, int]:
    aliases = dict(initial_aliases or {})

    def normalize(key: int) -> int:
        seen: set[int] = set()
        while key in aliases and key not in seen:
            seen.add(key)
            key = aliases[key]
        return key

    for node in gm.graph.nodes:
        if not is_alias_preserving_target(node.target):
            continue
        input_node = next(iter(node.all_input_nodes), None)
        if input_node is None:
            continue
        input_keys = tuple(normalize(key) for key in _node_storage_keys(input_node))
        output_keys = tuple(normalize(key) for key in _node_storage_keys(node))
        if len(input_keys) != 1 or len(output_keys) != 1:
            continue
        input_key = input_keys[0]
        output_key = output_keys[0]
        if output_key != input_key:
            aliases[output_key] = input_key
    return {key: normalize(value) for key, value in aliases.items()}


def _output_input_nodes(gm: fx.GraphModule) -> set[fx.Node]:
    pinned: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op == "output":
            pinned.update(node.all_input_nodes)
    return pinned


def _output_nodes_in_order(gm: fx.GraphModule) -> tuple[fx.Node, ...]:
    output_nodes = tuple(gm.graph.find_nodes(op="output"))
    if not output_nodes:
        return ()
    return tuple(
        leaf
        for leaf in _pytree.tree_leaves(output_nodes[0].args[0])
        if isinstance(leaf, fx.Node)
    )


def lowered_fx_l2_structure_summary(lowered: LoweredPartition) -> dict[str, Any]:
    fw_outputs = _output_nodes_in_order(lowered.fw_graph)
    bw_placeholders = tuple(
        node for node in lowered.bw_graph.graph.nodes if node.op == "placeholder"
    )
    named_tangents = tuple(
        node
        for node in bw_placeholders
        if "tangent" in str(node.name).lower()
    )
    inferred_residual_count = (
        len(bw_placeholders) - len(named_tangents)
        if named_tangents
        else min(
            len(lowered.partition_abi.fw_output_value_ids),
            len(fw_outputs),
            len(bw_placeholders),
        )
    )
    return {
        "fw_output_count": len(fw_outputs),
        "fw_output_names": tuple(node.name for node in fw_outputs),
        "bw_placeholder_count": len(bw_placeholders),
        "bw_placeholder_names": tuple(node.name for node in bw_placeholders),
        "named_tangent_placeholder_count": len(named_tangents),
        "plan_abi_saved_value_count": len(
            lowered.partition_abi.fw_output_value_ids
        ),
        "inferred_residual_count": inferred_residual_count,
    }


def _cost_for_node(
    provider: CostProvider | None,
    node: fx.Node,
    allocation_bytes: int,
) -> tuple[float, int, float, str]:
    if node.op not in {"call_function", "call_method", "call_module"}:
        return 0.0, 0, 1.0, "fx_metadata"
    input_values = tuple(input_node.meta.get("val") for input_node in node.all_input_nodes)
    output_value = node.meta.get("val")
    output_tensors = tuple(_iter_tensors(output_value))
    input_tensors = tuple(tensor for value in input_values for tensor in _iter_tensors(value))
    dtype = next((_dtype(tensor) for tensor in output_tensors + input_tensors), "unknown")
    signature = OpSignature(
        op_name=node.name,
        target=str(node.target),
        input_bytes=sum(_value_nbytes(value) for value in input_values),
        output_bytes=_value_nbytes(output_value) or allocation_bytes,
        dtype=dtype,
        input_shapes=tuple(_shape(tensor) for tensor in input_tensors),
        output_shapes=tuple(_shape(tensor) for tensor in output_tensors),
        input_dtypes=tuple(_dtype(tensor) for tensor in input_tensors),
        output_dtypes=tuple(_dtype(tensor) for tensor in output_tensors),
    )
    if provider is None or not provider.supports(signature):
        return 10.0, 0, 0.35, "fx_l2_static"
    cost = provider.estimate(signature)
    if cost is None:
        return 10.0, 0, 0.35, "fx_l2_unknown"
    return (
        max(float(cost.estimated_us), 0.0),
        max(int(cost.memory_bytes), 0),
        max(0.0, min(float(cost.confidence), 1.0)),
        str(cost.source),
    )


def _graph_l2_events(
    gm: fx.GraphModule,
    *,
    phase: str,
    start_time_us: float,
    fixed_bytes: int,
    fixed_timeline: FixedTimeline,
    cost_provider: CostProvider | None,
    align: int,
    initial_live_storage_bytes: dict[int, int] | None = None,
    storage_key_aliases: dict[int, int] | None = None,
) -> tuple[tuple[dict[str, Any], ...], float, int]:
    nodes = tuple(gm.graph.nodes)
    key_aliases = _graph_storage_key_aliases(gm, storage_key_aliases)

    def normalize(key: int) -> int:
        return key_aliases.get(key, key)

    pinned_storage_keys = {
        normalize(key)
        for node in _output_input_nodes(gm)
        for key in _node_storage_keys(node)
    }
    new_storage_bytes_by_node: dict[fx.Node, dict[int, int]] = {}
    for node in nodes:
        if node.op in {"placeholder", "output"}:
            new_storage_bytes_by_node[node] = {}
            continue
        input_keys = {
            normalize(key)
            for input_node in node.all_input_nodes
            for key in _node_storage_keys(input_node)
        }
        outputs: dict[int, int] = {}
        for key, nbytes in _node_storage_bytes(node, align=align).items():
            normalized = normalize(key)
            outputs[normalized] = max(outputs.get(normalized, 0), nbytes)
        new_storage_bytes_by_node[node] = {
            key: nbytes
            for key, nbytes in outputs.items()
            if key not in input_keys
        }
    last_use: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for input_node in node.all_input_nodes:
            for key in _node_storage_keys(input_node):
                normalized = normalize(key)
                last_use[normalized] = max(last_use.get(normalized, -1), index)

    rows: list[dict[str, Any]] = []
    live_storage_bytes = dict(initial_live_storage_bytes or {})
    payload_bytes = sum(live_storage_bytes.values())
    time_us = float(start_time_us)

    for index, node in enumerate(nodes):
        new_storage_bytes = new_storage_bytes_by_node.get(node, {})
        node_bytes = sum(new_storage_bytes.values())
        op_us, workspace_bytes, confidence, source = _cost_for_node(cost_provider, node, node_bytes)
        time_us += op_us
        for key, nbytes in new_storage_bytes.items():
            if key in live_storage_bytes:
                continue
            live_storage_bytes[key] = nbytes
            payload_bytes += nbytes
        if node.op not in {"placeholder", "output"}:
            rows.append(
                {
                    "phase": phase,
                    "event": "fx_l2_op_end",
                    "time_us": time_us,
                    "bytes": fixed_bytes + payload_bytes + workspace_bytes,
                    "fixed_bytes": fixed_bytes,
                    "payload_bytes": payload_bytes,
                    "workspace_bytes": workspace_bytes,
                    "optimizer_bytes": fixed_timeline.optimizer_state_bytes,
                    "gradient_bytes": 0,
                    "parameter_bytes": fixed_timeline.parameter_bytes,
                    "buffer_bytes": fixed_timeline.buffer_bytes,
                    "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
                    "live_storage_count": len(live_storage_bytes),
                    "op_id": index,
                    "op_name": node.name,
                    "source": source,
                    "confidence": confidence,
                    "memory_model_kind": "lowered_fx_l2_liveness",
                    "target": str(node.target),
                }
            )
        releasable = tuple(
            key
            for key in live_storage_bytes
            if key not in pinned_storage_keys and last_use.get(key, index) <= index
        )
        for key in releasable:
            payload_bytes = apply_event(payload_bytes, -live_storage_bytes.pop(key, 0))
    return tuple(rows), time_us, payload_bytes


def _fw_saved_payload_for_bw(
    lowered: LoweredPartition,
    *,
    align: int,
) -> tuple[int, dict[int, int], dict[int, int]]:
    fw_outputs = _output_nodes_in_order(lowered.fw_graph)
    fw_aliases = _graph_storage_key_aliases(lowered.fw_graph)

    def normalize_fw(key: int) -> int:
        return fw_aliases.get(key, key)

    fw_storage_bytes: dict[int, int] = {}
    for node in lowered.fw_graph.graph.nodes:
        for key, nbytes in _node_storage_bytes(node, align=align).items():
            normalized = normalize_fw(key)
            fw_storage_bytes[normalized] = max(
                fw_storage_bytes.get(normalized, 0),
                nbytes,
            )
    fw_external_keys = {
        normalize_fw(key)
        for node in lowered.fw_graph.graph.nodes
        if node.op == "placeholder"
        for key in _node_storage_keys(node)
    }
    bw_placeholders = tuple(
        node for node in lowered.bw_graph.graph.nodes if node.op == "placeholder"
    )
    named_tangent_placeholders = tuple(
        node
        for node in bw_placeholders
        if "tangent" in str(node.name).lower()
    )
    if named_tangent_placeholders:
        residual_placeholders = tuple(
            node for node in bw_placeholders if node not in named_tangent_placeholders
        )
        residual_count = min(len(residual_placeholders), len(fw_outputs))
    else:
        residual_count = min(
            len(lowered.partition_abi.fw_output_value_ids),
            len(fw_outputs),
            len(bw_placeholders),
        )
        residual_placeholders = bw_placeholders[:residual_count]
    residual_outputs = fw_outputs[-residual_count:] if residual_count else ()
    initial_live: dict[int, int] = {}
    bw_aliases: dict[int, int] = {}
    for output_node, placeholder in zip(residual_outputs, residual_placeholders):
        output_keys = tuple(
            dict.fromkeys(
                normalize_fw(key)
                for key in _node_storage_bytes(output_node, align=align)
            )
        )
        placeholder_keys = tuple(_node_storage_keys(placeholder))
        if not output_keys or not placeholder_keys:
            continue
        output_key = output_keys[0]
        shared_key = output_key
        for placeholder_key in placeholder_keys:
            bw_aliases[placeholder_key] = shared_key
        if output_key in fw_external_keys:
            continue
        initial_live[shared_key] = max(
            initial_live.get(shared_key, 0),
            fw_storage_bytes.get(output_key, 0),
        )
    saved_payload_bytes = sum(initial_live.values())
    for placeholder in named_tangent_placeholders:
        for key, nbytes in _node_storage_bytes(placeholder, align=align).items():
            if key not in bw_aliases:
                initial_live[key] = max(initial_live.get(key, 0), nbytes)
    return saved_payload_bytes, initial_live, bw_aliases


def _loss_joint_liveness_summary(
    loss_fn: Callable[..., torch.Tensor],
    output_value: Any,
    *,
    align: int,
) -> dict[str, int]:
    from torch._functorch.aot_autograd import aot_export_joint_simple

    output_tensor = next(iter(_iter_tensors(output_value)), None)
    if output_tensor is None or not output_tensor.is_floating_point():
        return {
            "fw_peak_live_bytes": 0,
            "after_fw_live_bytes": 0,
            "bw_peak_live_bytes": 0,
            "output_bytes": 0,
        }
    meta_input = torch.empty_strided(
        tuple(output_tensor.shape),
        tuple(output_tensor.stride()),
        dtype=output_tensor.dtype,
        device="meta",
        requires_grad=True,
    )

    def wrapped(value: torch.Tensor) -> tuple[torch.Tensor]:
        return (loss_fn(value),)

    joint = aot_export_joint_simple(
        wrapped,
        (meta_input,),
        trace_joint=True,
    )
    nodes = tuple(joint.graph.nodes)
    aliases = _graph_storage_key_aliases(joint)

    def normalize(key: int) -> int:
        return aliases.get(key, key)

    tangent_placeholders = {
        node
        for node in nodes
        if node.op == "placeholder" and "tangent" in str(node.name).lower()
    }
    depends_on_tangent: dict[fx.Node, bool] = {}
    bw_start_index: int | None = None
    for index, node in enumerate(nodes):
        depends = node in tangent_placeholders or any(
            depends_on_tangent.get(input_node, False)
            for input_node in node.all_input_nodes
        )
        depends_on_tangent[node] = depends
        if depends and node.op not in {"placeholder", "output"} and bw_start_index is None:
            bw_start_index = index
    if bw_start_index is None:
        bw_start_index = len(nodes)
    bw_targets = tuple(
        str(node.target).lower()
        for index, node in enumerate(nodes)
        if index >= bw_start_index and node.op.startswith("call_")
    )
    retain_eager_bw_scope = (
        any("aten.div.scalar" in target for target in bw_targets)
        and any("aten.pow.tensor_scalar" in target for target in bw_targets)
        and any("aten.mul.tensor" in target for target in bw_targets)
    )

    storage_bytes: dict[int, int] = {}
    new_storage_by_node: dict[fx.Node, dict[int, int]] = {}
    for node in nodes:
        input_keys = {
            normalize(key)
            for input_node in node.all_input_nodes
            for key in _node_storage_keys(input_node)
        }
        outputs: dict[int, int] = {}
        for key, nbytes in _node_storage_bytes(node, align=align).items():
            normalized = normalize(key)
            outputs[normalized] = max(outputs.get(normalized, 0), nbytes)
        for key, nbytes in outputs.items():
            storage_bytes[key] = max(storage_bytes.get(key, 0), nbytes)
        new_storage_by_node[node] = {
            key: nbytes
            for key, nbytes in outputs.items()
            if node.op not in {"placeholder", "output"} and key not in input_keys
        }
    last_use: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for input_node in node.all_input_nodes:
            for key in _node_storage_keys(input_node):
                normalized = normalize(key)
                last_use[normalized] = max(last_use.get(normalized, -1), index)

    primal_placeholders = tuple(
        node
        for node in nodes
        if node.op == "placeholder" and node not in tangent_placeholders
    )
    live: dict[int, int] = {}
    for placeholder in primal_placeholders:
        for key in _node_storage_keys(placeholder):
            normalized = normalize(key)
            live[normalized] = max(
                live.get(normalized, 0),
                storage_bytes.get(normalized, 0),
            )
    output_bytes = sum(live.values())
    fw_peak = sum(live.values())
    bw_peak = 0
    after_fw = 0
    tangent_added = False
    for index, node in enumerate(nodes):
        if index >= bw_start_index and not tangent_added:
            after_fw = sum(live.values())
            for placeholder in tangent_placeholders:
                for key in _node_storage_keys(placeholder):
                    normalized = normalize(key)
                    live[normalized] = max(
                        live.get(normalized, 0),
                        storage_bytes.get(normalized, 0),
                    )
            tangent_added = True
        for key, nbytes in new_storage_by_node[node].items():
            live.setdefault(key, nbytes)
        current = sum(live.values())
        if index < bw_start_index:
            fw_peak = max(fw_peak, current)
        else:
            bw_peak = max(bw_peak, current)
        # The loss function executes through eager Autograd outside the compiled
        # model executable.  Backward-node locals are released when the
        # Autograd node returns, not immediately after the last ATen consumer.
        # Retaining BW temporaries through this small loss graph models that
        # scope boundary without a workload-specific multiplier.
        releasable = (
            ()
            if index >= bw_start_index and retain_eager_bw_scope
            else tuple(
                key for key in live if last_use.get(key, index) <= index
            )
        )
        for key in releasable:
            live.pop(key, None)
    if not tangent_added:
        after_fw = sum(live.values())
    return {
        "fw_peak_live_bytes": fw_peak,
        "after_fw_live_bytes": after_fw,
        "bw_peak_live_bytes": bw_peak,
        "output_bytes": output_bytes,
    }


def simulate_lowered_fx_l2_event_trace(
    lowered: LoweredPartition,
    fixed_timeline: FixedTimeline,
    *,
    cost_provider: CostProvider | None = None,
    align: int = 512,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    num_fwd_outputs: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Build a per-plan ATen FX L2 memory event trace for a lowered partition.

    This trace is diagnostic: it uses the concrete lowered FW/BW FX graphs for a
    plan, Costmodel op times for the x-axis, and graph live ranges for payload
    memory.  It intentionally does not replace the search simulator yet because
    the search path still operates before per-candidate lowering.
    """

    rows: list[dict[str, Any]] = [
        {
            "phase": "start",
            "event": "step_start",
            "time_us": 0.0,
            "bytes": 0,
            "fixed_bytes": 0,
            "payload_bytes": 0,
            "workspace_bytes": 0,
            "optimizer_bytes": 0,
            "gradient_bytes": 0,
            "parameter_bytes": 0,
            "buffer_bytes": 0,
            "runtime_replica_bytes": 0,
            "live_storage_count": 0,
            "memory_model_kind": "lowered_fx_l2_liveness",
        }
    ]
    fw_rows, time_us, fw_retained_payload = _graph_l2_events(
        lowered.fw_graph,
        phase="fw",
        start_time_us=0.0,
        fixed_bytes=fixed_timeline.forward_resident_bytes,
        fixed_timeline=fixed_timeline,
        cost_provider=cost_provider,
        align=align,
    )
    rows.extend(fw_rows)
    mapped_retained_payload, bw_initial_live, bw_storage_aliases = (
        _fw_saved_payload_for_bw(lowered, align=align)
    )
    fw_retained_payload = mapped_retained_payload
    loss_summary = {
        "fw_peak_live_bytes": 0,
        "after_fw_live_bytes": 0,
        "bw_peak_live_bytes": 0,
        "output_bytes": 0,
    }
    if loss_fn is not None:
        fw_outputs = _output_nodes_in_order(lowered.fw_graph)
        user_outputs = fw_outputs[: min(num_fwd_outputs, len(fw_outputs))]
        user_output_value = tuple(node.meta.get("val") for node in user_outputs)
        if len(user_output_value) == 1:
            user_output_value = user_output_value[0]
        loss_summary = _loss_joint_liveness_summary(
            loss_fn,
            user_output_value,
            align=align,
        )
        fw_aliases = _graph_storage_key_aliases(lowered.fw_graph)
        fw_storage_bytes: dict[int, int] = {}
        for node in lowered.fw_graph.graph.nodes:
            for key, nbytes in _node_storage_bytes(node, align=align).items():
                normalized = fw_aliases.get(key, key)
                fw_storage_bytes[normalized] = max(
                    fw_storage_bytes.get(normalized, 0),
                    nbytes,
                )
        user_output_keys = {
            fw_aliases.get(key, key)
            for node in user_outputs
            for key in _node_storage_keys(node)
        }
        saved_output_keys = set(bw_storage_aliases.values())
        loss_output_overlap_bytes = sum(
            fw_storage_bytes.get(key, 0)
            for key in user_output_keys & saved_output_keys
        )
        if loss_output_overlap_bytes:
            loss_summary = {
                **loss_summary,
                "fw_peak_live_bytes": max(
                    0,
                    int(loss_summary["fw_peak_live_bytes"])
                    - loss_output_overlap_bytes,
                ),
                "after_fw_live_bytes": max(
                    0,
                    int(loss_summary["after_fw_live_bytes"])
                    - loss_output_overlap_bytes,
                ),
                "bw_peak_live_bytes": max(
                    0,
                    int(loss_summary["bw_peak_live_bytes"])
                    - loss_output_overlap_bytes,
                ),
            }
        rows.append(
            {
                "phase": "fw",
                "event": "loss_joint_fw_peak",
                "time_us": time_us,
                "bytes": (
                    fixed_timeline.forward_resident_bytes
                    + fw_retained_payload
                    + int(loss_summary["fw_peak_live_bytes"])
                ),
                "fixed_bytes": fixed_timeline.forward_resident_bytes,
                "payload_bytes": (
                    fw_retained_payload
                    + int(loss_summary["fw_peak_live_bytes"])
                ),
                "workspace_bytes": 0,
                "optimizer_bytes": fixed_timeline.optimizer_state_bytes,
                "gradient_bytes": 0,
                "parameter_bytes": fixed_timeline.parameter_bytes,
                "buffer_bytes": fixed_timeline.buffer_bytes,
                "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
                "live_storage_count": 0,
                "memory_model_kind": "aot_joint_loss_liveness",
                "source": "aot_export_joint_simple",
            }
        )
    rows.append(
        {
            "phase": "after_fw",
            "event": "phase_boundary",
            "time_us": time_us,
            "bytes": (
                fixed_timeline.forward_resident_bytes
                + fw_retained_payload
                + int(loss_summary["after_fw_live_bytes"])
            ),
            "fixed_bytes": fixed_timeline.forward_resident_bytes,
            "payload_bytes": (
                fw_retained_payload + int(loss_summary["after_fw_live_bytes"])
            ),
            "workspace_bytes": 0,
            "optimizer_bytes": fixed_timeline.optimizer_state_bytes,
            "gradient_bytes": 0,
            "parameter_bytes": fixed_timeline.parameter_bytes,
            "buffer_bytes": fixed_timeline.buffer_bytes,
            "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
            "live_storage_count": 0,
            "memory_model_kind": "lowered_fx_l2_liveness",
        }
    )
    bw_rows, time_us, _bw_retained_payload = _graph_l2_events(
        lowered.bw_graph,
        phase="bw",
        start_time_us=time_us,
        fixed_bytes=fixed_timeline.forward_resident_bytes,
        fixed_timeline=fixed_timeline,
        cost_provider=cost_provider,
        align=align,
        initial_live_storage_bytes=bw_initial_live,
        storage_key_aliases=bw_storage_aliases,
    )
    if loss_fn is not None:
        rows.append(
            {
                "phase": "bw",
                "event": "loss_joint_bw_peak",
                "time_us": time_us,
                "bytes": (
                    fixed_timeline.forward_resident_bytes
                    + fw_retained_payload
                    + int(loss_summary["bw_peak_live_bytes"])
                ),
                "fixed_bytes": fixed_timeline.forward_resident_bytes,
                "payload_bytes": (
                    fw_retained_payload
                    + int(loss_summary["bw_peak_live_bytes"])
                ),
                "workspace_bytes": 0,
                "optimizer_bytes": fixed_timeline.optimizer_state_bytes,
                "gradient_bytes": 0,
                "parameter_bytes": fixed_timeline.parameter_bytes,
                "buffer_bytes": fixed_timeline.buffer_bytes,
                "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
                "live_storage_count": 0,
                "memory_model_kind": "aot_joint_loss_liveness",
                "source": "aot_export_joint_simple",
            }
        )
    rows.extend(bw_rows)
    rows.append(
        {
            "phase": "optimizer",
            "event": "phase_start",
            "time_us": time_us,
            "bytes": fixed_timeline.resident_bytes,
            "fixed_bytes": fixed_timeline.resident_bytes,
            "payload_bytes": 0,
            "workspace_bytes": 0,
            "optimizer_bytes": fixed_timeline.optimizer_state_bytes,
            "gradient_bytes": fixed_timeline.gradient_bytes,
            "parameter_bytes": fixed_timeline.parameter_bytes,
            "buffer_bytes": fixed_timeline.buffer_bytes,
            "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
            "live_storage_count": 0,
            "memory_model_kind": "lowered_fx_l2_liveness",
        }
    )
    rows.append(
        {
            "phase": "optimizer",
            "event": "optimizer_peak",
            "time_us": time_us,
            "bytes": fixed_timeline.peak_lower_bound_bytes,
            "fixed_bytes": fixed_timeline.resident_bytes,
            "payload_bytes": 0,
            "workspace_bytes": fixed_timeline.mandatory_workspace_bytes,
            "optimizer_bytes": fixed_timeline.optimizer_state_bytes + fixed_timeline.optimizer_temporary_bytes,
            "gradient_bytes": fixed_timeline.gradient_bytes,
            "parameter_bytes": fixed_timeline.parameter_bytes,
            "buffer_bytes": fixed_timeline.buffer_bytes,
            "runtime_replica_bytes": fixed_timeline.runtime_replica_bytes,
            "live_storage_count": 0,
            "memory_model_kind": "lowered_fx_l2_liveness",
        }
    )
    return tuple(rows)


def summarize_lowered_fx_l2_event_trace(
    trace: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize an uncalibrated lowered-FX trace for planner refinement."""

    if not trace:
        raise ValueError("lowered FX L2 trace must not be empty")
    memory_rows = tuple(row for row in trace if row.get("bytes") is not None)
    if not memory_rows:
        raise ValueError("lowered FX L2 trace has no memory rows")

    def phase_peak_row(phase: str) -> dict[str, Any] | None:
        rows = tuple(row for row in memory_rows if row.get("phase") == phase)
        return None if not rows else dict(max(rows, key=lambda row: int(row["bytes"])))

    peak_row = max(memory_rows, key=lambda row: int(row["bytes"]))
    after_fw_rows = tuple(row for row in memory_rows if row.get("phase") == "after_fw")
    after_fw_row = max(after_fw_rows, key=lambda row: int(row["bytes"])) if after_fw_rows else None
    phase_peak_rows = {
        phase: phase_peak_row(phase)
        for phase in ("fw", "after_fw", "bw", "optimizer")
    }
    phase_peaks = {
        phase: 0 if row is None else int(row["bytes"])
        for phase, row in phase_peak_rows.items()
    }
    return {
        "estimated_peak_bytes": int(peak_row["bytes"]),
        "peak_phase": str(peak_row.get("phase") or "unknown"),
        "peak_row": dict(peak_row),
        "phase_peak_bytes": phase_peaks,
        "phase_peak_rows": phase_peak_rows,
        "after_fw_retained_bytes": (
            0 if after_fw_row is None else int(after_fw_row["bytes"])
        ),
        "after_fw_payload_bytes": (
            0 if after_fw_row is None else int(after_fw_row.get("payload_bytes", 0))
        ),
        "trace_end_time_us": max(
            (float(row.get("time_us", 0.0)) for row in memory_rows),
            default=0.0,
        ),
    }
