from __future__ import annotations

from typing import Any

import torch
from torch import fx
from torch.utils import _pytree

from peakaware.contracts import (
    CapturedJointGraph,
    IRValidationReport,
    JointTrainingIR,
    OpInfo,
    RegionInfo,
    StorageInfo,
    ValueInfo,
)

from .alias import (
    build_storage_groups,
    is_alias_preserving_target,
    merge_alias_storage_groups,
    validate_alias_invariants,
)
from .legality import classify_recompute_legality, mark_mandatory_saves


def _node_value(node: fx.Node) -> Any:
    if "val" in node.meta:
        return node.meta["val"]
    return node.meta.get("tensor_meta")


def _logical_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            element_size = torch.empty((), dtype=dtype).element_size()
            numel = 1
            for dim in tuple(shape):
                numel *= int(dim)
            return int(numel * element_size)
        except Exception:
            return 0
    if isinstance(value, (tuple, list)):
        return sum(_logical_nbytes(v) for v in value)
    return 0


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], str]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return (), "unknown"
    try:
        return tuple(int(dim) for dim in tuple(shape)), str(dtype)
    except Exception:
        return (), str(dtype)


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, torch.Tensor) or hasattr(value, "shape")


def _input_nodes(node: fx.Node) -> tuple[fx.Node, ...]:
    found: list[fx.Node] = []

    def collect(arg: Any) -> None:
        if isinstance(arg, fx.Node):
            found.append(arg)
        elif isinstance(arg, (tuple, list)):
            for item in arg:
                collect(item)
        elif isinstance(arg, dict):
            for item in arg.values():
                collect(item)

    collect(node.args)
    collect(node.kwargs)
    return tuple(found)


def _output_leaves(node: fx.Node) -> tuple[Any, ...]:
    if node.op != "output":
        return ()
    try:
        from torch.utils import _pytree

        return tuple(_pytree.tree_leaves(node.args[0]))
    except Exception:
        return _input_nodes(node)


def _module_compute_node_names(module: fx.GraphModule | None) -> frozenset[str]:
    if module is None:
        return frozenset()
    return frozenset(
        node.name
        for node in module.graph.nodes
        if node.op not in {"placeholder", "get_attr", "output"}
    )


def _partition_residual_names(capture: CapturedJointGraph) -> frozenset[str]:
    if capture.fw_module is None:
        return frozenset()
    output_nodes = tuple(capture.fw_module.graph.find_nodes(op="output"))
    if not output_nodes:
        return frozenset()
    leaves = tuple(_pytree.tree_leaves(output_nodes[0].args[0]))
    return frozenset(
        leaf.name
        for leaf in leaves[capture.num_fwd_outputs :]
        if isinstance(leaf, fx.Node)
    )


def _classify_phase(
    node: fx.Node,
    *,
    fw_compute_names: frozenset[str] = frozenset(),
    bw_compute_names: frozenset[str] = frozenset(),
) -> str:
    if node.op in {"placeholder", "get_attr"}:
        return "input"
    if node.op == "output":
        return "output"
    if node.name in bw_compute_names and node.name not in fw_compute_names:
        return "bw"
    if node.name in fw_compute_names:
        return "fw"
    text = f"{node.name} {node.target}".lower()
    if "backward" in text or "grad" in text:
        return "bw"
    return "fw"


def build_joint_ir(capture: CapturedJointGraph, passes: tuple[Any, ...] = ()) -> tuple[JointTrainingIR, IRValidationReport]:
    del passes
    gm = capture.joint_module
    nodes = tuple(gm.graph.nodes)
    node_to_op_id = {node: idx for idx, node in enumerate(nodes)}
    node_to_value_id: dict[fx.Node, int] = {}
    raw_values: dict[int, Any] = {}

    for node in nodes:
        value = _node_value(node)
        if _is_tensor_like(value):
            value_id = len(node_to_value_id)
            node_to_value_id[node] = value_id
            raw_values[value_id] = value

    external_value_ids = frozenset(
        value_id
        for node, value_id in node_to_value_id.items()
        if node.op in {"placeholder", "get_attr"}
    )
    storage_groups = build_storage_groups(raw_values, external_value_ids)
    alias_edges: list[tuple[int, int]] = []
    for node in nodes:
        output_value_id = node_to_value_id.get(node)
        if output_value_id is None or not is_alias_preserving_target(node.target):
            continue
        input_value_id = next(
            (
                node_to_value_id[input_node]
                for input_node in _input_nodes(node)
                if input_node in node_to_value_id
            ),
            None,
        )
        if input_value_id is not None:
            alias_edges.append((output_value_id, input_value_id))
    storage_groups = merge_alias_storage_groups(storage_groups, tuple(alias_edges))
    value_to_storage: dict[int, int] = {}
    for storage_id, (value_ids, _, _) in storage_groups.items():
        for value_id in value_ids:
            value_to_storage[value_id] = storage_id

    consumers: dict[int, list[int]] = {value_id: [] for value_id in raw_values}
    for node in nodes:
        op_id = node_to_op_id[node]
        for input_node in _input_nodes(node):
            value_id = node_to_value_id.get(input_node)
            if value_id is not None:
                consumers.setdefault(value_id, []).append(op_id)

    fw_compute_names = _module_compute_node_names(capture.fw_module)
    bw_compute_names = _module_compute_node_names(capture.bw_module)
    residual_names = _partition_residual_names(capture)
    uses_partition_residuals = capture.fw_module is not None
    residual_output_nodes = set()
    if not uses_partition_residuals:
        for node in nodes:
            if node.op == "output":
                leaves = _output_leaves(node)
                residual_output_nodes.update(
                    leaf
                    for leaf in leaves[capture.num_fwd_outputs :]
                    if isinstance(leaf, fx.Node)
                )

    value_infos: list[ValueInfo] = []
    op_infos: list[OpInfo] = []
    for node in nodes:
        op_id = node_to_op_id[node]
        output_ids = (node_to_value_id[node],) if node in node_to_value_id else ()
        input_ids = tuple(
            node_to_value_id[input_node]
            for input_node in _input_nodes(node)
            if input_node in node_to_value_id
        )
        recomputable, illegal_reason = classify_recompute_legality(node)
        phase = _classify_phase(
            node,
            fw_compute_names=fw_compute_names,
            bw_compute_names=bw_compute_names,
        )
        crosses_fw_bw = (
            node.name in residual_names
            if uses_partition_residuals
            else node in residual_output_nodes
        )
        mandatory_reason = mark_mandatory_saves(node, crosses_fw_bw)
        op_infos.append(
            OpInfo(
                id=op_id,
                name=node.name,
                target=str(node.target),
                phase=phase,
                input_value_ids=input_ids,
                output_value_ids=output_ids,
                recomputable=recomputable,
                mandatory_save_reason=mandatory_reason or illegal_reason,
                input_shapes=tuple(_shape_dtype(raw_values[value_id])[0] for value_id in input_ids),
                output_shapes=tuple(_shape_dtype(raw_values[value_id])[0] for value_id in output_ids),
                input_dtypes=tuple(_shape_dtype(raw_values[value_id])[1] for value_id in input_ids),
                output_dtypes=tuple(_shape_dtype(raw_values[value_id])[1] for value_id in output_ids),
            )
        )
        for value_id in output_ids:
            shape, dtype = _shape_dtype(raw_values[value_id])
            value_infos.append(
                ValueInfo(
                    id=value_id,
                    producer_id=op_id if node.op not in {"placeholder", "get_attr"} else None,
                    consumer_ids=tuple(sorted(set(consumers.get(value_id, [])))),
                    storage_id=value_to_storage.get(value_id),
                    logical_nbytes=_logical_nbytes(raw_values[value_id]),
                    phase=phase,
                    crosses_fw_bw=crosses_fw_bw,
                    recomputable=recomputable,
                    mandatory_save_reason=mandatory_reason,
                    name=node.name,
                    shape=shape,
                    dtype=dtype,
                )
            )

    storages = tuple(
        StorageInfo(id=storage_id, value_ids=tuple(value_ids), physical_nbytes=nbytes, is_external=is_external)
        for storage_id, (value_ids, nbytes, is_external) in sorted(storage_groups.items())
    )
    regions = (RegionInfo(id=0, name="whole_graph", op_ids=tuple(op.id for op in op_infos), kind="m0"),)
    ir = JointTrainingIR(
        ops=tuple(op_infos),
        values=tuple(sorted(value_infos, key=lambda v: v.id)),
        storages=storages,
        regions=regions,
        graph_key=capture.capture_key,
    )
    return ir, validate_ir(ir)


def validate_ir(ir: JointTrainingIR) -> IRValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    op_ids = {op.id for op in ir.ops}
    value_ids = {value.id for value in ir.values}
    storage_ids = {storage.id for storage in ir.storages}
    if len(op_ids) != len(ir.ops):
        errors.append("duplicate op ids")
    if len(value_ids) != len(ir.values):
        errors.append("duplicate value ids")
    for value in ir.values:
        if value.producer_id is not None and value.producer_id not in op_ids:
            errors.append(f"value {value.id} has missing producer {value.producer_id}")
        for consumer_id in value.consumer_ids:
            if consumer_id not in op_ids:
                errors.append(f"value {value.id} has missing consumer {consumer_id}")
        if value.storage_id is not None and value.storage_id not in storage_ids:
            errors.append(f"value {value.id} references missing storage {value.storage_id}")
    storage_map = {storage.id: storage.value_ids for storage in ir.storages}
    errors.extend(validate_alias_invariants(storage_map))
    if not ir.values:
        warnings.append("IR contains no tensor-like values")
    return IRValidationReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))
