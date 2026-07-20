from __future__ import annotations

import copy
from typing import Any, Callable

from torch import fx
from torch.utils import _pytree
from torch.utils.checkpoint import CheckpointPolicy

from peakaware.contracts import JointTrainingIR, LoweredPartition, PartitionABI, RecomputePlan
from peakaware.errors import PartitionError


def _build_fw_outputs(plan: RecomputePlan) -> tuple[int, ...]:
    return tuple(sorted(plan.saved_value_ids | plan.mandatory_value_ids))


def _build_bw_inputs(plan: RecomputePlan) -> tuple[int, ...]:
    return _build_fw_outputs(plan)


def make_partition_fn(plan: RecomputePlan, ir: JointTrainingIR) -> Callable[..., tuple[fx.GraphModule, fx.GraphModule]]:
    def partition_fn(joint_module: fx.GraphModule, joint_inputs: Any = None, **kwargs: Any) -> tuple[fx.GraphModule, fx.GraphModule]:
        lowered = partition_joint_graph(joint_module, plan, ir, joint_inputs=joint_inputs, **kwargs)
        return lowered.fw_graph, lowered.bw_graph

    return partition_fn


def _value_name_map(ir: JointTrainingIR) -> dict[int, str]:
    return {value.id: value.name for value in ir.values}


def _graph_output_leaves(joint_module: fx.GraphModule) -> tuple[Any, ...]:
    output_nodes = tuple(joint_module.graph.find_nodes(op="output"))
    if not output_nodes:
        return ()
    return tuple(_pytree.tree_leaves(output_nodes[0].args[0]))


def _apply_saved_value_policy(
    joint_module: fx.GraphModule,
    plan: RecomputePlan,
    ir: JointTrainingIR,
    *,
    num_fwd_outputs: int,
) -> None:
    name_by_value_id = _value_name_map(ir)
    saved_names = {
        name_by_value_id[value_id]
        for value_id in plan.saved_value_ids | plan.mandatory_value_ids
        if value_id in name_by_value_id
    }
    backward_output_names = {
        node.name
        for node in _graph_output_leaves(joint_module)[num_fwd_outputs:]
        if isinstance(node, fx.Node)
    }
    value_by_name = {value.name: value for value in ir.values}
    for node in joint_module.graph.nodes:
        if node.name in backward_output_names or node.meta.get("autograd_backward"):
            continue
        value = value_by_name.get(node.name)
        if value is None or value.phase != "fw" or value.producer_id is None:
            continue
        if node.name in saved_names or not value.recomputable:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        else:
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE


def _partition_with_saved_value_policy(
    joint_module: fx.GraphModule,
    plan: RecomputePlan,
    ir: JointTrainingIR,
    *,
    joint_inputs: Any = None,
    num_fwd_outputs: int = 1,
    static_lifetime_input_indices: tuple[int, ...] = (),
) -> tuple[fx.GraphModule, fx.GraphModule]:
    from torch._functorch.partitioners import default_partition, min_cut_rematerialization_partition

    working = copy.deepcopy(joint_module)
    _apply_saved_value_policy(working, plan, ir, num_fwd_outputs=num_fwd_outputs)
    try:
        return default_partition(
            working,
            joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=list(static_lifetime_input_indices),
        )
    except AssertionError as exc:
        if "was invalid, but is output" not in str(exc):
            raise
        return min_cut_rematerialization_partition(
            copy.deepcopy(joint_module),
            joint_inputs,
            compiler="inductor",
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=list(static_lifetime_input_indices),
        )


def partition_joint_graph(
    joint_module: fx.GraphModule,
    plan: RecomputePlan,
    ir: JointTrainingIR,
    *,
    joint_inputs: Any = None,
    num_fwd_outputs: int = 1,
    static_lifetime_input_indices: tuple[int, ...] = (),
) -> LoweredPartition:
    fw_graph, bw_graph = _partition_with_saved_value_policy(
        joint_module,
        plan,
        ir,
        joint_inputs=joint_inputs,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_indices=static_lifetime_input_indices,
    )
    return lower_partition_graphs(joint_module, fw_graph, bw_graph, plan, ir)


def partition_default_graph(joint_module: fx.GraphModule, plan: RecomputePlan, ir: JointTrainingIR) -> LoweredPartition:
    return lower_partition_graphs(joint_module, joint_module, joint_module, plan, ir)


def lower_partition_graphs(
    joint_module: fx.GraphModule,
    fw_graph: fx.GraphModule | None,
    bw_graph: fx.GraphModule | None,
    plan: RecomputePlan,
    ir: JointTrainingIR,
) -> LoweredPartition:
    if fw_graph is None and bw_graph is None:
        raise PartitionError("lower_partition_graphs requires at least one FW or BW graph")
    abi = PartitionABI(
        fw_output_value_ids=_build_fw_outputs(plan),
        bw_placeholder_value_ids=_build_bw_inputs(plan),
        tangent_value_ids=(),
        rng_state_value_ids=tuple(
            value.id
            for value in ir.values
            if value.mandatory_save_reason == "requires_rng_preservation"
        ),
    )
    del joint_module
    return LoweredPartition(
        plan_id=plan.plan_id,
        fw_graph=fw_graph if fw_graph is not None else bw_graph,
        bw_graph=bw_graph if bw_graph is not None else fw_graph,
        partition_abi=abi,
    )
