from __future__ import annotations

from typing import Any, Callable

from torch import fx

from peakaware.contracts import JointTrainingIR, LoweredPartition, PartitionABI, RecomputePlan


def _build_fw_outputs(plan: RecomputePlan) -> tuple[int, ...]:
    return tuple(sorted(plan.saved_value_ids | plan.mandatory_value_ids))


def _build_bw_inputs(plan: RecomputePlan) -> tuple[int, ...]:
    return _build_fw_outputs(plan)


def make_partition_fn(plan: RecomputePlan, ir: JointTrainingIR) -> Callable[..., tuple[fx.GraphModule, fx.GraphModule]]:
    def partition_fn(joint_module: fx.GraphModule, _joint_inputs: Any = None, **_kwargs: Any) -> tuple[fx.GraphModule, fx.GraphModule]:
        lowered = partition_joint_graph(joint_module, plan, ir)
        return lowered.fw_graph, lowered.bw_graph

    return partition_fn


def partition_joint_graph(joint_module: fx.GraphModule, plan: RecomputePlan, ir: JointTrainingIR) -> LoweredPartition:
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
    return LoweredPartition(
        plan_id=plan.plan_id,
        fw_graph=joint_module,
        bw_graph=joint_module,
        partition_abi=abi,
    )
