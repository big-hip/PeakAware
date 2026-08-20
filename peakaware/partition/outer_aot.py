from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd

from peakaware.contracts import LoweredPartition, PartitionABI


class _OuterAOTPartitionCaptured(RuntimeError):
    pass


@dataclass
class _CaptureState:
    fw_graph: fx.GraphModule | None = None
    bw_graph: fx.GraphModule | None = None


def capture_outer_aot_partition(
    executable: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    plan_id: str,
) -> LoweredPartition:
    """Capture the outer AOTAutograd FW/BW graphs without running them.

    Dynamo/AOTAutograd traces with fake tensors before executing the compiled
    callable.  The partition callback records the compiler-realized graphs and
    raises a private sentinel, so no candidate kernel is launched by this call.
    """

    from torch._functorch.partitioners import min_cut_rematerialization_partition

    state = _CaptureState()

    def partition_fn(
        joint_module: fx.GraphModule,
        joint_inputs: Any = None,
        **partition_kwargs: Any,
    ) -> tuple[fx.GraphModule, fx.GraphModule]:
        # Both the public aot_eager backend and Inductor's training path use
        # min-cut rematerialization.  Capturing default_partition here would
        # refine a graph that is different from the runtime compiler graph.
        fw_graph, bw_graph = min_cut_rematerialization_partition(
            joint_module,
            joint_inputs,
            compiler="inductor",
            **partition_kwargs,
        )
        state.fw_graph = fw_graph
        state.bw_graph = bw_graph
        raise _OuterAOTPartitionCaptured(plan_id)

    backend = aot_autograd(
        fw_compiler=lambda gm, _inputs: gm.forward,
        bw_compiler=lambda gm, _inputs: gm.forward,
        partition_fn=partition_fn,
    )
    compiled = torch.compile(executable, backend=backend)
    try:
        compiled(*args, **kwargs)
    except Exception as exc:
        current: BaseException | None = exc
        while current is not None and not isinstance(
            current, _OuterAOTPartitionCaptured
        ):
            current = current.__cause__ or current.__context__
        if current is None:
            raise
    finally:
        torch.compiler.reset()
    if state.fw_graph is None or state.bw_graph is None:
        raise RuntimeError("outer AOT partition callback was not reached")
    fw_output_nodes = tuple(state.fw_graph.graph.find_nodes(op="output"))
    fw_output_count = 0
    if fw_output_nodes:
        from torch.utils import _pytree

        fw_output_count = len(_pytree.tree_leaves(fw_output_nodes[0].args[0]))
    bw_placeholder_count = len(tuple(state.bw_graph.graph.find_nodes(op="placeholder")))
    residual_count = min(fw_output_count, bw_placeholder_count)
    return LoweredPartition(
        plan_id=plan_id,
        fw_graph=state.fw_graph,
        bw_graph=state.bw_graph,
        partition_abi=PartitionABI(
            fw_output_value_ids=tuple(range(residual_count)),
            bw_placeholder_value_ids=tuple(range(residual_count)),
            tangent_value_ids=(),
            rng_state_value_ids=(),
        ),
    )
