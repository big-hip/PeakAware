from __future__ import annotations

import operator

import torch
from torch import fx

from peakaware.contracts import FixedTimeline, LoweredPartition, PartitionABI
from peakaware.memory.fx_timeline import (
    simulate_lowered_fx_l2_event_trace,
    summarize_lowered_fx_l2_event_trace,
)


def _lowered_view_partition() -> LoweredPartition:
    fw_graph = fx.Graph()
    fw_input = fw_graph.placeholder("x")
    cloned = fw_graph.call_function(torch.ops.aten.clone.default, (fw_input,))
    viewed = fw_graph.call_function(torch.ops.aten.view.default, (cloned, (2, 2)))
    fw_graph.output((viewed, viewed))
    fw_module = fx.GraphModule({}, fw_graph)

    input_tensor = torch.empty(4)
    cloned_tensor = torch.empty(4)
    fw_input.meta["val"] = input_tensor
    cloned.meta["val"] = cloned_tensor
    viewed.meta["val"] = cloned_tensor.view(2, 2)

    bw_graph = fx.Graph()
    saved = bw_graph.placeholder("saved")
    tangent = bw_graph.placeholder("tangent")
    added = bw_graph.call_function(operator.add, (saved, tangent))
    bw_graph.output((added,))
    bw_module = fx.GraphModule({}, bw_graph)
    saved.meta["val"] = torch.empty(2, 2)
    tangent.meta["val"] = torch.empty(2, 2)
    added.meta["val"] = torch.empty(2, 2)

    return LoweredPartition(
        plan_id="view-pinning",
        fw_graph=fw_module,
        bw_graph=bw_module,
        partition_abi=PartitionABI(
            # The synthetic plan ABI may over-count logical saved values.  The
            # lowered graph's named tangent boundary is the physical source.
            fw_output_value_ids=(1, 2, 3),
            bw_placeholder_value_ids=(1, 2, 3),
            tangent_value_ids=(),
            rng_state_value_ids=(),
        ),
    )


def _lowered_fake_slice_alias_partition() -> LoweredPartition:
    fw_graph = fx.Graph()
    fw_input = fw_graph.placeholder("x")
    cloned = fw_graph.call_function(torch.ops.aten.clone.default, (fw_input,))
    sliced = fw_graph.call_function(
        torch.ops.aten.slice.Tensor,
        (cloned, 0, 0, 4),
    )
    fw_graph.output((sliced, sliced))
    fw_module = fx.GraphModule({}, fw_graph)

    fw_input.meta["val"] = torch.empty(8)
    cloned.meta["val"] = torch.empty(8)
    # Reproduce FakeTensor metadata that loses the physical alias relation.
    sliced.meta["val"] = torch.empty(4)

    bw_graph = fx.Graph()
    saved = bw_graph.placeholder("saved")
    tangent = bw_graph.placeholder("tangent")
    added = bw_graph.call_function(operator.add, (saved, tangent))
    bw_graph.output((added,))
    bw_module = fx.GraphModule({}, bw_graph)
    saved.meta["val"] = torch.empty(4)
    tangent.meta["val"] = torch.empty(4)
    added.meta["val"] = torch.empty(4)

    return LoweredPartition(
        plan_id="fake-slice-alias",
        fw_graph=fw_module,
        bw_graph=bw_module,
        partition_abi=PartitionABI(
            fw_output_value_ids=(1,),
            bw_placeholder_value_ids=(1,),
            tangent_value_ids=(),
            rng_state_value_ids=(),
        ),
    )


def test_lowered_fx_timeline_pins_view_base_and_releases_saved_bw_input() -> None:
    trace = simulate_lowered_fx_l2_event_trace(
        _lowered_view_partition(),
        FixedTimeline(
            parameter_bytes=100,
            buffer_bytes=0,
            gradient_bytes=100,
            optimizer_state_bytes=200,
            optimizer_temporary_bytes=0,
        ),
        align=1,
    )

    after_fw = next(row for row in trace if row["phase"] == "after_fw")
    bw_rows = [row for row in trace if row["phase"] == "bw"]

    assert after_fw["fixed_bytes"] == 300
    assert after_fw["payload_bytes"] == 16
    assert after_fw["bytes"] == 316
    assert max(row["payload_bytes"] for row in bw_rows) == 48
    assert bw_rows[-1]["payload_bytes"] == 48

    summary = summarize_lowered_fx_l2_event_trace(trace)
    assert summary["estimated_peak_bytes"] == 400
    assert summary["peak_phase"] == "optimizer"
    assert summary["phase_peak_bytes"] == {
        "fw": 316,
        "after_fw": 316,
        "bw": 348,
        "optimizer": 400,
    }


def test_lowered_fx_timeline_recovers_fake_slice_alias_and_base_size() -> None:
    trace = simulate_lowered_fx_l2_event_trace(
        _lowered_fake_slice_alias_partition(),
        FixedTimeline(
            parameter_bytes=0,
            buffer_bytes=0,
            gradient_bytes=0,
            optimizer_state_bytes=0,
            optimizer_temporary_bytes=0,
        ),
        align=1,
    )

    after_fw = next(row for row in trace if row["phase"] == "after_fw")
    assert after_fw["payload_bytes"] == 32
