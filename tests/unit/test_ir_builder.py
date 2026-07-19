import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.partition.aot import lower_partition_graphs
from peakaware.search.plan import build_recompute_plan


def _request(model, args, optimizer, config=None):
    return TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=config or PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )


def test_build_joint_ir_has_values_storages_and_mandatory_boundary():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    args = (torch.randn(2, 4),)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    capture = capture_joint_graph(_request(model, args, optimizer))
    ir, report = build_joint_ir(capture)

    assert report.valid
    assert ir.graph_key == capture.capture_key
    assert ir.values
    assert ir.storages
    assert len({s.id for s in ir.storages}) == len(ir.storages)
    assert any(value.phase == "fw" for value in ir.values)


def test_graph_key_includes_static_input_shape_guards():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    small = capture_joint_graph(
        _request(model, (torch.randn(1, 4),), optimizer, PeakAwareConfig(capture_backend="fx")),
    )
    large = capture_joint_graph(
        _request(model, (torch.randn(3, 4),), optimizer, PeakAwareConfig(capture_backend="fx")),
    )
    small_ir, small_report = build_joint_ir(small)
    large_ir, large_report = build_joint_ir(large)

    assert small_report.valid
    assert large_report.valid
    assert small.capture_key != large.capture_key
    assert small_ir.graph_key != large_ir.graph_key
    assert ("arg0.shape", "(1, 4)") in {(guard.name, guard.value) for guard in small.guards}
    assert ("arg0.shape", "(3, 4)") in {(guard.name, guard.value) for guard in large.guards}


def test_aot_capture_intercepts_joint_fw_and_bw_graphs():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    args = (torch.randn(2, 4),)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )

    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)

    assert capture.backend == "aot"
    assert capture.fw_module is not None
    assert capture.bw_module is not None
    assert len(list(capture.joint_module.graph.nodes)) > len(list(capture.fw_module.graph.nodes))
    assert report.valid
    assert ir.values

    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = lower_partition_graphs(capture.joint_module, capture.fw_module, capture.bw_module, plan, ir)
    assert lowered.fw_graph is capture.fw_module
    assert lowered.bw_graph is capture.bw_module
