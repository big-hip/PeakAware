import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec


def _request(model, args, optimizer):
    return TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(),
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
