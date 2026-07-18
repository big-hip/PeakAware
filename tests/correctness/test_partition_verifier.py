import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.partition.aot import partition_joint_graph
from peakaware.partition.verifier import run_aot_eager_dry_run
from peakaware.search.plan import build_recompute_plan


def test_dry_run_compares_loss_and_gradients():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(p=0.1), nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 3),)
    request = TrainingRequest(
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
    capture = capture_joint_graph(request)
    ir, _ = build_joint_ir(capture)
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(capture.joint_module, plan, ir)

    result = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={},
        loss_fn=lambda out: out.sum(),
        atol=1e-5,
        rtol=1e-4,
    )

    assert result.abi_valid
    assert result.outputs_match
    assert result.gradients_match
