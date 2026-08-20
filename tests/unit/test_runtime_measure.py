import pytest
import torch
from torch import nn
from torch.utils import _pytree

from peakaware.runtime.measure import (
    _clone_grads,
    _clone_model_state,
    _clone_optimizer_state,
    measure_training_step_phases,
)


def test_phase_measurement_repeats_restore_state_and_report_reserved_metrics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    args = (torch.randn(2, 4),)
    before = tuple(param.detach().clone() for param in model.parameters())

    metrics = measure_training_step_phases(
        model,
        optimizer,
        model,
        lambda out: out.pow(2).mean(),
        args,
        {},
        zero_grad_set_to_none=True,
        warmup_steps=1,
        repeat_count=2,
    )

    after = tuple(param.detach().clone() for param in model.parameters())
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert metrics["measurement_warmup_steps"] == 1
    assert metrics["measurement_repeats"] == 2
    assert metrics["step_us_median"] > 0
    assert metrics["step_us_p10"] > 0
    assert metrics["step_us_p90"] > 0
    assert "overall_reserved_peak_bytes" in metrics
    assert "after_fw_allocated_bytes" in metrics
    assert "after_fw_reserved_bytes" in metrics
    assert metrics["actual_memory_trace_kind"] == "phase_boundary_anchor"
    assert metrics["actual_memory_trace_sampled"] is False
    assert [point["phase"] for point in metrics["actual_memory_trace"]] == [
        "start",
        "fw_peak",
        "after_fw",
        "bw_peak",
        "optimizer_peak",
        "overall_peak",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_measurement_restore_snapshots_do_not_remain_in_cuda_allocator():
    model = nn.Linear(8, 8).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.randn(2, 8, device="cuda")).square().mean()
    loss.backward()
    optimizer.step()

    model_state = _clone_model_state(model)
    grad_state = _clone_grads(model)
    optimizer_state = _clone_optimizer_state(optimizer)
    optimizer_leaves, _ = _pytree.tree_flatten(optimizer_state)

    assert model_state
    assert all(tensor.device.type == "cpu" for tensor in model_state.values())
    assert all(grad is None or grad.device.type == "cpu" for grad in grad_state)
    assert all(
        leaf.device.type == "cpu"
        for leaf in optimizer_leaves
        if isinstance(leaf, torch.Tensor)
    )
