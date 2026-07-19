import torch
from torch import nn

from peakaware.runtime.measure import measure_training_step_phases


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
