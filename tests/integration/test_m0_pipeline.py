import torch
from torch import nn

from peakaware import PeakAwareConfig, optimize_training


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        residual = self.a(x).relu()
        hidden = self.b(residual).relu()
        return self.out(hidden + residual)


def test_optimize_training_builds_executor_and_runs_step():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    result = optimize_training(
        model,
        (x,),
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )

    before = tuple(p.detach().clone() for p in model.parameters())
    step = result.executor.step(x)
    after = tuple(p.detach().clone() for p in model.parameters())

    assert step.optimizer_step_performed is True
    assert step.loss.ndim == 0
    assert result.dry_run is not None and result.dry_run.gradients_match
    assert result.analysis is not None and result.analysis.ir.values
    assert result.analysis is not None and result.analysis.ir.graph_key == result.selected_plan.graph_key
    assert any(not torch.equal(left, right) for left, right in zip(before, after))


def test_optimize_training_does_not_advance_user_state_before_executor_step():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    before = tuple(p.detach().clone() for p in model.parameters())

    result = optimize_training(
        model,
        (x,),
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )
    after_optimize = tuple(p.detach().clone() for p in model.parameters())

    assert result.executable.correctness_passed
    assert all(torch.equal(left, right) for left, right in zip(before, after_optimize))
