import torch
from torch import nn

from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec


def test_adamw_fixed_memory_estimate_before_state_materialization():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    spec = build_optimizer_spec(optimizer, model)
    fixed, report = analyze_coarse_feasibility(model, optimizer, 1 << 30)

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert spec.name == "AdamW"
    assert spec.state_bytes == 2 * parameter_bytes
    assert fixed.optimizer_temporary_bytes == parameter_bytes
    assert report.status == "FEASIBLE"


def test_sgd_fixed_memory_has_no_optimizer_state_before_materialization():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    spec = build_optimizer_spec(optimizer, model)
    fixed, report = analyze_coarse_feasibility(model, optimizer, 1 << 30)

    assert spec.name == "SGD"
    assert spec.state_bytes == 0
    assert spec.temporary_bytes == 0
    assert fixed.optimizer_state_bytes == 0
    assert fixed.optimizer_temporary_bytes == 0
    assert report.status == "FEASIBLE"


def test_fused_adamw_mode_is_part_of_optimizer_spec_and_memory_estimate():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    spec = build_optimizer_spec(optimizer, model)
    fixed, report = analyze_coarse_feasibility(model, optimizer, 1 << 30)

    assert spec.name == "AdamW[fused]"
    assert spec.state_bytes == 2 * parameter_bytes
    assert spec.temporary_bytes == parameter_bytes
    assert fixed.optimizer_temporary_bytes == parameter_bytes
    assert report.status == "FEASIBLE"


def test_materialized_optimizer_state_is_counted_exactly():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    spec = build_optimizer_spec(optimizer, model)
    state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )

    assert spec.name == "AdamW"
    assert spec.state_bytes == state_bytes
    assert spec.temporary_bytes == sum(p.numel() * p.element_size() for p in model.parameters())


def test_unknown_optimizer_without_state_uses_nonzero_temporary_estimate():
    class ToyOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {"lr": 0.1})

        def step(self, closure=None):
            del closure

    model = nn.Linear(4, 3)
    optimizer = ToyOptimizer(model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    spec = build_optimizer_spec(optimizer, model)

    assert spec.name == "ToyOptimizer"
    assert spec.state_bytes == 0
    assert spec.temporary_bytes == parameter_bytes


def test_coarse_feasibility_does_not_reject_on_temporary_upper_bound_only():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fixed = analyze_coarse_feasibility(model, optimizer, memory_budget_bytes=1 << 30)[0]
    budget = fixed.steady_bytes + max(fixed.optimizer_temporary_bytes // 2, 1)

    fixed, report = analyze_coarse_feasibility(model, optimizer, memory_budget_bytes=budget)

    assert fixed.steady_bytes <= report.user_budget_bytes
    assert fixed.peak_lower_bound_bytes > report.user_budget_bytes
    assert report.status == "LOW_ACTIVATION_HEADROOM"
