import pytest
import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.errors import PlanValidationError
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.search.exact import solve_exact_small_graph
from peakaware.search.engine import search_plans


def _small_ir():
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1))
    args = (torch.randn(2, 3),)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="fx"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    fixed, _ = analyze_coarse_feasibility(model, optimizer, 1 << 30)
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    return ir, fixed


def test_exact_solver_returns_no_worse_peak_than_search_topk():
    ir, fixed = _small_ir()

    exact = solve_exact_small_graph(ir, fixed, budget_bytes=1 << 30, safety_margin_bytes=0)
    searched = search_plans(ir, fixed, budget_bytes=1 << 30, safety_margin_bytes=0, top_k=5)

    assert exact.feasible
    assert exact.simulation.estimated_peak_bytes <= min(plan.simulation.estimated_peak_bytes for plan in searched)


def test_exact_solver_fails_closed_for_large_candidate_sets():
    ir, fixed = _small_ir()

    with pytest.raises(PlanValidationError):
        solve_exact_small_graph(ir, fixed, budget_bytes=1 << 30, max_candidate_count=0)
