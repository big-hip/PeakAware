import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.search.engine import search_plans


def test_search_plans_returns_ranked_evaluated_m0_plans():
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
        config=PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    fixed, _ = analyze_coarse_feasibility(model, optimizer, 1 << 30)
    capture = capture_joint_graph(request)
    ir, _ = build_joint_ir(capture)

    evaluated = search_plans(ir, fixed, budget_bytes=1 << 30, safety_margin_bytes=0, top_k=3)

    assert len(evaluated) == 3
    assert evaluated[0].feasible
    assert evaluated[0].plan.estimated_peak_bytes == evaluated[0].simulation.estimated_peak_bytes
    plan_ids = {plan.plan.plan_id for plan in evaluated}
    assert "all_save" in plan_ids
    assert plan_ids <= {"all_save", "mandatory_only", "manual_alternating", "greedy_drop_0", "greedy_drop_1"}
