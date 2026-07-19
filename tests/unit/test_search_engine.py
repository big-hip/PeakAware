import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.search.engine import search_plans, search_plans_with_diagnostics
from peakaware.search.plan import build_recompute_plan, plan_identity_key


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

    assert len(evaluated) == 6
    assert evaluated[0].feasible
    assert evaluated[0].plan.estimated_peak_bytes == evaluated[0].simulation.estimated_peak_bytes
    plan_ids = {plan.plan.plan_id for plan in evaluated}
    assert {"all_save", "torch_min_cut", "block_checkpoint"}.issubset(plan_ids)
    provenance_by_plan = {plan.plan.plan_id: plan.plan.strategy_expectation_provenance for plan in evaluated}
    assert provenance_by_plan["all_save"]["source"] == "all_save_baseline"
    assert provenance_by_plan["torch_min_cut"]["source"] == "pytorch_min_cut_proxy"
    assert provenance_by_plan["block_checkpoint"]["source"] == "block_checkpoint_proxy"
    assert plan_ids <= {
        "all_save",
        "torch_min_cut",
        "block_checkpoint",
        "greedy_drop_0",
        "greedy_drop_1",
        "greedy_drop_2",
        "greedy_drop_3",
    }


def test_search_keeps_baselines_when_extra_candidates_are_topk_limited():
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

    evaluated = search_plans(ir, fixed, budget_bytes=1 << 30, safety_margin_bytes=0, top_k=1)
    plan_ids = tuple(plan.plan.plan_id for plan in evaluated)

    assert plan_ids[:3] == ("all_save", "torch_min_cut", "block_checkpoint")
    assert len(evaluated) == 4


def test_search_reports_diagnostic_hint_trace():
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

    evaluated, diagnostics = search_plans_with_diagnostics(
        ir,
        fixed,
        budget_bytes=1 << 30,
        safety_margin_bytes=0,
        repair_hints=(),
        top_k=3,
    )
    _, disabled = search_plans_with_diagnostics(
        ir,
        fixed,
        budget_bytes=1 << 30,
        safety_margin_bytes=0,
        enable_diagnostic_hints=False,
        top_k=3,
    )

    assert evaluated
    assert diagnostics.diagnostic_hints_enabled is True
    assert diagnostics.greedy_plan_count >= 1
    assert diagnostics.feasible_after_repair_count >= diagnostics.feasible_before_repair_count
    assert disabled.diagnostic_hints_enabled is False
    assert disabled.diagnostic_hint_count == 0
    assert disabled.diagnostic_hint_kinds == ()


def test_plan_identity_key_is_determined_by_save_set_not_label():
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
    capture = capture_joint_graph(request)
    ir, _ = build_joint_ir(capture)
    saved = frozenset(v.id for v in ir.values if v.phase == "fw")

    readable = build_recompute_plan(ir, budget_bytes=1 << 30, saved_value_ids=saved, label="all_save")
    relabeled = build_recompute_plan(ir, budget_bytes=1 << 30, saved_value_ids=saved, label="baseline_alias")

    assert readable.plan_id != relabeled.plan_id
    readable_key = plan_identity_key(
        readable.graph_key,
        readable.saved_value_ids | readable.mandatory_value_ids,
        readable.budget_bytes,
    )
    relabeled_key = plan_identity_key(
        relabeled.graph_key,
        relabeled.saved_value_ids | relabeled.mandatory_value_ids,
        relabeled.budget_bytes,
    )
    assert readable_key == relabeled_key
