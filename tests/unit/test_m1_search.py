import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, MeasuredExecutable, RepairHint, TrainingRequest
from peakaware.cost.base import StaticCostProvider
from peakaware.diagnostics import RootCause, diagnose_plan, export_diagnostic_json
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.search.candidates import compute_storage_effect, reject_alias_pinned_gain, select_save_candidates
from peakaware.search.engine import evaluate_plan, search_plans
from peakaware.search.plan import build_recompute_plan


def _ir_and_fixed():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    args = (torch.randn(4, 8),)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
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
    ir, report = build_joint_ir(capture)
    assert report.valid
    return ir, fixed


def test_select_save_candidates_are_storage_aware_and_costed():
    ir, _ = _ir_and_fixed()

    candidates = select_save_candidates(ir, cost_provider=StaticCostProvider())

    assert candidates
    assert candidates == tuple(sorted(candidates, key=lambda c: (-c.score, -c.bytes_at_peak, c.storage_id)))
    assert all(candidate.estimated_recompute_us > 0 for candidate in candidates)
    assert all(0 <= candidate.confidence <= 1 for candidate in candidates)


def test_repair_hint_boosts_candidate_score():
    ir, _ = _ir_and_fixed()
    candidates = select_save_candidates(ir, cost_provider=StaticCostProvider())
    hinted = candidates[-1]

    boosted = select_save_candidates(
        ir,
        cost_provider=StaticCostProvider(),
        hints=(
            RepairHint(
                kind="SAVE_PEAK_STORAGE",
                target_ids=(hinted.storage_id,),
                priority=1.0,
                reason="test",
            ),
        ),
    )
    boosted_by_id = {candidate.storage_id: candidate for candidate in boosted}

    assert boosted_by_id[hinted.storage_id].score == hinted.score * 2.0


def test_alias_pinned_drop_effect_has_no_released_gain():
    ir, _ = _ir_and_fixed()
    mandatory_storage = next(
        value.storage_id
        for value in ir.values
        if value.mandatory_save_reason and value.storage_id is not None
    )

    effect = compute_storage_effect(ir, mandatory_storage, "DROP")

    assert reject_alias_pinned_gain(effect)
    assert effect.pinning_value_ids


def test_m1_search_returns_pareto_ranked_greedy_candidates():
    ir, fixed = _ir_and_fixed()

    evaluated = search_plans(
        ir,
        fixed,
        budget_bytes=1 << 30,
        safety_margin_bytes=0,
        cost_provider=StaticCostProvider(),
        repair_hints=(RepairHint("SAVE_PEAK_STORAGE", (), 0.1, "test"),),
        top_k=5,
    )

    assert evaluated
    assert len(evaluated) <= 8
    assert {plan.plan.plan_id for plan in evaluated[:3]} == {"all_save", "torch_min_cut", "block_checkpoint"}
    assert any(plan.plan.plan_id.startswith("greedy_drop_") for plan in evaluated)
    assert all(0 <= plan.plan.risk_score <= 1 for plan in evaluated)
    assert all(0 <= plan.plan.confidence <= 1 for plan in evaluated)


def test_diagnostics_reports_recompute_wave_hint():
    ir, fixed = _ir_and_fixed()
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    drop_all = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.mandatory_save_reason),
        label="drop_all_legal",
    )
    baseline = evaluate_plan(ir, all_save, fixed)
    candidate = evaluate_plan(ir, drop_all, fixed)

    report = diagnose_plan(baseline, candidate)

    assert report.bw_recompute_transient_change >= 0
    if report.bw_recompute_transient_change > 0:
        assert RootCause.REMATERIALIZATION_WAVE.name in report.root_causes
        assert report.repair_hints
    assert len(report.counterfactuals) == 6
    assert report.counterfactuals[0].level == "D0"
    assert report.counterfactuals[3].level == "D3"
    assert report.counterfactuals[4].level == "D4"
    assert report.counterfactuals[4].status == "unavailable"
    assert report.counterfactuals[4].peak_gain_bytes is None


def test_diagnostics_exports_json_and_marks_runtime_level_available():
    ir, fixed = _ir_and_fixed()
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    baseline = evaluate_plan(ir, all_save, fixed)
    measured = MeasuredExecutable(
        plan_id="all_save",
        forward_backward=lambda x: x,
        measured_peak_bytes=baseline.simulation.estimated_peak_bytes + 128,
        measured_step_us=99.0,
        correctness_passed=True,
    )

    report = diagnose_plan(baseline, baseline, measured=measured)
    text = export_diagnostic_json(report)

    assert '"primary_cause": "UNKNOWN"' in text
    assert report.counterfactuals[-1].level == "D5"
    assert report.counterfactuals[-1].status == "available"
    assert report.counterfactuals[-1].candidate_peak is not None
    assert report.counterfactuals[-1].candidate_peak.live_bytes == measured.measured_peak_bytes
