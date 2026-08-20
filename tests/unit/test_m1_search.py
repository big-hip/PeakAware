from dataclasses import replace

import pytest
import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import (
    FixedTimeline,
    HardwareSpec,
    JointTrainingIR,
    MeasuredExecutable,
    OpInfo,
    RepairHint,
    StorageInfo,
    TrainingRequest,
    ValueInfo,
)
from peakaware.cost.base import OpCost, StaticCostProvider
from peakaware.diagnostics import (
    RootCause,
    RootCauseGroundTruth,
    RootCausePrediction,
    diagnose_plan,
    evaluate_root_cause_predictions,
    export_diagnostic_json,
)
from peakaware.ir.alias import is_alias_preserving_target
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.memory.simulator import build_simulation_cost_cache
from peakaware.search.candidates import compute_storage_effect, reject_alias_pinned_gain, select_save_candidates
from peakaware.search.closure import derive_recompute_closure
from peakaware.search.engine import apply_early_stop_policy, evaluate_plan, search_plans
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
    value_by_id = {value.id: value for value in ir.values}
    assert all(
        any(value_by_id[value_id].crosses_fw_bw for value_id in candidate.value_ids)
        for candidate in candidates
    )


def test_aot_ir_uses_partition_residuals_and_backward_nodes():
    ir, _ = _ir_and_fixed()

    crossing = tuple(value for value in ir.values if value.crosses_fw_bw)

    assert crossing
    assert all(value.phase in {"input", "fw"} for value in crossing)
    assert any(op.phase == "bw" and "backward" not in op.name for op in ir.ops)
    gradient_output_ids = {
        value.id
        for value in ir.values
        if value.consumer_ids and any(op.phase == "output" for op in ir.ops if op.id in value.consumer_ids)
    }
    assert any(value.id not in gradient_output_ids for value in crossing)


def test_aot_ir_merges_alias_preserving_outputs_with_tensor_inputs() -> None:
    ir, _ = _ir_and_fixed()
    value_by_id = {value.id: value for value in ir.values}

    checked = 0
    for op in ir.ops:
        if not is_alias_preserving_target(op.target):
            continue
        input_storage_ids = {
            value_by_id[value_id].storage_id
            for value_id in op.input_value_ids
            if value_id in value_by_id
            and value_by_id[value_id].storage_id is not None
        }
        if not input_storage_ids:
            continue
        for value_id in op.output_value_ids:
            output = value_by_id[value_id]
            assert output.storage_id in input_storage_ids
            checked += 1
    assert checked > 0


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
    mandatory_value = next(
        value
        for value in ir.values
        if value.crosses_fw_bw and value.storage_id is not None
    )
    ir = replace(
        ir,
        values=tuple(
            replace(value, mandatory_save_reason="test_pin")
            if value.id == mandatory_value.id
            else value
            for value in ir.values
        ),
    )
    mandatory_storage = mandatory_value.storage_id

    effect = compute_storage_effect(ir, mandatory_storage, "DROP")

    assert reject_alias_pinned_gain(effect)
    assert effect.pinning_value_ids


class _WorkspaceCostProvider:
    source = "workspace_test"
    cache_safe = True

    def __init__(self, *, workspace_bytes: int, estimated_us: float = 7.0) -> None:
        self.workspace_bytes = workspace_bytes
        self.estimated_us = estimated_us
        self.estimate_calls = 0

    def supports(self, signature):
        del signature
        return True

    def estimate(self, signature):
        del signature
        self.estimate_calls += 1
        return OpCost(
            estimated_us=self.estimated_us,
            memory_bytes=self.workspace_bytes,
            source=self.source,
            confidence=0.8,
        )


def _linear_recompute_ir() -> JointTrainingIR:
    return JointTrainingIR(
        ops=(
            OpInfo(0, "fw0", "aten.fw0", "fw", (), (0,), True, None),
            OpInfo(1, "fw1", "aten.fw1", "fw", (0,), (1,), True, None),
            OpInfo(2, "fw2", "aten.fw2", "fw", (1,), (2,), True, None),
        ),
        values=(
            ValueInfo(0, 0, (1,), 0, 100, "fw", True, True, None, "v0"),
            ValueInfo(1, 1, (2,), 1, 100, "fw", True, True, None, "v1"),
            ValueInfo(2, 2, (), 2, 100, "fw", True, True, None, "v2"),
        ),
        storages=(
            StorageInfo(0, (0,), 100, False),
            StorageInfo(1, (1,), 100, False),
            StorageInfo(2, (2,), 100, False),
        ),
        regions=(),
        graph_key="linear-recompute",
    )


def _residual_dependency_ir() -> JointTrainingIR:
    return JointTrainingIR(
        ops=(
            OpInfo(0, "fw0", "aten.fw0", "fw", (0,), (1,), True, None),
            OpInfo(1, "fw1", "aten.fw1", "fw", (1,), (2,), True, None),
            OpInfo(2, "fw2", "aten.fw2", "fw", (2,), (3,), True, None),
        ),
        values=(
            ValueInfo(0, None, (0,), 0, 100, "input", False, False, None, "input"),
            ValueInfo(1, 0, (1,), 1, 100, "fw", False, True, None, "internal0"),
            ValueInfo(2, 1, (2,), 2, 100, "fw", False, True, None, "internal1"),
            ValueInfo(3, 2, (), 3, 100, "fw", True, True, None, "residual"),
        ),
        storages=(
            StorageInfo(0, (0,), 100, True),
            StorageInfo(1, (1,), 100, False),
            StorageInfo(2, (2,), 100, False),
            StorageInfo(3, (3,), 100, False),
        ),
        regions=(),
        graph_key="residual-dependency",
    )


def test_recompute_closure_traces_only_dropped_residual_ancestors():
    ir = _residual_dependency_ir()

    all_save = derive_recompute_closure(ir, frozenset({3}))
    drop_residual = derive_recompute_closure(ir, frozenset())

    assert all_save.recomputed_op_ids == frozenset()
    assert drop_residual.recomputed_op_ids == frozenset({0, 1, 2})
    assert drop_residual.recomputed_value_ids == frozenset({1, 2, 3})
    assert drop_residual.barrier_value_ids == frozenset()


def test_simulator_uses_recompute_live_range_instead_of_total_dropped_bytes():
    ir = _linear_recompute_ir()
    fixed = FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    plan = build_recompute_plan(ir, budget_bytes=1 << 30, saved_value_ids=frozenset(), label="drop_all")

    evaluated = evaluate_plan(ir, plan, fixed)

    assert evaluated.simulation.max_recompute_live_bytes == 200
    assert evaluated.simulation.max_recompute_live_bytes < sum(storage.physical_nbytes for storage in ir.storages)
    assert evaluated.simulation.recompute_span_ops == 3
    assert evaluated.simulation.recompute_before_first_bw_op_bytes == 200


def test_recompute_event_trace_contains_the_simulated_peak() -> None:
    ir = _linear_recompute_ir()
    fixed = FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(),
        label="drop_all",
    )

    evaluated = evaluate_plan(ir, plan, fixed)
    trace = evaluated.simulation.simulated_memory_event_trace

    assert any(
        row["event"] == "phase_peak_bound" and row["recomputed_bytes"] > 0
        for row in trace
    )
    assert max(row["bytes"] for row in trace) == evaluated.simulation.estimated_peak_bytes


def test_simulator_uses_cost_provider_workspace_and_latency():
    ir = _linear_recompute_ir()
    fixed = FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    plan = build_recompute_plan(ir, budget_bytes=1 << 30, saved_value_ids=frozenset(), label="drop_all")
    baseline = evaluate_plan(ir, plan, fixed)

    calibrated = evaluate_plan(
        ir,
        plan,
        fixed,
        cost_provider=_WorkspaceCostProvider(workspace_bytes=64, estimated_us=13.0),
    )

    assert calibrated.simulation.fw_peak_bytes == baseline.simulation.fw_peak_bytes + 64
    assert calibrated.simulation.bw_peak_bytes == baseline.simulation.bw_peak_bytes + 64
    assert calibrated.simulation.estimated_step_us == len(ir.ops) * 10 + len(ir.ops) * 13.0
    assert calibrated.simulation.confidence == 0.75
    assert calibrated.plan.cost_sources == ("m0_static", "workspace_test")


def test_simulation_cost_cache_preserves_results_and_reuses_provider_queries():
    ir = _linear_recompute_ir()
    fixed = FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(),
        label="drop_all",
    )
    uncached_provider = _WorkspaceCostProvider(workspace_bytes=64, estimated_us=13.0)
    uncached = evaluate_plan(ir, plan, fixed, cost_provider=uncached_provider)
    cached_provider = _WorkspaceCostProvider(workspace_bytes=64, estimated_us=13.0)
    cache = build_simulation_cost_cache(ir, fixed, cached_provider)
    calls_after_build = cached_provider.estimate_calls
    cached = evaluate_plan(
        ir,
        plan,
        fixed,
        cost_provider=cached_provider,
        simulation_cost_cache=cache,
    )

    assert cached == uncached
    assert calls_after_build < uncached_provider.estimate_calls
    assert cached_provider.estimate_calls == calls_after_build


def test_simulation_cost_cache_rejects_a_different_provider_instance():
    ir = _linear_recompute_ir()
    fixed = FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(),
        label="drop_all",
    )
    cached_provider = _WorkspaceCostProvider(workspace_bytes=64, estimated_us=13.0)
    stale_cache = build_simulation_cost_cache(ir, fixed, cached_provider)
    current_provider = _WorkspaceCostProvider(workspace_bytes=128, estimated_us=29.0)

    with pytest.raises(ValueError, match="different cost provider"):
        evaluate_plan(
            ir,
            plan,
            fixed,
            cost_provider=current_provider,
            simulation_cost_cache=stale_cache,
        )


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


def test_early_stop_reports_empty_search_evidence():
    report = apply_early_stop_policy(())

    assert report is not None
    assert report.reason == "no candidate plans were generated"
    assert report.best_plan_id is None
    assert report.evidence.evaluated_plan_count == 0
    assert report.evidence.feasible_plan_count == 0


def test_early_stop_reports_no_feasible_best_so_far():
    ir, fixed = _ir_and_fixed()
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    mandatory = build_recompute_plan(
        ir,
        budget_bytes=1,
        saved_value_ids=frozenset(v.id for v in ir.values if v.mandatory_save_reason),
        label="torch_min_cut",
    )

    evaluated = (evaluate_plan(ir, all_save, fixed), evaluate_plan(ir, mandatory, fixed))
    report = apply_early_stop_policy(evaluated, fixed_timeline=fixed)

    assert report is not None
    assert report.reason == "no feasible plan under search budget"
    assert report.best_plan_id in {"all_save", "torch_min_cut"}
    assert report.evidence.evaluated_plan_count == 2
    assert report.evidence.feasible_plan_count == 0
    assert report.evidence.best_estimated_peak_bytes is not None
    assert report.evidence.fixed_peak_lower_bound_bytes == fixed.peak_lower_bound_bytes


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
    assert report.strategy_expectation_status == "unavailable"
    assert report.strategy_expected_saved_reduction is None
    assert report.normalized_saved_reduction == report.expected_saved_reduction
    assert report.strategy_estimation_gap is None
    assert report.realization_gap == report.normalized_saved_reduction - report.actual_overall_peak_reduction
    assert 0 <= report.confidence <= 1
    assert any(item.metric == "normalized_saved_reduction_bytes" for item in report.evidence)
    assert any(item.metric == "estimated_overall_peak_reduction_bytes" for item in report.evidence)
    if report.bw_recompute_transient_change > 0:
        assert RootCause.REMATERIALIZATION_WAVE.name in report.root_causes
        assert any(item.root_cause == RootCause.REMATERIALIZATION_WAVE.name for item in report.evidence)
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
    assert RootCause.MEASUREMENT_NOISE.name in report.root_causes
    assert report.confidence >= 0.8
    assert any(item.root_cause == RootCause.MEASUREMENT_NOISE.name for item in report.evidence)
    assert report.compiler_workspace_allocator_change == 0
    assert report.counterfactuals[-1].level == "D5"
    assert report.counterfactuals[-1].status == "available"
    assert report.counterfactuals[-1].candidate_peak is not None
    assert report.counterfactuals[-1].candidate_peak.live_bytes == measured.measured_peak_bytes


def test_diagnostics_marks_workspace_growth_and_cost_misrank_from_runtime_residuals():
    ir, fixed = _ir_and_fixed()
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    baseline = evaluate_plan(ir, all_save, fixed)
    measured_peak = baseline.simulation.estimated_peak_bytes + (4 << 20)
    measured_step_us = max(
        baseline.simulation.estimated_step_us * 3.0,
        baseline.simulation.estimated_step_us + 5_000.0,
    )
    measured = MeasuredExecutable(
        plan_id="all_save",
        forward_backward=lambda x: x,
        measured_peak_bytes=measured_peak,
        measured_step_us=measured_step_us,
        correctness_passed=True,
    )

    report = diagnose_plan(baseline, baseline, measured=measured)

    assert report.primary_cause is RootCause.WORKSPACE_GROWTH
    assert RootCause.WORKSPACE_GROWTH.name in report.root_causes
    assert RootCause.COST_MODEL_MISRANK.name in report.root_causes
    assert report.confidence >= 0.8
    assert any(item.root_cause == RootCause.WORKSPACE_GROWTH.name for item in report.evidence)
    assert any(item.root_cause == RootCause.COST_MODEL_MISRANK.name for item in report.evidence)
    assert report.compiler_workspace_allocator_change == measured_peak - baseline.simulation.estimated_peak_bytes


def test_root_cause_evaluator_scores_ground_truth_labels():
    ir, fixed = _ir_and_fixed()
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    mandatory = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.mandatory_save_reason),
        label="torch_min_cut",
    )
    baseline = evaluate_plan(ir, all_save, fixed)
    candidate = evaluate_plan(ir, mandatory, fixed)
    report = diagnose_plan(baseline, candidate)

    evaluation = evaluate_root_cause_predictions(
        (report,),
        (
            RootCauseGroundTruth(
                plan_id=report.plan_id,
                primary_cause=report.primary_cause.name,
                root_causes=report.root_causes,
            ),
            RootCauseGroundTruth(
                plan_id="missing",
                primary_cause=RootCause.REMATERIALIZATION_WAVE.name,
                root_causes=(RootCause.REMATERIALIZATION_WAVE.name,),
            ),
        ),
    )

    assert evaluation.case_count == 2
    assert evaluation.matched_case_count == 1
    assert evaluation.missing_prediction_count == 1
    assert evaluation.primary_accuracy == 1.0
    if report.root_causes != (RootCause.UNKNOWN.name,):
        assert evaluation.micro_precision == 1.0
        assert evaluation.micro_recall < 1.0

    row_evaluation = evaluate_root_cause_predictions(
        (
            RootCausePrediction(
                plan_id="row",
                primary_cause=RootCause.REMATERIALIZATION_WAVE,
                root_causes=(RootCause.REMATERIALIZATION_WAVE,),
            ),
        ),
        (
            RootCauseGroundTruth(
                plan_id="row",
                primary_cause=RootCause.REMATERIALIZATION_WAVE,
                root_causes=(RootCause.REMATERIALIZATION_WAVE,),
            ),
        ),
    )
    assert row_evaluation.primary_accuracy == 1.0
    assert row_evaluation.micro_f1 == 1.0
