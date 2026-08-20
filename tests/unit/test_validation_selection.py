from __future__ import annotations

from peakaware.api import (
    _measured_candidate_rejection_reason,
    _order_validation_candidates,
    _release_dry_run_temporaries,
    _select_validation_candidates,
)
from peakaware.contracts import (
    EvaluatedPlan,
    MeasuredExecutable,
    PeakSnapshot,
    RecomputePlan,
    SimulationResult,
)


def _candidate(
    plan_id: str,
    *,
    saved_value_ids: frozenset[int],
    estimated_step_us: float,
    feasible: bool = True,
) -> EvaluatedPlan:
    plan = RecomputePlan(
        graph_key="graph",
        budget_bytes=1_000,
        storage_effects=(),
        saved_value_ids=saved_value_ids,
        mandatory_value_ids=frozenset(),
        estimated_peak_bytes=100,
        estimated_step_us=estimated_step_us,
        max_recompute_live_bytes=0,
        recompute_span_ops=0,
        recompute_before_first_bw_op_bytes=0,
        risk_score=0.0,
        confidence=1.0,
        safety_margin_bytes=0,
        cost_sources=(),
        plan_id=plan_id,
    )
    snapshot = PeakSnapshot(
        phase="bw",
        op_id=None,
        live_storage_ids=frozenset(),
        live_bytes=100,
        parameter_bytes=0,
        gradient_bytes=0,
        optimizer_bytes=0,
        saved_activation_bytes=0,
        recomputed_bytes=0,
        workspace_bytes=0,
    )
    simulation = SimulationResult(
        plan_id=plan_id,
        estimated_peak_bytes=100,
        estimated_step_us=estimated_step_us,
        peak_snapshot=snapshot,
        after_fw_retained_bytes=0,
        fw_peak_bytes=100,
        bw_peak_bytes=100,
        optimizer_peak_bytes=100,
        max_recompute_live_bytes=0,
        recompute_span_ops=0,
        recompute_before_first_bw_op_bytes=0,
        risk_score=0.0,
        confidence=1.0,
    )
    return EvaluatedPlan(
        plan=plan,
        simulation=simulation,
        feasible=feasible,
        rejection_reason=None if feasible else "estimated infeasible",
    )


def _pool(*, baselines_feasible: bool = True) -> tuple[EvaluatedPlan, ...]:
    return (
        _candidate("all_save", saved_value_ids=frozenset(range(1, 11)), estimated_step_us=1),
        _candidate("greedy_drop_0", saved_value_ids=frozenset(range(1, 10)), estimated_step_us=2),
        _candidate("greedy_drop_1", saved_value_ids=frozenset(range(1, 9)), estimated_step_us=3),
        _candidate("greedy_drop_2", saved_value_ids=frozenset(range(1, 8)), estimated_step_us=4),
        _candidate("greedy_drop_3", saved_value_ids=frozenset(range(1, 7)), estimated_step_us=5),
        _candidate(
            "block_checkpoint",
            saved_value_ids=frozenset(range(1, 6)),
            estimated_step_us=50,
            feasible=baselines_feasible,
        ),
        _candidate(
            "torch_min_cut",
            saved_value_ids=frozenset({1}),
            estimated_step_us=60,
            feasible=baselines_feasible,
        ),
    )


def test_ranked_validation_policy_keeps_objective_prefix():
    selected = _select_validation_candidates(
        _pool(),
        validation_top_k=5,
        selection_objective="min_time_then_peak",
        selection_policy="ranked",
    )

    assert [candidate.plan.plan_id for candidate in selected] == [
        "all_save",
        "greedy_drop_0",
        "greedy_drop_1",
        "greedy_drop_2",
        "greedy_drop_3",
    ]


def test_structural_diverse_policy_reserves_baselines_and_spans_saved_sets():
    selected = _select_validation_candidates(
        _pool(),
        validation_top_k=5,
        selection_objective="min_time_then_peak",
        selection_policy="structural_diverse",
    )
    selected_ids = {candidate.plan.plan_id for candidate in selected}

    assert {"all_save", "block_checkpoint", "torch_min_cut"} <= selected_ids
    saved_counts = sorted(len(candidate.plan.saved_value_ids) for candidate in selected)
    assert saved_counts[0] == 1
    assert saved_counts[-1] == 10
    assert len(selected) == 5


def test_structural_diverse_policy_does_not_spend_slots_on_infeasible_baselines():
    selected = _select_validation_candidates(
        _pool(baselines_feasible=False),
        validation_top_k=5,
        selection_objective="min_time_then_peak",
        selection_policy="structural_diverse",
    )

    assert all(candidate.feasible for candidate in selected)
    assert {candidate.plan.plan_id for candidate in selected} == {
        "all_save",
        "greedy_drop_0",
        "greedy_drop_1",
        "greedy_drop_2",
        "greedy_drop_3",
    }


def test_candidate_measurement_order_is_seeded_and_preserves_membership():
    candidates = _pool()

    first = _order_validation_candidates(candidates, seed=17)
    repeated = _order_validation_candidates(candidates, seed=17)
    different = _order_validation_candidates(candidates, seed=18)

    assert first == repeated
    assert first != different
    assert {candidate.plan.plan_id for candidate in first} == {
        candidate.plan.plan_id for candidate in candidates
    }
    assert _order_validation_candidates(candidates, seed=None) is candidates


def test_measured_candidate_rejection_reports_budget_gap() -> None:
    measured = MeasuredExecutable(
        plan_id="candidate",
        forward_backward=lambda: None,
        measured_peak_bytes=120,
        measured_step_us=10.0,
        correctness_passed=True,
    )

    reason = _measured_candidate_rejection_reason(
        measured,
        memory_budget_bytes=100,
    )

    assert reason == "measured peak 120 bytes exceeds budget 100 bytes by 20 bytes"


def test_release_dry_run_temporaries_collects_cpu_objects(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("peakaware.api.gc.collect", lambda: calls.append("collect"))

    _release_dry_run_temporaries(__import__("torch").nn.Linear(2, 2))

    assert calls == ["collect"]
