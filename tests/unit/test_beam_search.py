from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from peakaware.config import PeakAwareConfig
from peakaware.contracts import FixedTimeline, JointTrainingIR, OpInfo, StorageInfo, ValueInfo
from peakaware.cost.base import OpCost, StaticCostProvider, provider_cache_safe
from peakaware.errors import PlanValidationError
from peakaware.search.candidates import select_save_candidates
from peakaware.search.beam import (
    _BeamState,
    _coarsen_candidates,
    _pareto_layers,
    _prune_beam,
    build_beam_search_workspace,
    solve_exact_candidate_sets,
    solve_peak_aware_beam,
)


def _linear_ir(candidate_count: int = 5) -> JointTrainingIR:
    ops = []
    values = []
    storages = []
    for index in range(candidate_count):
        ops.append(
            OpInfo(
                index,
                f"fw{index}",
                f"aten.fw{index}",
                "fw",
                () if index == 0 else (index - 1,),
                (index,),
                True,
                None,
            )
        )
        values.append(
            ValueInfo(
                index,
                index,
                () if index + 1 == candidate_count else (index + 1,),
                index,
                100 + index * 20,
                "fw",
                True,
                True,
                None,
                f"v{index}",
            )
        )
        storages.append(StorageInfo(index, (index,), 100 + index * 20, False))
    return JointTrainingIR(
        ops=tuple(ops),
        values=tuple(values),
        storages=tuple(storages),
        regions=(),
        graph_key="beam-linear",
    )


def _fixed() -> FixedTimeline:
    return FixedTimeline(
        parameter_bytes=1_000,
        buffer_bytes=0,
        gradient_bytes=1_000,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )


class _StatefulProvider:
    source = "stateful_test"

    def __init__(self) -> None:
        self.estimate_calls = 0

    def supports(self, signature) -> bool:
        del signature
        return True

    def estimate(self, signature) -> OpCost:
        del signature
        self.estimate_calls += 1
        return OpCost(
            estimated_us=float(self.estimate_calls),
            memory_bytes=0,
            source=self.source,
            confidence=0.5,
        )


class _InheritedStatefulProvider(StaticCostProvider):
    source = "inherited_stateful_test"

    def __init__(self) -> None:
        self.estimate_calls = 0

    def estimate(self, signature) -> OpCost:
        del signature
        self.estimate_calls += 1
        return OpCost(
            estimated_us=float(self.estimate_calls),
            memory_bytes=0,
            source=self.source,
            confidence=0.5,
        )


def test_exhaustive_width_beam_matches_exact_candidate_set_oracle() -> None:
    ir = _linear_ir()
    exact = solve_exact_candidate_sets(ir, _fixed(), budget_bytes=2_520)
    beam = solve_peak_aware_beam(ir, _fixed(), budget_bytes=2_520, beam_width=1 << 5)

    assert exact.best.feasible
    assert beam.best.feasible
    assert beam.best.simulation.estimated_step_us == exact.best.simulation.estimated_step_us
    assert beam.best.simulation.estimated_peak_bytes == exact.best.simulation.estimated_peak_bytes
    assert beam.optimality_proven is True
    assert len({item.plan.plan_id for item in exact.evaluated}) == len(exact.evaluated)
    assert len({item.plan.plan_id for item in beam.evaluated}) == len(beam.evaluated)


def test_narrow_beam_reduces_simulated_plan_count() -> None:
    ir = _linear_ir(candidate_count=8)
    beam = solve_peak_aware_beam(ir, _fixed(), budget_bytes=2_800, beam_width=4)

    assert beam.candidate_count == 8
    assert beam.evaluated_plan_count < beam.exhaustive_plan_count
    assert beam.evaluated_plan_count <= 1 + 2 * beam.beam_width * beam.candidate_count
    assert beam.optimality_proven is False


def test_large_candidate_space_is_coarsened_without_losing_full_drop_endpoint() -> None:
    ir = _linear_ir(candidate_count=12)
    result = solve_peak_aware_beam(
        ir,
        _fixed(),
        budget_bytes=2_200,
        beam_width=8,
        max_candidate_count=4,
        candidate_overflow_policy="coarsen_tail",
    )

    assert result.source_candidate_count == 12
    assert result.candidate_count == 4
    assert result.candidate_coarsened is True
    assert result.optimality_proven is False
    assert result.evaluated_plan_count <= 1 + 2 * 8 * 4
    assert any(not item.plan.saved_value_ids for item in result.evaluated)


def test_candidate_coarsening_preserves_high_score_prefix_and_all_values():
    ir = _linear_ir(candidate_count=10)
    candidates = select_save_candidates(ir)

    coarsened = _coarsen_candidates(candidates, 4)

    assert coarsened[:2] == candidates[:2]
    assert len(coarsened) == 4
    assert frozenset(value for item in coarsened for value in item.value_ids) == frozenset(
        value for item in candidates for value in item.value_ids
    )


def test_large_candidate_space_requires_explicit_lossy_coarsening() -> None:
    with pytest.raises(PlanValidationError, match="experimental lossy coarsening"):
        solve_peak_aware_beam(
            _linear_ir(candidate_count=12),
            _fixed(),
            budget_bytes=2_200,
            beam_width=8,
            max_candidate_count=4,
        )


def test_pairwise_pareto_layers_preserve_fronts_and_input_order() -> None:
    def state(identifier: int, peak: int, step_us: float) -> _BeamState:
        return _BeamState(
            frozenset({identifier}),
            SimpleNamespace(
                simulation=SimpleNamespace(
                    estimated_peak_bytes=peak,
                    estimated_step_us=step_us,
                )
            ),
        )

    states = (
        state(0, 1, 5.0),
        state(1, 2, 4.0),
        state(2, 3, 6.0),
        state(3, 4, 7.0),
        state(4, 1, 5.0),
    )

    layers = _pareto_layers(states)

    assert [[next(iter(item.saved_value_ids)) for item in layer] for layer in layers] == [
        [0, 1, 4],
        [2],
        [3],
    ]


def test_lagrangian_pruning_keeps_best_relaxed_state_and_feasible_anchor() -> None:
    def state(identifier: int, peak: int, step_us: float) -> _BeamState:
        return _BeamState(
            frozenset({identifier}),
            SimpleNamespace(
                simulation=SimpleNamespace(
                    estimated_peak_bytes=peak,
                    estimated_step_us=step_us,
                )
            ),
        )

    states = (
        state(0, 150, 10.0),
        state(1, 110, 20.0),
        state(2, 90, 40.0),
        state(3, 80, 80.0),
    )

    selected = _prune_beam(
        states,
        beam_width=2,
        budget_limit_bytes=100,
        pruning_strategy="lagrangian",
        selection_objective="min_time_then_peak",
    )

    assert [next(iter(item.saved_value_ids)) for item in selected] == [1, 2]


def test_lagrangian_sweep_preserves_multiple_shadow_price_tradeoffs() -> None:
    def state(identifier: int, peak: int, step_us: float) -> _BeamState:
        return _BeamState(
            frozenset({identifier}),
            SimpleNamespace(
                simulation=SimpleNamespace(
                    estimated_peak_bytes=peak,
                    estimated_step_us=step_us,
                )
            ),
        )

    states = (
        state(0, 150, 10.0),
        state(1, 110, 20.0),
        state(2, 90, 40.0),
        state(3, 80, 80.0),
    )

    selected = _prune_beam(
        states,
        beam_width=4,
        budget_limit_bytes=100,
        pruning_strategy="lagrangian_sweep",
        selection_objective="min_time_then_peak",
    )

    assert {next(iter(item.saved_value_ids)) for item in selected} == {0, 1, 2, 3}


def test_fast_pareto_layers_match_pairwise_reference_on_ties() -> None:
    def state(identifier: int, peak: int, step_us: float) -> _BeamState:
        return _BeamState(
            frozenset({identifier}),
            SimpleNamespace(
                simulation=SimpleNamespace(
                    estimated_peak_bytes=peak,
                    estimated_step_us=step_us,
                )
            ),
        )

    def reference(states):
        remaining = list(range(len(states)))
        layers = []
        while remaining:
            layer = [
                index
                for index in remaining
                if not any(
                    other != index
                    and (
                        states[other].evaluated.simulation.estimated_peak_bytes
                        <= states[index].evaluated.simulation.estimated_peak_bytes
                    )
                    and (
                        states[other].evaluated.simulation.estimated_step_us
                        <= states[index].evaluated.simulation.estimated_step_us
                    )
                    and (
                        states[other].evaluated.simulation.estimated_peak_bytes
                        < states[index].evaluated.simulation.estimated_peak_bytes
                        or states[other].evaluated.simulation.estimated_step_us
                        < states[index].evaluated.simulation.estimated_step_us
                    )
                    for other in remaining
                )
            ]
            layers.append(layer)
            layer_set = set(layer)
            remaining = [index for index in remaining if index not in layer_set]
        return layers

    rng = random.Random(20260801)
    for sample_size in range(1, 40):
        states = tuple(
            state(index, rng.randrange(1, 8), float(rng.randrange(1, 8)))
            for index in range(sample_size)
        )
        actual = [
            [next(iter(item.saved_value_ids)) for item in layer]
            for layer in _pareto_layers(states)
        ]
        assert actual == reference(states)


def test_unknown_stateful_provider_uses_uncached_queries() -> None:
    provider = _StatefulProvider()
    result = solve_peak_aware_beam(
        _linear_ir(candidate_count=5),
        _fixed(),
        budget_bytes=2_800,
        beam_width=4,
        cost_provider=provider,
    )

    assert not hasattr(provider, "cache_safe")
    assert provider.estimate_calls > result.candidate_count + len(_linear_ir().ops)


def test_stateful_subclass_must_explicitly_redeclare_cache_safety() -> None:
    provider = _InheritedStatefulProvider()
    result = solve_peak_aware_beam(
        _linear_ir(candidate_count=5),
        _fixed(),
        budget_bytes=2_800,
        beam_width=4,
        cost_provider=provider,
    )

    assert "cache_safe" not in type(provider).__dict__
    assert provider_cache_safe(provider) is False
    assert provider.estimate_calls > result.candidate_count + len(_linear_ir().ops)


def test_shared_workspace_rejects_unknown_stateful_provider() -> None:
    with pytest.raises(ValueError, match="cache-safe"):
        build_beam_search_workspace(
            _linear_ir(candidate_count=5),
            _fixed(),
            cost_provider=_StatefulProvider(),
        )


def test_event_trace_materialization_is_lazy_without_changing_search_results() -> None:
    kwargs = {
        "ir": _linear_ir(candidate_count=6),
        "fixed_timeline": _fixed(),
        "budget_bytes": 2_800,
        "beam_width": 8,
    }
    all_traces = solve_peak_aware_beam(**kwargs, event_trace_materialization="all")
    best_trace = solve_peak_aware_beam(**kwargs, event_trace_materialization="best")
    no_traces = solve_peak_aware_beam(**kwargs, event_trace_materialization="none")

    def summaries(result):
        return tuple(
            (
                item.plan.plan_id,
                item.feasible,
                item.simulation.estimated_peak_bytes,
                item.simulation.estimated_step_us,
            )
            for item in result.evaluated
        )

    assert summaries(all_traces) == summaries(best_trace) == summaries(no_traces)
    assert all(item.simulation.simulated_memory_event_trace for item in all_traces.evaluated)
    assert sum(bool(item.simulation.simulated_memory_event_trace) for item in best_trace.evaluated) == 1
    assert best_trace.best.simulation.simulated_memory_event_trace
    assert not any(item.simulation.simulated_memory_event_trace for item in no_traces.evaluated)


def test_exact_solver_materializes_only_the_best_trace_by_default() -> None:
    result = solve_exact_candidate_sets(_linear_ir(), _fixed(), budget_bytes=2_800)

    assert result.best.simulation.simulated_memory_event_trace
    assert sum(bool(item.simulation.simulated_memory_event_trace) for item in result.evaluated) == 1


def test_shared_workspace_reuses_exact_save_set_simulations_across_widths() -> None:
    ir = _linear_ir(candidate_count=6)
    fixed = _fixed()
    workspace = build_beam_search_workspace(ir, fixed)
    exhaustive = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_800,
        beam_width=1 << 6,
        event_trace_materialization="none",
        workspace=workspace,
    )
    misses_after_exhaustive = workspace.evaluation_cache.miss_count
    cached = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_760,
        beam_width=4,
        event_trace_materialization="none",
        workspace=workspace,
    )
    standalone = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_760,
        beam_width=4,
        event_trace_materialization="none",
    )

    def summary(result):
        return tuple(
            (
                item.plan.saved_value_ids,
                item.feasible,
                item.simulation.estimated_peak_bytes,
                item.simulation.estimated_step_us,
            )
            for item in result.evaluated
        )

    assert exhaustive.optimality_proven
    assert summary(cached) == summary(standalone)
    assert cached.best.plan.saved_value_ids == standalone.best.plan.saved_value_ids
    assert workspace.evaluation_cache.miss_count == misses_after_exhaustive
    assert workspace.evaluation_cache.hit_count >= cached.evaluated_plan_count


def test_shared_workspace_reuses_completed_evaluations_for_same_search() -> None:
    ir = _linear_ir(candidate_count=6)
    fixed = _fixed()
    workspace = build_beam_search_workspace(ir, fixed)
    first = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_800,
        beam_width=8,
        event_trace_materialization="none",
        workspace=workspace,
    )
    completed_count = len(workspace.completed_evaluations)
    hits_before = workspace.evaluation_cache.hit_count
    second = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_800,
        beam_width=8,
        event_trace_materialization="none",
        workspace=workspace,
    )

    assert second.best is first.best
    assert second.evaluated == first.evaluated
    assert len(workspace.completed_evaluations) == completed_count
    assert workspace.evaluation_cache.hit_count - hits_before == second.evaluated_plan_count


def test_shared_workspace_rejects_different_ir_instances() -> None:
    ir = _linear_ir(candidate_count=4)
    fixed = _fixed()
    workspace = build_beam_search_workspace(ir, fixed)

    with pytest.raises(ValueError, match="different IR"):
        solve_peak_aware_beam(
            _linear_ir(candidate_count=4),
            fixed,
            budget_bytes=2_800,
            workspace=workspace,
        )


def test_shared_workspace_keeps_trace_materialization_modes_separate() -> None:
    ir = _linear_ir(candidate_count=4)
    fixed = _fixed()
    workspace = build_beam_search_workspace(ir, fixed)
    traced = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_800,
        beam_width=16,
        event_trace_materialization="all",
        workspace=workspace,
    )
    summary_only = solve_peak_aware_beam(
        ir,
        fixed,
        budget_bytes=2_800,
        beam_width=16,
        event_trace_materialization="none",
        workspace=workspace,
    )

    assert all(item.simulation.simulated_memory_event_trace for item in traced.evaluated)
    assert not any(
        item.simulation.simulated_memory_event_trace
        for item in summary_only.evaluated
    )


def test_optimality_proven_tracks_actual_exhaustive_evaluation() -> None:
    candidate_count = 5
    beam = solve_peak_aware_beam(
        _linear_ir(candidate_count=candidate_count),
        _fixed(),
        budget_bytes=2_800,
        beam_width=1 << (candidate_count - 1),
    )

    assert beam.beam_width < beam.exhaustive_plan_count
    assert beam.evaluated_plan_count == beam.exhaustive_plan_count
    assert beam.optimality_proven is True


def test_min_peak_objective_controls_pruning_order() -> None:
    ir = _linear_ir(candidate_count=5)
    exact = solve_exact_candidate_sets(
        ir,
        _fixed(),
        budget_bytes=2_800,
        selection_objective="min_peak_then_time",
    )
    beam = solve_peak_aware_beam(
        ir,
        _fixed(),
        budget_bytes=2_800,
        beam_width=1,
        selection_objective="min_peak_then_time",
    )

    assert beam.best.simulation.estimated_peak_bytes == exact.best.simulation.estimated_peak_bytes
    assert beam.best.simulation.estimated_step_us == exact.best.simulation.estimated_step_us


def test_exact_solver_falls_back_to_lowest_peak_when_budget_is_impossible() -> None:
    exact = solve_exact_candidate_sets(_linear_ir(), _fixed(), budget_bytes=1)

    assert exact.best.feasible is False
    assert exact.best.simulation.estimated_peak_bytes == min(
        item.simulation.estimated_peak_bytes for item in exact.evaluated
    )


def test_beam_search_config_validates_algorithm_limits() -> None:
    PeakAwareConfig(search_algorithm="pareto_beam", beam_width=8, max_beam_candidates=32).validate()
    PeakAwareConfig(search_algorithm="lagrangian_beam").validate()

    with pytest.raises(ValueError, match="search_algorithm"):
        PeakAwareConfig(search_algorithm="unknown").validate()
    with pytest.raises(ValueError, match="beam_width"):
        PeakAwareConfig(beam_width=0).validate()
    with pytest.raises(ValueError, match="max_beam_candidates"):
        PeakAwareConfig(max_beam_candidates=0).validate()
    with pytest.raises(ValueError, match="beam_candidate_overflow_policy"):
        PeakAwareConfig(beam_candidate_overflow_policy="unknown").validate()


def test_beam_search_rejects_unknown_ablation_modes() -> None:
    ir = _linear_ir()
    with pytest.raises(ValueError, match="pruning strategy"):
        solve_peak_aware_beam(ir, _fixed(), budget_bytes=2_520, pruning_strategy="unknown")
    with pytest.raises(ValueError, match="candidate order"):
        solve_peak_aware_beam(ir, _fixed(), budget_bytes=2_520, candidate_order="unknown")
    with pytest.raises(ValueError, match="selection objective"):
        solve_peak_aware_beam(
            ir,
            _fixed(),
            budget_bytes=2_520,
            selection_objective="unknown",
        )
    with pytest.raises(ValueError, match="event trace materialization"):
        solve_peak_aware_beam(
            ir,
            _fixed(),
            budget_bytes=2_520,
            event_trace_materialization="unknown",
        )


def test_standalone_solvers_require_positive_candidate_limits() -> None:
    ir = _linear_ir()
    with pytest.raises(ValueError, match="max_candidate_count"):
        solve_peak_aware_beam(ir, _fixed(), budget_bytes=2_520, max_candidate_count=0)
    with pytest.raises(ValueError, match="candidate_overflow_policy"):
        solve_peak_aware_beam(
            ir,
            _fixed(),
            budget_bytes=2_520,
            candidate_overflow_policy="unknown",
        )
    with pytest.raises(ValueError, match="max_candidate_count"):
        solve_exact_candidate_sets(ir, _fixed(), budget_bytes=2_520, max_candidate_count=0)
