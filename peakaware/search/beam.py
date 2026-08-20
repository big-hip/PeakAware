from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf

from peakaware.contracts import EvaluatedPlan, FixedTimeline, JointTrainingIR
from peakaware.cost.base import CostProvider, provider_cache_safe
from peakaware.errors import PlanValidationError
from peakaware.memory.simulator import SimulationCostCache, build_simulation_cost_cache

from .candidates import SaveCandidate, select_save_candidates
from .engine import (
    PlanEvaluationCache,
    build_plan_evaluation_cache,
    evaluate_plan,
)
from .plan import (
    PlanBuildCache,
    build_plan_build_cache,
    build_recompute_plan,
    plan_identity_key,
)


@dataclass(frozen=True)
class SearchResult:
    best: EvaluatedPlan
    evaluated: tuple[EvaluatedPlan, ...]
    candidate_count: int
    evaluated_plan_count: int
    exhaustive_plan_count: int
    optimality_proven: bool
    beam_width: int | None
    source_candidate_count: int = 0
    candidate_coarsened: bool = False


@dataclass(frozen=True)
class _BeamState:
    saved_value_ids: frozenset[int]
    evaluated: EvaluatedPlan


@dataclass
class BeamSearchWorkspace:
    """Reusable graph-level and save-set-level state for related searches."""

    ir: JointTrainingIR
    fixed_timeline: FixedTimeline
    cost_provider: CostProvider | None
    candidates: tuple[SaveCandidate, ...]
    mandatory_value_ids: frozenset[int]
    plan_build_cache: PlanBuildCache
    evaluation_cache: PlanEvaluationCache
    completed_evaluations: dict[
        tuple[str, int, int, frozenset[int], bool],
        EvaluatedPlan,
    ]

    def validate_for(
        self,
        ir: JointTrainingIR,
        fixed_timeline: FixedTimeline,
        cost_provider: CostProvider | None,
    ) -> None:
        if self.ir is not ir:
            raise ValueError("beam workspace belongs to a different IR instance")
        if self.fixed_timeline is not fixed_timeline:
            raise ValueError("beam workspace belongs to a different fixed timeline")
        if self.cost_provider is not cost_provider:
            raise ValueError("beam workspace belongs to a different cost provider")
        self.evaluation_cache.validate_for(ir, fixed_timeline, cost_provider)


def build_beam_search_workspace(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    cost_provider: CostProvider | None = None,
    simulation_cost_cache: SimulationCostCache | None = None,
) -> BeamSearchWorkspace:
    """Build state that can be shared across widths, budgets and prune modes."""

    evaluation_cache = build_plan_evaluation_cache(
        ir,
        fixed_timeline,
        cost_provider,
        simulation_cost_cache=simulation_cost_cache,
    )
    return BeamSearchWorkspace(
        ir=ir,
        fixed_timeline=fixed_timeline,
        cost_provider=cost_provider,
        candidates=select_save_candidates(ir, cost_provider=cost_provider),
        mandatory_value_ids=_mandatory_value_ids(ir),
        plan_build_cache=build_plan_build_cache(ir),
        evaluation_cache=evaluation_cache,
        completed_evaluations={},
    )


def _mandatory_value_ids(ir: JointTrainingIR) -> frozenset[int]:
    return frozenset(value.id for value in ir.values if value.mandatory_save_reason)


def _all_candidate_value_ids(candidates: tuple[SaveCandidate, ...]) -> frozenset[int]:
    return frozenset(value_id for candidate in candidates for value_id in candidate.value_ids)


def _coarsen_candidates(
    candidates: tuple[SaveCandidate, ...],
    max_candidate_count: int,
) -> tuple[SaveCandidate, ...]:
    """Keep high-value decisions fine-grained and bundle the long tail.

    Contiguous score-ordered tail groups preserve the full save/drop endpoint,
    unlike simple truncation, while reducing beam depth on large AOT graphs.
    """

    if len(candidates) <= max_candidate_count:
        return candidates
    fine_count = 0 if max_candidate_count == 1 else max(1, max_candidate_count // 2)
    fine = candidates[:fine_count]
    tail = candidates[fine_count:]
    group_count = max_candidate_count - fine_count
    grouped: list[SaveCandidate] = []
    for group_index in range(group_count):
        start = len(tail) * group_index // group_count
        end = len(tail) * (group_index + 1) // group_count
        members = tail[start:end]
        if not members:
            continue
        value_ids = frozenset(
            value_id for member in members for value_id in member.value_ids
        )
        bytes_at_peak = sum(member.bytes_at_peak for member in members)
        recompute_us = sum(member.estimated_recompute_us for member in members)
        grouped.append(
            SaveCandidate(
                storage_id=-(group_index + 1),
                value_ids=value_ids,
                bytes_at_peak=bytes_at_peak,
                estimated_recompute_us=max(recompute_us, 1.0),
                score=bytes_at_peak / max(recompute_us, 1.0),
                reason=f"coarsened_tail:{len(members)}",
                confidence=min(member.confidence for member in members),
            )
        )
    return tuple((*fine, *grouped))


def _objective_key(item: EvaluatedPlan, selection_objective: str) -> tuple[object, ...]:
    if selection_objective == "min_time_then_peak":
        return (
            not item.feasible,
            item.simulation.estimated_step_us,
            item.simulation.estimated_peak_bytes,
            item.simulation.risk_score,
            -item.simulation.confidence,
            item.plan.plan_id,
        )
    if selection_objective == "min_peak_then_time":
        return (
            not item.feasible,
            item.simulation.estimated_peak_bytes,
            item.simulation.estimated_step_us,
            item.simulation.risk_score,
            -item.simulation.confidence,
            item.plan.plan_id,
        )
    raise ValueError(f"unsupported selection objective: {selection_objective}")


def _select_best(
    evaluated: tuple[EvaluatedPlan, ...], selection_objective: str
) -> EvaluatedPlan:
    feasible = tuple(item for item in evaluated if item.feasible)
    if feasible:
        return min(feasible, key=lambda item: _objective_key(item, selection_objective))
    return min(
        evaluated,
        key=lambda item: (
            item.simulation.estimated_peak_bytes,
            item.simulation.estimated_step_us,
            item.simulation.risk_score,
            -item.simulation.confidence,
            item.plan.plan_id,
        ),
    )


def _dominates(left: _BeamState, right: _BeamState) -> bool:
    left_peak = left.evaluated.simulation.estimated_peak_bytes
    left_time = left.evaluated.simulation.estimated_step_us
    right_peak = right.evaluated.simulation.estimated_peak_bytes
    right_time = right.evaluated.simulation.estimated_step_us
    return (
        left_peak <= right_peak
        and left_time <= right_time
        and (left_peak < right_peak or left_time < right_time)
    )


def _pareto_layers(states: tuple[_BeamState, ...]) -> tuple[tuple[_BeamState, ...], ...]:
    """Build exact 2-D Pareto fronts in O(n log n).

    Peak and step time are the only dominance dimensions. Processing equal-peak
    and equal-time groups together preserves the strict part of ``_dominates``:
    points with identical objectives remain in the same layer.
    """

    if not states:
        return ()
    objectives = tuple(
        (
            state.evaluated.simulation.estimated_peak_bytes,
            float(state.evaluated.simulation.estimated_step_us),
            index,
        )
        for index, state in enumerate(states)
    )
    ordered = sorted(objectives)
    time_values = sorted({step_us for _peak, step_us, _index in objectives})
    time_rank = {value: index + 1 for index, value in enumerate(time_values)}
    fenwick = [0] * (len(time_values) + 1)

    def query(index: int) -> int:
        result = 0
        while index > 0:
            result = max(result, fenwick[index])
            index -= index & -index
        return result

    def update(index: int, value: int) -> None:
        while index < len(fenwick):
            fenwick[index] = max(fenwick[index], value)
            index += index & -index

    layer_by_index = [0] * len(states)
    peak_start = 0
    while peak_start < len(ordered):
        peak = ordered[peak_start][0]
        peak_end = peak_start
        while peak_end < len(ordered) and ordered[peak_end][0] == peak:
            peak_end += 1

        lower_time_layer = 0
        time_start = peak_start
        while time_start < peak_end:
            step_us = ordered[time_start][1]
            time_end = time_start
            while time_end < peak_end and ordered[time_end][1] == step_us:
                time_end += 1
            layer = 1 + max(query(time_rank[step_us]), lower_time_layer)
            for _peak, _step_us, original_index in ordered[time_start:time_end]:
                layer_by_index[original_index] = layer
            lower_time_layer = max(lower_time_layer, layer)
            time_start = time_end

        for _peak, step_us, original_index in ordered[peak_start:peak_end]:
            update(time_rank[step_us], layer_by_index[original_index])
        peak_start = peak_end

    max_layer = max(layer_by_index)
    return tuple(
        tuple(
            state
            for index, state in enumerate(states)
            if layer_by_index[index] == layer
        )
        for layer in range(1, max_layer + 1)
    )


def _crowding_distance(states: tuple[_BeamState, ...]) -> dict[frozenset[int], float]:
    distance = {state.saved_value_ids: 0.0 for state in states}
    if len(states) <= 2:
        return {state.saved_value_ids: inf for state in states}
    objectives = (
        lambda state: float(state.evaluated.simulation.estimated_peak_bytes),
        lambda state: float(state.evaluated.simulation.estimated_step_us),
    )
    for objective in objectives:
        ordered = sorted(states, key=lambda state: (objective(state), tuple(sorted(state.saved_value_ids))))
        distance[ordered[0].saved_value_ids] = inf
        distance[ordered[-1].saved_value_ids] = inf
        low = objective(ordered[0])
        high = objective(ordered[-1])
        if high <= low:
            continue
        for index in range(1, len(ordered) - 1):
            key = ordered[index].saved_value_ids
            if distance[key] == inf:
                continue
            distance[key] += (objective(ordered[index + 1]) - objective(ordered[index - 1])) / (high - low)
    return distance


def _layer_order(
    states: tuple[_BeamState, ...],
    *,
    budget_limit_bytes: int,
    selection_objective: str,
) -> tuple[_BeamState, ...]:
    crowding = _crowding_distance(states)

    def key(state: _BeamState) -> tuple[object, ...]:
        peak = state.evaluated.simulation.estimated_peak_bytes
        step_us = state.evaluated.simulation.estimated_step_us
        violation = max(0, peak - budget_limit_bytes)
        if selection_objective == "min_time_then_peak":
            objective = (step_us, peak)
        elif selection_objective == "min_peak_then_time":
            objective = (peak, len(state.saved_value_ids), step_us)
        else:
            raise ValueError(f"unsupported selection objective: {selection_objective}")
        return (
            -crowding[state.saved_value_ids],
            violation,
            *objective,
            tuple(sorted(state.saved_value_ids)),
        )

    return tuple(sorted(states, key=key))


def _prune_beam(
    states: tuple[_BeamState, ...],
    *,
    beam_width: int,
    budget_limit_bytes: int,
    pruning_strategy: str,
    selection_objective: str,
) -> tuple[_BeamState, ...]:
    if len(states) <= beam_width:
        return states
    if pruning_strategy == "objective":
        def objective_key(state: _BeamState) -> tuple[object, ...]:
            peak = state.evaluated.simulation.estimated_peak_bytes
            step_us = state.evaluated.simulation.estimated_step_us
            objective = (
                (step_us, peak)
                if selection_objective == "min_time_then_peak"
                else (peak, len(state.saved_value_ids), step_us)
            )
            return (
                max(0, peak - budget_limit_bytes),
                *objective,
                tuple(sorted(state.saved_value_ids)),
            )

        return tuple(
            sorted(states, key=objective_key)[:beam_width]
        )
    if pruning_strategy == "peak_only":
        return tuple(
            sorted(
                states,
                key=lambda state: (
                    state.evaluated.simulation.estimated_peak_bytes,
                    state.evaluated.simulation.estimated_step_us,
                    tuple(sorted(state.saved_value_ids)),
                ),
            )[:beam_width]
        )
    if pruning_strategy == "lagrangian":
        peaks = [
            float(state.evaluated.simulation.estimated_peak_bytes)
            for state in states
        ]
        steps = [
            float(state.evaluated.simulation.estimated_step_us)
            for state in states
        ]
        peak_span = max(peaks) - min(peaks)
        step_span = max(steps) - min(steps)
        penalty_us_per_byte = step_span / peak_span if peak_span > 0 else 0.0

        def lagrangian_key(state: _BeamState) -> tuple[object, ...]:
            peak = float(state.evaluated.simulation.estimated_peak_bytes)
            step_us = float(state.evaluated.simulation.estimated_step_us)
            violation = max(0.0, peak - float(budget_limit_bytes))
            return (
                step_us + penalty_us_per_byte * violation,
                violation,
                peak,
                step_us,
                tuple(sorted(state.saved_value_ids)),
            )

        ordered = list(sorted(states, key=lagrangian_key))
        if beam_width == 1:
            return (ordered[0],)
        minimum_peak = min(
            states,
            key=lambda state: (
                state.evaluated.simulation.estimated_peak_bytes,
                state.evaluated.simulation.estimated_step_us,
                tuple(sorted(state.saved_value_ids)),
            ),
        )
        minimum_time = min(
            states,
            key=lambda state: (
                state.evaluated.simulation.estimated_step_us,
                state.evaluated.simulation.estimated_peak_bytes,
                tuple(sorted(state.saved_value_ids)),
            ),
        )
        feasible_states = tuple(
            state
            for state in states
            if state.evaluated.simulation.estimated_peak_bytes <= budget_limit_bytes
        )
        best_feasible = (
            None
            if not feasible_states
            else min(
                feasible_states,
                key=lambda state: (
                    state.evaluated.simulation.estimated_step_us,
                    state.evaluated.simulation.estimated_peak_bytes,
                    tuple(sorted(state.saved_value_ids)),
                ),
            )
        )
        selected = [ordered[0]]
        for anchor in (best_feasible, minimum_peak, minimum_time):
            if anchor is not None and anchor not in selected and len(selected) < beam_width:
                selected.append(anchor)
        for state in ordered[1:]:
            if state not in selected:
                selected.append(state)
            if len(selected) >= beam_width:
                break
        return tuple(selected)
    if pruning_strategy == "lagrangian_sweep":
        peaks = [float(state.evaluated.simulation.estimated_peak_bytes) for state in states]
        steps = [float(state.evaluated.simulation.estimated_step_us) for state in states]
        peak_span = max(peaks) - min(peaks)
        step_span = max(steps) - min(steps)
        shadow_price = step_span / peak_span if peak_span > 0.0 and step_span > 0.0 else 0.0

        def stable_key(state: _BeamState) -> tuple[object, ...]:
            return (
                state.evaluated.simulation.estimated_peak_bytes,
                state.evaluated.simulation.estimated_step_us,
                tuple(sorted(state.saved_value_ids)),
            )

        def relaxed_key(state: _BeamState, multiplier: float) -> tuple[object, ...]:
            peak = float(state.evaluated.simulation.estimated_peak_bytes)
            step_us = float(state.evaluated.simulation.estimated_step_us)
            return (
                step_us + shadow_price * multiplier * (peak - float(budget_limit_bytes)),
                abs(peak - float(budget_limit_bytes)),
                peak,
                step_us,
                tuple(sorted(state.saved_value_ids)),
            )

        selected_sweep: list[_BeamState] = []

        def select(state: _BeamState) -> None:
            if state not in selected_sweep and len(selected_sweep) < beam_width:
                selected_sweep.append(state)

        feasible = tuple(
            state
            for state in states
            if state.evaluated.simulation.estimated_peak_bytes <= budget_limit_bytes
        )
        infeasible = tuple(state for state in states if state not in feasible)
        if feasible:
            select(
                min(
                    feasible,
                    key=lambda state: (
                        state.evaluated.simulation.estimated_step_us,
                        -state.evaluated.simulation.estimated_peak_bytes,
                        tuple(sorted(state.saved_value_ids)),
                    ),
                )
            )
        if infeasible:
            select(
                min(
                    infeasible,
                    key=lambda state: (
                        state.evaluated.simulation.estimated_peak_bytes - budget_limit_bytes,
                        state.evaluated.simulation.estimated_step_us,
                        tuple(sorted(state.saved_value_ids)),
                    ),
                )
            )
        select(min(states, key=stable_key))
        select(
            min(
                states,
                key=lambda state: (
                    state.evaluated.simulation.estimated_step_us,
                    state.evaluated.simulation.estimated_peak_bytes,
                    tuple(sorted(state.saved_value_ids)),
                ),
            )
        )
        for multiplier in (0.0, 1.0 / 64.0, 1.0 / 16.0, 0.25, 1.0, 4.0, 16.0, 64.0):
            if shadow_price <= 0.0 and multiplier > 0.0:
                continue
            select(min(states, key=lambda state, value=multiplier: relaxed_key(state, value)))
        if len(selected_sweep) < beam_width:
            for layer in _pareto_layers(states):
                for state in _layer_order(
                    layer,
                    budget_limit_bytes=budget_limit_bytes,
                    selection_objective=selection_objective,
                ):
                    select(state)
                    if len(selected_sweep) >= beam_width:
                        break
                if len(selected_sweep) >= beam_width:
                    break
        return tuple(selected_sweep)
    selected: list[_BeamState] = []
    if (
        pruning_strategy == "pareto_lexicographic"
        and selection_objective == "min_peak_then_time"
    ):
        # Equal current peaks can hide different future peak potential. Prefer the
        # state that has already dropped more values before using time as a tie-break.
        return tuple(
            sorted(
                states,
                key=lambda state: (
                    max(
                        0,
                        state.evaluated.simulation.estimated_peak_bytes
                        - budget_limit_bytes,
                    ),
                    state.evaluated.simulation.estimated_peak_bytes,
                    len(state.saved_value_ids),
                    state.evaluated.simulation.estimated_step_us,
                    tuple(sorted(state.saved_value_ids)),
                ),
            )[:beam_width]
        )
    for layer in _pareto_layers(states):
        if pruning_strategy == "pareto_lexicographic":
            def lexicographic_key(state: _BeamState) -> tuple[object, ...]:
                peak = state.evaluated.simulation.estimated_peak_bytes
                step_us = state.evaluated.simulation.estimated_step_us
                objective = (
                    (step_us, peak)
                    if selection_objective == "min_time_then_peak"
                    else (peak, step_us)
                )
                return (
                    max(0, peak - budget_limit_bytes),
                    *objective,
                    tuple(sorted(state.saved_value_ids)),
                )

            ordered = tuple(
                sorted(layer, key=lexicographic_key)
            )
        else:
            ordered = _layer_order(
                layer,
                budget_limit_bytes=budget_limit_bytes,
                selection_objective=selection_objective,
            )
        remaining = beam_width - len(selected)
        selected.extend(ordered[:remaining])
        if len(selected) >= beam_width:
            break
    return tuple(selected)


def _evaluate_saved_set(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    saved_value_ids: frozenset[int],
    budget_bytes: int,
    safety_margin_bytes: int,
    cost_provider: CostProvider | None,
    simulation_cost_cache: SimulationCostCache | None,
    evaluation_cache: PlanEvaluationCache | None,
    materialize_event_trace: bool,
    label: str,
    strategy_expectation_source: str,
    plan_build_cache: PlanBuildCache,
    completed_evaluations: dict[
        tuple[str, int, int, frozenset[int], bool],
        EvaluatedPlan,
    ]
    | None = None,
) -> EvaluatedPlan:
    identity = plan_identity_key(
        ir.graph_key,
        saved_value_ids,
        budget_bytes,
        plan_cache=plan_build_cache,
    )
    plan = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=saved_value_ids,
        safety_margin_bytes=safety_margin_bytes,
        label=f"{label}_{identity}",
        strategy_expectation_source=strategy_expectation_source,
        plan_cache=plan_build_cache,
    )
    completed_key = (
        plan.plan_id,
        plan.budget_bytes,
        plan.safety_margin_bytes,
        plan.saved_value_ids,
        materialize_event_trace,
    )
    if completed_evaluations is not None:
        completed = completed_evaluations.get(completed_key)
        if completed is not None:
            if evaluation_cache is not None:
                evaluation_cache.hit_count += 1
            return completed
    evaluated = evaluate_plan(
        ir,
        plan,
        fixed_timeline,
        cost_provider=cost_provider,
        simulation_cost_cache=simulation_cost_cache,
        evaluation_cache=evaluation_cache,
        materialize_event_trace=materialize_event_trace,
    )
    if completed_evaluations is not None:
        completed_evaluations[completed_key] = evaluated
    return evaluated


def _materialize_best_event_trace(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    evaluated: tuple[EvaluatedPlan, ...],
    best: EvaluatedPlan,
    *,
    cost_provider: CostProvider | None,
    simulation_cost_cache: SimulationCostCache | None,
    evaluation_cache: PlanEvaluationCache | None = None,
) -> tuple[tuple[EvaluatedPlan, ...], EvaluatedPlan]:
    if best.simulation.simulated_memory_event_trace:
        return evaluated, best
    traced = evaluate_plan(
        ir,
        best.plan,
        fixed_timeline,
        cost_provider=cost_provider,
        simulation_cost_cache=simulation_cost_cache,
        evaluation_cache=evaluation_cache,
        materialize_event_trace=True,
    )
    return (
        tuple(traced if item.plan.plan_id == best.plan.plan_id else item for item in evaluated),
        traced,
    )


def solve_peak_aware_beam(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int = 0,
    beam_width: int = 16,
    max_candidate_count: int = 64,
    candidate_overflow_policy: str = "error",
    cost_provider: CostProvider | None = None,
    selection_objective: str = "min_time_then_peak",
    pruning_strategy: str = "pareto_lexicographic",
    candidate_order: str = "bytes_per_cost",
    event_trace_materialization: str = "best",
    simulation_cost_cache: SimulationCostCache | None = None,
    workspace: BeamSearchWorkspace | None = None,
) -> SearchResult:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_candidate_count <= 0:
        raise ValueError("max_candidate_count must be positive")
    if candidate_overflow_policy not in {"error", "coarsen_tail"}:
        raise ValueError(
            "candidate_overflow_policy must be one of: error, coarsen_tail"
        )
    if pruning_strategy not in {
        "pareto_crowding",
        "pareto_lexicographic",
        "objective",
        "peak_only",
        "lagrangian",
        "lagrangian_sweep",
    }:
        raise ValueError(f"unsupported pruning strategy: {pruning_strategy}")
    if candidate_order not in {"bytes_per_cost", "reverse", "storage_id"}:
        raise ValueError(f"unsupported candidate order: {candidate_order}")
    if selection_objective not in {"min_time_then_peak", "min_peak_then_time"}:
        raise ValueError(f"unsupported selection objective: {selection_objective}")
    if event_trace_materialization not in {"all", "best", "none"}:
        raise ValueError(
            f"unsupported event trace materialization: {event_trace_materialization}"
        )
    evaluation_cache = None
    completed_evaluations = None
    if workspace is not None:
        workspace.validate_for(ir, fixed_timeline, cost_provider)
        if (
            simulation_cost_cache is not None
            and simulation_cost_cache
            is not workspace.evaluation_cache.simulation_cost_cache
        ):
            raise ValueError(
                "simulation cost cache differs from the beam workspace"
            )
        candidates = workspace.candidates
        mandatory = workspace.mandatory_value_ids
        plan_build_cache = workspace.plan_build_cache
        evaluation_cache = workspace.evaluation_cache
        completed_evaluations = workspace.completed_evaluations
        simulation_cost_cache = evaluation_cache.simulation_cost_cache
    else:
        candidates = select_save_candidates(ir, cost_provider=cost_provider)
        mandatory = _mandatory_value_ids(ir)
        plan_build_cache = build_plan_build_cache(ir)
        if provider_cache_safe(cost_provider):
            evaluation_cache = build_plan_evaluation_cache(
                ir,
                fixed_timeline,
                cost_provider,
                simulation_cost_cache=simulation_cost_cache,
            )
            simulation_cost_cache = evaluation_cache.simulation_cost_cache
    if candidate_order == "reverse":
        candidates = tuple(reversed(candidates))
    elif candidate_order == "storage_id":
        candidates = tuple(sorted(candidates, key=lambda candidate: candidate.storage_id))
    source_candidate_count = len(candidates)
    if source_candidate_count > max_candidate_count:
        if candidate_overflow_policy == "error":
            raise PlanValidationError(
                f"beam search supports at most {max_candidate_count} candidates, "
                f"got {source_candidate_count}; set "
                "candidate_overflow_policy='coarsen_tail' for the experimental "
                "lossy coarsening route"
            )
        candidates = _coarsen_candidates(candidates, max_candidate_count)
    initial_saved = _all_candidate_value_ids(candidates) | mandatory
    evaluated_by_saved: dict[frozenset[int], EvaluatedPlan] = {}
    if simulation_cost_cache is not None and evaluation_cache is None:
        simulation_cost_cache.validate_for(ir, fixed_timeline, cost_provider)
    elif simulation_cost_cache is None and provider_cache_safe(cost_provider):
        simulation_cost_cache = build_simulation_cost_cache(ir, fixed_timeline, cost_provider)
    materialize_during_search = event_trace_materialization == "all" or (
        event_trace_materialization == "best" and simulation_cost_cache is None
    )
    strategy_expectation_source = f"peak_aware_{pruning_strategy}_beam"

    def evaluate(saved: frozenset[int], label: str) -> EvaluatedPlan:
        normalized = saved | mandatory
        existing = evaluated_by_saved.get(normalized)
        if existing is not None:
            return existing
        item = _evaluate_saved_set(
            ir,
            fixed_timeline,
            saved_value_ids=normalized,
            budget_bytes=budget_bytes,
            safety_margin_bytes=safety_margin_bytes,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            evaluation_cache=evaluation_cache,
            materialize_event_trace=materialize_during_search,
            label=label,
            strategy_expectation_source=strategy_expectation_source,
            plan_build_cache=plan_build_cache,
            completed_evaluations=completed_evaluations,
        )
        evaluated_by_saved[normalized] = item
        return item

    initial = evaluate(initial_saved, "beam_all_save")
    beam = (_BeamState(initial_saved, initial),)
    budget_limit = budget_bytes - safety_margin_bytes
    for depth, candidate in enumerate(candidates):
        expanded: dict[frozenset[int], _BeamState] = {}
        for state in beam:
            keep_saved = state.saved_value_ids
            expanded.setdefault(keep_saved, state)
            drop_saved = frozenset(keep_saved - candidate.value_ids) | mandatory
            dropped = evaluate(drop_saved, f"beam_d{depth}_drop_s{candidate.storage_id}")
            expanded.setdefault(drop_saved, _BeamState(drop_saved, dropped))
        beam = _prune_beam(
            tuple(expanded.values()),
            beam_width=beam_width,
            budget_limit_bytes=budget_limit,
            pruning_strategy=pruning_strategy,
            selection_objective=selection_objective,
        )

    evaluated = tuple(evaluated_by_saved.values())
    best = _select_best(evaluated, selection_objective)
    if event_trace_materialization == "best" and simulation_cost_cache is not None:
        evaluated, best = _materialize_best_event_trace(
            ir,
            fixed_timeline,
            evaluated,
            best,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            evaluation_cache=evaluation_cache,
        )
    return SearchResult(
        best=best,
        evaluated=evaluated,
        candidate_count=len(candidates),
        evaluated_plan_count=len(evaluated),
        exhaustive_plan_count=1 << len(candidates),
        optimality_proven=(
            source_candidate_count == len(candidates)
            and len(evaluated) == (1 << len(candidates))
        ),
        beam_width=beam_width,
        source_candidate_count=source_candidate_count,
        candidate_coarsened=source_candidate_count != len(candidates),
    )


def solve_exact_candidate_sets(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int = 0,
    max_candidate_count: int = 16,
    cost_provider: CostProvider | None = None,
    selection_objective: str = "min_time_then_peak",
    event_trace_materialization: str = "best",
    simulation_cost_cache: SimulationCostCache | None = None,
) -> SearchResult:
    if max_candidate_count <= 0:
        raise ValueError("max_candidate_count must be positive")
    if event_trace_materialization not in {"all", "best", "none"}:
        raise ValueError(
            f"unsupported event trace materialization: {event_trace_materialization}"
        )
    candidates = select_save_candidates(ir, cost_provider=cost_provider)
    if len(candidates) > max_candidate_count:
        raise PlanValidationError(
            f"exact candidate-set solver supports at most {max_candidate_count} candidates, got {len(candidates)}"
        )
    mandatory = _mandatory_value_ids(ir)
    plan_build_cache = build_plan_build_cache(ir)
    if simulation_cost_cache is not None:
        simulation_cost_cache.validate_for(ir, fixed_timeline, cost_provider)
    elif provider_cache_safe(cost_provider):
        simulation_cost_cache = build_simulation_cost_cache(ir, fixed_timeline, cost_provider)
    materialize_during_search = event_trace_materialization == "all" or (
        event_trace_materialization == "best" and simulation_cost_cache is None
    )
    evaluated: list[EvaluatedPlan] = []
    for saved_count in range(len(candidates) + 1):
        for saved_candidates in combinations(candidates, saved_count):
            saved = frozenset(
                value_id
                for candidate in saved_candidates
                for value_id in candidate.value_ids
            ) | mandatory
            evaluated.append(
                _evaluate_saved_set(
                    ir,
                    fixed_timeline,
                    saved_value_ids=saved,
                    budget_bytes=budget_bytes,
                    safety_margin_bytes=safety_margin_bytes,
                    cost_provider=cost_provider,
                    simulation_cost_cache=simulation_cost_cache,
                    evaluation_cache=None,
                    materialize_event_trace=materialize_during_search,
                    label=f"exact_saved_{saved_count}",
                    strategy_expectation_source="peak_aware_exact_candidate_sets",
                    plan_build_cache=plan_build_cache,
                )
            )
    evaluated_tuple = tuple(evaluated)
    best = _select_best(evaluated_tuple, selection_objective)
    if event_trace_materialization == "best" and simulation_cost_cache is not None:
        evaluated_tuple, best = _materialize_best_event_trace(
            ir,
            fixed_timeline,
            evaluated_tuple,
            best,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
        )
    return SearchResult(
        best=best,
        evaluated=evaluated_tuple,
        candidate_count=len(candidates),
        evaluated_plan_count=len(evaluated),
        exhaustive_plan_count=len(evaluated),
        optimality_proven=True,
        beam_width=None,
        source_candidate_count=len(candidates),
        candidate_coarsened=False,
    )
