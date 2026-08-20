from __future__ import annotations

from dataclasses import dataclass

from peakaware.contracts import EvaluatedPlan, FixedTimeline, JointTrainingIR, RepairHint
from peakaware.cost.base import CostProvider
from peakaware.memory.simulator import SimulationCostCache
from peakaware.search.candidates import SaveCandidate, select_save_candidates
from peakaware.search.engine import PlanEvaluationCache, evaluate_plan
from peakaware.search.plan import build_recompute_plan


@dataclass(frozen=True)
class RepairResult:
    evaluated: EvaluatedPlan
    evaluation_count: int


def estimate_move_delta(candidate: SaveCandidate) -> tuple[int, float]:
    return candidate.bytes_at_peak, candidate.estimated_recompute_us


def apply_move(saved_value_ids: frozenset[int], candidate: SaveCandidate, decision: str) -> frozenset[int]:
    if decision == "DROP":
        return frozenset(saved_value_ids - candidate.value_ids)
    return frozenset(saved_value_ids | candidate.value_ids)


def rank_peak_live_moves(candidates: tuple[SaveCandidate, ...], peak_storage_ids: frozenset[int]) -> tuple[SaveCandidate, ...]:
    live = [candidate for candidate in candidates if candidate.storage_id in peak_storage_ids]
    return tuple(sorted(live or candidates, key=lambda c: (-c.score, c.storage_id)))


def repair_to_budget(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    evaluated: EvaluatedPlan,
    *,
    hints: tuple[RepairHint, ...] = (),
    cost_provider: CostProvider | None = None,
    simulation_cost_cache: SimulationCostCache | None = None,
    evaluation_cache: PlanEvaluationCache | None = None,
    materialize_event_trace: bool = True,
) -> EvaluatedPlan:
    return repair_to_budget_with_count(
        ir,
        fixed_timeline,
        evaluated,
        hints=hints,
        cost_provider=cost_provider,
        simulation_cost_cache=simulation_cost_cache,
        evaluation_cache=evaluation_cache,
        materialize_event_trace=materialize_event_trace,
    ).evaluated


def repair_to_budget_with_count(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    evaluated: EvaluatedPlan,
    *,
    hints: tuple[RepairHint, ...] = (),
    cost_provider: CostProvider | None = None,
    simulation_cost_cache: SimulationCostCache | None = None,
    evaluation_cache: PlanEvaluationCache | None = None,
    materialize_event_trace: bool = True,
) -> RepairResult:
    if evaluated.feasible:
        return RepairResult(evaluated=evaluated, evaluation_count=0)
    candidates = select_save_candidates(ir, hints=hints, cost_provider=cost_provider)
    ranked = rank_peak_live_moves(candidates, evaluated.simulation.peak_snapshot.live_storage_ids)
    saved = set(evaluated.plan.saved_value_ids)
    best = evaluated
    evaluation_count = 0
    for candidate in ranked:
        saved = set(apply_move(frozenset(saved), candidate, "DROP"))
        repaired_plan = build_recompute_plan(
            ir,
            budget_bytes=evaluated.plan.budget_bytes,
            saved_value_ids=frozenset(saved),
            safety_margin_bytes=evaluated.plan.safety_margin_bytes,
            label=f"{evaluated.plan.plan_id}_repair",
        )
        best = evaluate_plan(
            ir,
            repaired_plan,
            fixed_timeline,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            evaluation_cache=evaluation_cache,
            materialize_event_trace=materialize_event_trace,
        )
        evaluation_count += 1
        if best.feasible:
            return RepairResult(evaluated=best, evaluation_count=evaluation_count)
    return RepairResult(evaluated=best, evaluation_count=evaluation_count)
