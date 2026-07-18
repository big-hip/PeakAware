from __future__ import annotations

from itertools import combinations

from peakaware.contracts import EvaluatedPlan, FixedTimeline, JointTrainingIR
from peakaware.errors import PlanValidationError
from peakaware.search.candidates import select_save_candidates
from peakaware.search.engine import evaluate_plan
from peakaware.search.plan import build_recompute_plan


def _powerset(values: tuple[int, ...]) -> tuple[frozenset[int], ...]:
    sets: list[frozenset[int]] = []
    for size in range(len(values) + 1):
        for combo in combinations(values, size):
            sets.append(frozenset(combo))
    return tuple(sets)


def solve_exact_small_graph(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int = 0,
    max_candidate_count: int = 12,
) -> EvaluatedPlan:
    candidates = select_save_candidates(ir)
    if len(candidates) > max_candidate_count:
        raise PlanValidationError(
            f"exact solver supports at most {max_candidate_count} candidates, got {len(candidates)}"
        )
    mandatory = frozenset(value.id for value in ir.values if value.mandatory_save_reason)
    candidate_value_ids = tuple(sorted({value_id for candidate in candidates for value_id in candidate.value_ids}))
    evaluated: list[EvaluatedPlan] = []
    for saved_subset in _powerset(candidate_value_ids):
        plan = build_recompute_plan(
            ir,
            budget_bytes=budget_bytes,
            saved_value_ids=saved_subset | mandatory,
            safety_margin_bytes=safety_margin_bytes,
        )
        evaluated.append(evaluate_plan(ir, plan, fixed_timeline))
    evaluated.sort(
        key=lambda item: (
            not item.feasible,
            item.simulation.estimated_peak_bytes,
            item.simulation.estimated_step_us,
            item.simulation.risk_score,
            item.plan.plan_id,
        )
    )
    return evaluated[0]
