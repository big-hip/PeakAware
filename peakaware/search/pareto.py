from __future__ import annotations

from peakaware.contracts import EvaluatedPlan


def dominates(left: EvaluatedPlan, right: EvaluatedPlan) -> bool:
    return (
        left.simulation.estimated_peak_bytes <= right.simulation.estimated_peak_bytes
        and left.simulation.estimated_step_us <= right.simulation.estimated_step_us
        and (
            left.simulation.estimated_peak_bytes < right.simulation.estimated_peak_bytes
            or left.simulation.estimated_step_us < right.simulation.estimated_step_us
        )
    )


def pareto_frontier(evaluated: tuple[EvaluatedPlan, ...]) -> tuple[EvaluatedPlan, ...]:
    frontier = []
    for candidate in evaluated:
        if any(dominates(other, candidate) for other in evaluated if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(
        key=lambda e: (
            not e.feasible,
            e.simulation.risk_score,
            -e.simulation.confidence,
            e.simulation.estimated_peak_bytes,
            e.simulation.estimated_step_us,
            e.plan.plan_id,
        )
    )
    return tuple(frontier)


def select_pareto_topk(evaluated: tuple[EvaluatedPlan, ...], top_k: int) -> tuple[EvaluatedPlan, ...]:
    frontier = list(pareto_frontier(evaluated))
    if len(frontier) < top_k:
        seen = {plan.plan.plan_id for plan in frontier}
        for candidate in sorted(
            evaluated,
            key=lambda e: (
                not e.feasible,
                e.simulation.estimated_peak_bytes,
                e.simulation.estimated_step_us,
                e.plan.plan_id,
            ),
        ):
            if candidate.plan.plan_id not in seen:
                frontier.append(candidate)
                seen.add(candidate.plan.plan_id)
            if len(frontier) >= top_k:
                break
    return tuple(frontier[:top_k])
