from __future__ import annotations

from peakaware.contracts import EvaluatedPlan, FixedTimeline, JointTrainingIR, RepairHint
from peakaware.search.candidates import SaveCandidate, select_save_candidates
from peakaware.search.engine import evaluate_plan
from peakaware.search.plan import build_recompute_plan


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
) -> EvaluatedPlan:
    if evaluated.feasible:
        return evaluated
    candidates = select_save_candidates(ir, hints=hints)
    ranked = rank_peak_live_moves(candidates, evaluated.simulation.peak_snapshot.live_storage_ids)
    saved = set(evaluated.plan.saved_value_ids)
    best = evaluated
    for candidate in ranked:
        saved = set(apply_move(frozenset(saved), candidate, "DROP"))
        repaired_plan = build_recompute_plan(
            ir,
            budget_bytes=evaluated.plan.budget_bytes,
            saved_value_ids=frozenset(saved),
            safety_margin_bytes=evaluated.plan.safety_margin_bytes,
            label=f"{evaluated.plan.plan_id}_repair",
        )
        best = evaluate_plan(ir, repaired_plan, fixed_timeline)
        if best.feasible:
            return best
    return best
