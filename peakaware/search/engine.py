from __future__ import annotations

from dataclasses import replace

from peakaware.contracts import EvaluatedPlan, FixedTimeline, JointTrainingIR, RecomputePlan
from peakaware.cost.base import CostProvider
from peakaware.memory.simulator import simulate_plan

from .candidates import select_save_candidates
from .closure import derive_recompute_closure, validate_closure
from .pareto import select_pareto_topk
from .plan import build_recompute_plan


def evaluate_plan(
    ir: JointTrainingIR,
    plan: RecomputePlan,
    fixed_timeline: FixedTimeline,
) -> EvaluatedPlan:
    closure = derive_recompute_closure(ir, plan.saved_value_ids)
    closure_valid, reason = validate_closure(closure)
    simulation = simulate_plan(ir, plan, fixed_timeline)
    completed = replace(
        plan,
        estimated_peak_bytes=simulation.estimated_peak_bytes,
        estimated_step_us=simulation.estimated_step_us,
        max_recompute_live_bytes=simulation.max_recompute_live_bytes,
        recompute_span_ops=simulation.recompute_span_ops,
        recompute_before_first_bw_op_bytes=simulation.recompute_before_first_bw_op_bytes,
        risk_score=simulation.risk_score,
        confidence=simulation.confidence,
    )
    feasible = closure_valid and simulation.estimated_peak_bytes <= (plan.budget_bytes - plan.safety_margin_bytes)
    rejection_reason = reason
    if closure_valid and not feasible:
        rejection_reason = "estimated peak exceeds search budget"
    return EvaluatedPlan(plan=completed, simulation=simulation, feasible=feasible, rejection_reason=rejection_reason)


def _all_forward_value_ids(ir: JointTrainingIR) -> frozenset[int]:
    return frozenset(v.id for v in ir.values if v.phase == "fw")


def _mandatory_value_ids(ir: JointTrainingIR) -> frozenset[int]:
    return frozenset(v.id for v in ir.values if v.mandatory_save_reason)


def _manual_default_plans(ir: JointTrainingIR, budget_bytes: int, safety_margin_bytes: int) -> tuple[RecomputePlan, ...]:
    all_save = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=_all_forward_value_ids(ir),
        safety_margin_bytes=safety_margin_bytes,
        label="all_save",
    )
    mandatory_only = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=_mandatory_value_ids(ir),
        safety_margin_bytes=safety_margin_bytes,
        label="mandatory_only",
    )
    middle_values = tuple(sorted(_all_forward_value_ids(ir) - _mandatory_value_ids(ir)))
    half = frozenset(middle_values[::2]) | _mandatory_value_ids(ir)
    alternating = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=half,
        safety_margin_bytes=safety_margin_bytes,
        label="manual_alternating",
    )
    return (all_save, mandatory_only, alternating)


def _greedy_seed_plans(
    ir: JointTrainingIR,
    budget_bytes: int,
    safety_margin_bytes: int,
    cost_provider: CostProvider | None,
) -> tuple[RecomputePlan, ...]:
    all_fw = _all_forward_value_ids(ir)
    mandatory = _mandatory_value_ids(ir)
    candidates = select_save_candidates(ir, cost_provider=cost_provider)
    saved = set(all_fw)
    plans: list[RecomputePlan] = []
    for index, candidate in enumerate(candidates):
        saved -= set(candidate.value_ids)
        plans.append(
            build_recompute_plan(
                ir,
                budget_bytes=budget_bytes,
                saved_value_ids=frozenset(saved | mandatory),
                safety_margin_bytes=safety_margin_bytes,
                label=f"greedy_drop_{index}",
            )
        )
        if len(plans) >= 4:
            break
    return tuple(plans)


def search_plans(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int,
    manual_saved_value_ids: tuple[frozenset[int], ...] = (),
    cost_provider: CostProvider | None = None,
    top_k: int = 3,
) -> tuple[EvaluatedPlan, ...]:
    plans = list(_manual_default_plans(ir, budget_bytes, safety_margin_bytes))
    plans.extend(_greedy_seed_plans(ir, budget_bytes, safety_margin_bytes, cost_provider))
    for index, saved in enumerate(manual_saved_value_ids):
        plans.append(
            build_recompute_plan(
                ir,
                budget_bytes=budget_bytes,
                saved_value_ids=saved,
                safety_margin_bytes=safety_margin_bytes,
                label=f"manual_{index}",
            )
        )
    evaluated = [evaluate_plan(ir, plan, fixed_timeline) for plan in plans]
    from .repair import repair_to_budget

    repaired = [repair_to_budget(ir, fixed_timeline, plan) for plan in evaluated if not plan.feasible]
    unique: dict[str, EvaluatedPlan] = {}
    for plan in evaluated + repaired:
        unique.setdefault(plan.plan.plan_id, plan)
    return select_pareto_topk(tuple(unique.values()), top_k)


def improve_feasible_plan(evaluated: EvaluatedPlan) -> EvaluatedPlan:
    return evaluated


def apply_early_stop_policy(evaluated: tuple[EvaluatedPlan, ...]) -> str | None:
    if not any(plan.feasible for plan in evaluated):
        return "no feasible M0 plan under search budget"
    return None


def compute_plan_risk_score(plan: RecomputePlan) -> float:
    return plan.risk_score


def select_compile_topk(evaluated: tuple[EvaluatedPlan, ...], top_k: int) -> tuple[EvaluatedPlan, ...]:
    return tuple(evaluated[:top_k])
