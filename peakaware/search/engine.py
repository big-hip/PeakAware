from __future__ import annotations

from dataclasses import replace

from peakaware.contracts import (
    EarlyStopEvidence,
    EarlyStopReport,
    EvaluatedPlan,
    FixedTimeline,
    JointTrainingIR,
    RecomputePlan,
    RepairHint,
)
from peakaware.cost.base import CostProvider
from peakaware.diagnostics import diagnose_plan
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
        label="torch_min_cut",
    )
    middle_values = tuple(sorted(_all_forward_value_ids(ir) - _mandatory_value_ids(ir)))
    checkpoint_boundary = max(1, len(middle_values) // 2)
    checkpoint_values = frozenset(middle_values[:checkpoint_boundary]) | _mandatory_value_ids(ir)
    block_checkpoint = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=checkpoint_values,
        safety_margin_bytes=safety_margin_bytes,
        label="block_checkpoint",
    )
    return (all_save, mandatory_only, block_checkpoint)


def _greedy_seed_plans(
    ir: JointTrainingIR,
    budget_bytes: int,
    safety_margin_bytes: int,
    cost_provider: CostProvider | None,
    hints: tuple[RepairHint, ...] = (),
) -> tuple[RecomputePlan, ...]:
    all_fw = _all_forward_value_ids(ir)
    mandatory = _mandatory_value_ids(ir)
    candidates = select_save_candidates(ir, cost_provider=cost_provider, hints=hints)
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
    repair_hints: tuple[RepairHint, ...] = (),
    top_k: int = 3,
) -> tuple[EvaluatedPlan, ...]:
    plans = list(_manual_default_plans(ir, budget_bytes, safety_margin_bytes))
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
    baseline_evaluated = [evaluate_plan(ir, plan, fixed_timeline) for plan in plans]
    baseline = next((plan for plan in baseline_evaluated if plan.plan.plan_id == "all_save"), baseline_evaluated[0])
    diagnostic_hints = tuple(
        hint
        for candidate in baseline_evaluated
        for hint in diagnose_plan(baseline, candidate).repair_hints
    )
    hints = repair_hints + diagnostic_hints
    plans.extend(_greedy_seed_plans(ir, budget_bytes, safety_margin_bytes, cost_provider, hints))
    searched = [
        evaluate_plan(ir, plan, fixed_timeline)
        for plan in plans[len(baseline_evaluated) :]
    ]
    from .repair import repair_to_budget

    repaired = [repair_to_budget(ir, fixed_timeline, plan, hints=hints) for plan in searched if not plan.feasible]
    unique: dict[str, EvaluatedPlan] = {}
    for plan in select_pareto_topk(tuple(searched + repaired), top_k):
        unique.setdefault(plan.plan.plan_id, plan)
    return tuple(baseline_evaluated) + tuple(unique.values())


def search_plans_with_report(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int,
    manual_saved_value_ids: tuple[frozenset[int], ...] = (),
    cost_provider: CostProvider | None = None,
    repair_hints: tuple[RepairHint, ...] = (),
    top_k: int = 3,
) -> tuple[tuple[EvaluatedPlan, ...], EarlyStopReport | None]:
    evaluated = search_plans(
        ir,
        fixed_timeline,
        budget_bytes=budget_bytes,
        safety_margin_bytes=safety_margin_bytes,
        manual_saved_value_ids=manual_saved_value_ids,
        cost_provider=cost_provider,
        repair_hints=repair_hints,
        top_k=top_k,
    )
    return evaluated, apply_early_stop_policy(evaluated, fixed_timeline=fixed_timeline)


def improve_feasible_plan(evaluated: EvaluatedPlan) -> EvaluatedPlan:
    return evaluated


def _best_so_far(evaluated: tuple[EvaluatedPlan, ...]) -> EvaluatedPlan | None:
    if not evaluated:
        return None
    return sorted(
        evaluated,
        key=lambda plan: (
            not plan.feasible,
            plan.simulation.estimated_peak_bytes,
            plan.simulation.estimated_step_us,
            plan.simulation.risk_score,
            -plan.simulation.confidence,
            plan.plan.plan_id,
        ),
    )[0]


def _early_stop_report(
    reason: str,
    evaluated: tuple[EvaluatedPlan, ...],
    best: EvaluatedPlan | None,
    fixed_timeline: FixedTimeline | None,
) -> EarlyStopReport:
    evidence = EarlyStopEvidence(
        evaluated_plan_count=len(evaluated),
        feasible_plan_count=sum(1 for plan in evaluated if plan.feasible),
        best_plan_id=None if best is None else best.plan.plan_id,
        best_estimated_peak_bytes=None if best is None else best.simulation.estimated_peak_bytes,
        best_estimated_step_us=None if best is None else best.simulation.estimated_step_us,
        best_risk_score=None if best is None else best.simulation.risk_score,
        best_confidence=None if best is None else best.simulation.confidence,
        fixed_peak_lower_bound_bytes=None if fixed_timeline is None else fixed_timeline.peak_lower_bound_bytes,
        budget_bytes=None if best is None else best.plan.budget_bytes,
    )
    return EarlyStopReport(reason=reason, evidence=evidence, best_plan_id=evidence.best_plan_id)


def apply_early_stop_policy(
    evaluated: tuple[EvaluatedPlan, ...],
    *,
    fixed_timeline: FixedTimeline | None = None,
) -> EarlyStopReport | None:
    best = _best_so_far(evaluated)
    if not evaluated:
        return _early_stop_report("no candidate plans were generated", evaluated, best, fixed_timeline)
    if not any(plan.feasible for plan in evaluated):
        return _early_stop_report("no feasible plan under search budget", evaluated, best, fixed_timeline)
    if (
        fixed_timeline is not None
        and best is not None
        and best.feasible
        and best.simulation.estimated_peak_bytes <= fixed_timeline.peak_lower_bound_bytes
    ):
        return _early_stop_report("fixed frontier dominates peak; further drops cannot reduce peak", evaluated, best, fixed_timeline)
    return None


def compute_plan_risk_score(plan: RecomputePlan) -> float:
    return plan.risk_score


def select_compile_topk(evaluated: tuple[EvaluatedPlan, ...], top_k: int) -> tuple[EvaluatedPlan, ...]:
    return tuple(evaluated[:top_k])
