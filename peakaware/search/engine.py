from __future__ import annotations

from dataclasses import dataclass, field, replace

from peakaware.contracts import (
    EarlyStopEvidence,
    EarlyStopReport,
    EvaluatedPlan,
    FixedTimeline,
    JointTrainingIR,
    RecomputePlan,
    RepairHint,
    SearchDiagnostics,
    SimulationResult,
)
from peakaware.cost.base import CostProvider, provider_cache_safe
from peakaware.diagnostics import diagnose_plan
from peakaware.memory.simulator import (
    SimulationCostCache,
    build_simulation_cost_cache,
    simulate_plan,
)

from .candidates import SaveCandidate, select_save_candidates
from .closure import derive_recompute_closure, validate_closure
from .pareto import select_pareto_topk
from .plan import build_recompute_plan


@dataclass(frozen=True)
class _CachedPlanEvaluation:
    simulation: SimulationResult
    closure_valid: bool
    closure_reason: str | None


@dataclass
class PlanEvaluationCache:
    """Reusable exact plan simulations for one IR/timeline/provider triple.

    The cache key is the normalized saved-value set. Budget, safety margin and
    human-readable plan labels do not affect the simulated peak/time, so a
    cached simulation can be safely retargeted to plans from later searches.
    Traced and summary-only simulations are kept separately to preserve lazy
    event materialization.
    """

    ir: JointTrainingIR
    fixed_timeline: FixedTimeline
    cost_provider: CostProvider | None
    simulation_cost_cache: SimulationCostCache
    summary_entries: dict[frozenset[int], _CachedPlanEvaluation] = field(
        default_factory=dict
    )
    traced_entries: dict[frozenset[int], _CachedPlanEvaluation] = field(
        default_factory=dict
    )
    hit_count: int = 0
    miss_count: int = 0

    def validate_for(
        self,
        ir: JointTrainingIR,
        fixed_timeline: FixedTimeline,
        cost_provider: CostProvider | None,
    ) -> None:
        if self.ir is not ir:
            raise ValueError("plan evaluation cache belongs to a different IR instance")
        if self.fixed_timeline is not fixed_timeline:
            raise ValueError(
                "plan evaluation cache belongs to a different fixed timeline"
            )
        if self.cost_provider is not cost_provider:
            raise ValueError(
                "plan evaluation cache belongs to a different cost provider"
            )


def build_plan_evaluation_cache(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    cost_provider: CostProvider | None = None,
    *,
    simulation_cost_cache: SimulationCostCache | None = None,
) -> PlanEvaluationCache:
    if not provider_cache_safe(cost_provider):
        raise ValueError(
            "cross-plan evaluation caching requires an explicitly cache-safe "
            "cost provider"
        )
    if simulation_cost_cache is None:
        simulation_cost_cache = build_simulation_cost_cache(
            ir, fixed_timeline, cost_provider
        )
    else:
        simulation_cost_cache.validate_for(ir, fixed_timeline, cost_provider)
    return PlanEvaluationCache(
        ir=ir,
        fixed_timeline=fixed_timeline,
        cost_provider=cost_provider,
        simulation_cost_cache=simulation_cost_cache,
    )


def _complete_evaluated_plan(
    plan: RecomputePlan,
    simulation: SimulationResult,
    *,
    closure_valid: bool,
    closure_reason: str | None,
    cost_provider: CostProvider | None,
) -> EvaluatedPlan:
    if simulation.plan_id != plan.plan_id:
        simulation = replace(simulation, plan_id=plan.plan_id)
    cost_sources = plan.cost_sources
    if cost_provider is not None:
        provider_source = str(
            getattr(cost_provider, "source", cost_provider.__class__.__name__)
        )
        cost_sources = tuple(dict.fromkeys((*cost_sources, provider_source)))
    completed = replace(
        plan,
        estimated_peak_bytes=simulation.estimated_peak_bytes,
        estimated_step_us=simulation.estimated_step_us,
        max_recompute_live_bytes=simulation.max_recompute_live_bytes,
        recompute_span_ops=simulation.recompute_span_ops,
        recompute_before_first_bw_op_bytes=simulation.recompute_before_first_bw_op_bytes,
        risk_score=simulation.risk_score,
        confidence=simulation.confidence,
        cost_sources=cost_sources,
    )
    feasible = closure_valid and simulation.estimated_peak_bytes <= (
        plan.budget_bytes - plan.safety_margin_bytes
    )
    rejection_reason = closure_reason
    if closure_valid and not feasible:
        rejection_reason = "estimated peak exceeds search budget"
    return EvaluatedPlan(
        plan=completed,
        simulation=simulation,
        feasible=feasible,
        rejection_reason=rejection_reason,
    )


def evaluate_plan(
    ir: JointTrainingIR,
    plan: RecomputePlan,
    fixed_timeline: FixedTimeline,
    *,
    cost_provider: CostProvider | None = None,
    simulation_cost_cache: SimulationCostCache | None = None,
    evaluation_cache: PlanEvaluationCache | None = None,
    materialize_event_trace: bool = True,
) -> EvaluatedPlan:
    if evaluation_cache is not None:
        evaluation_cache.validate_for(ir, fixed_timeline, cost_provider)
        if (
            simulation_cost_cache is not None
            and simulation_cost_cache is not evaluation_cache.simulation_cost_cache
        ):
            raise ValueError(
                "simulation cost cache differs from the plan evaluation cache"
            )
        simulation_cost_cache = evaluation_cache.simulation_cost_cache
        normalized_saved = plan.saved_value_ids | plan.mandatory_value_ids
        entries = (
            evaluation_cache.traced_entries
            if materialize_event_trace
            else evaluation_cache.summary_entries
        )
        cached = entries.get(normalized_saved)
        if cached is not None:
            evaluation_cache.hit_count += 1
            return _complete_evaluated_plan(
                plan,
                cached.simulation,
                closure_valid=cached.closure_valid,
                closure_reason=cached.closure_reason,
                cost_provider=cost_provider,
            )
        evaluation_cache.miss_count += 1
    closure = derive_recompute_closure(
        ir,
        plan.saved_value_ids,
        graph_cache=(
            None
            if simulation_cost_cache is None
            else simulation_cost_cache.recompute_graph_cache
        ),
    )
    closure_valid, reason = validate_closure(closure)
    simulation = simulate_plan(
        ir,
        plan,
        fixed_timeline,
        cost_provider=cost_provider,
        cost_cache=simulation_cost_cache,
        materialize_event_trace=materialize_event_trace,
        recompute_closure=closure,
    )
    if evaluation_cache is not None:
        cached = _CachedPlanEvaluation(
            simulation=simulation,
            closure_valid=closure_valid,
            closure_reason=reason,
        )
        entries[normalized_saved] = cached
        if materialize_event_trace:
            evaluation_cache.summary_entries.setdefault(
                normalized_saved,
                _CachedPlanEvaluation(
                    simulation=replace(
                        simulation,
                        simulated_memory_event_trace=(),
                    ),
                    closure_valid=closure_valid,
                    closure_reason=reason,
                ),
            )
    return _complete_evaluated_plan(
        plan,
        simulation,
        closure_valid=closure_valid,
        closure_reason=reason,
        cost_provider=cost_provider,
    )


def _residual_forward_value_ids(ir: JointTrainingIR) -> frozenset[int]:
    return frozenset(v.id for v in ir.values if v.phase == "fw" and v.crosses_fw_bw)


def _mandatory_value_ids(ir: JointTrainingIR) -> frozenset[int]:
    return frozenset(v.id for v in ir.values if v.mandatory_save_reason)


def _manual_default_plans(ir: JointTrainingIR, budget_bytes: int, safety_margin_bytes: int) -> tuple[RecomputePlan, ...]:
    residual_values = _residual_forward_value_ids(ir)
    mandatory_values = _mandatory_value_ids(ir)
    all_save = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=residual_values | mandatory_values,
        safety_margin_bytes=safety_margin_bytes,
        label="all_save",
        strategy_expectation_source="all_save_baseline",
    )
    min_cut = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=mandatory_values,
        safety_margin_bytes=safety_margin_bytes,
        label="torch_min_cut",
        strategy_expectation_source="pytorch_min_cut_proxy",
    )
    middle_values = tuple(sorted(residual_values - mandatory_values))
    checkpoint_boundary = max(1, len(middle_values) // 2)
    checkpoint_values = frozenset(middle_values[:checkpoint_boundary]) | mandatory_values
    block_checkpoint = build_recompute_plan(
        ir,
        budget_bytes=budget_bytes,
        saved_value_ids=checkpoint_values,
        safety_margin_bytes=safety_margin_bytes,
        label="block_checkpoint",
        strategy_expectation_source="block_checkpoint_proxy",
    )
    return (all_save, min_cut, block_checkpoint)


def _greedy_seed_plans(
    ir: JointTrainingIR,
    budget_bytes: int,
    safety_margin_bytes: int,
    cost_provider: CostProvider | None,
    hints: tuple[RepairHint, ...] = (),
    max_candidates: int = 4,
) -> tuple[RecomputePlan, ...]:
    all_fw = _residual_forward_value_ids(ir)
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
                strategy_expectation_source="greedy_bytes_per_cost",
            )
        )
        if len(plans) >= max_candidates:
            break
    return tuple(plans)


def _candidate_order(candidates: tuple[SaveCandidate, ...]) -> tuple[int, ...]:
    return tuple(candidate.storage_id for candidate in candidates)


def _order_delta_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    shared_positions = sum(
        1
        for index, storage_id in enumerate(left)
        if index < len(right) and right[index] == storage_id
    )
    return max(len(left), len(right)) - shared_positions


def search_plans(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int,
    manual_saved_value_ids: tuple[frozenset[int], ...] = (),
    cost_provider: CostProvider | None = None,
    repair_hints: tuple[RepairHint, ...] = (),
    enable_diagnostic_hints: bool = True,
    top_k: int = 3,
    max_greedy_candidates: int = 4,
) -> tuple[EvaluatedPlan, ...]:
    evaluated, _ = search_plans_with_diagnostics(
        ir,
        fixed_timeline,
        budget_bytes=budget_bytes,
        safety_margin_bytes=safety_margin_bytes,
        manual_saved_value_ids=manual_saved_value_ids,
        cost_provider=cost_provider,
        repair_hints=repair_hints,
        enable_diagnostic_hints=enable_diagnostic_hints,
        top_k=top_k,
        max_greedy_candidates=max_greedy_candidates,
    )
    return evaluated


def search_plans_with_diagnostics(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int,
    manual_saved_value_ids: tuple[frozenset[int], ...] = (),
    cost_provider: CostProvider | None = None,
    repair_hints: tuple[RepairHint, ...] = (),
    enable_diagnostic_hints: bool = True,
    top_k: int = 3,
    max_greedy_candidates: int = 4,
) -> tuple[tuple[EvaluatedPlan, ...], SearchDiagnostics]:
    simulation_cost_cache = None
    evaluation_cache = None
    if provider_cache_safe(cost_provider):
        simulation_cost_cache = build_simulation_cost_cache(
            ir,
            fixed_timeline,
            cost_provider,
        )
        evaluation_cache = build_plan_evaluation_cache(
            ir,
            fixed_timeline,
            cost_provider,
            simulation_cost_cache=simulation_cost_cache,
        )

    def evaluate_summary(plan: RecomputePlan) -> EvaluatedPlan:
        return evaluate_plan(
            ir,
            plan,
            fixed_timeline,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            evaluation_cache=evaluation_cache,
            materialize_event_trace=False,
        )

    plans = list(_manual_default_plans(ir, budget_bytes, safety_margin_bytes))
    for index, saved in enumerate(manual_saved_value_ids):
        plans.append(
            build_recompute_plan(
                ir,
                budget_bytes=budget_bytes,
                saved_value_ids=saved,
                safety_margin_bytes=safety_margin_bytes,
                label=f"manual_{index}",
                strategy_expectation_source="manual_saved_value_ids",
            )
        )
    baseline_evaluated = [evaluate_summary(plan) for plan in plans]
    baseline = next((plan for plan in baseline_evaluated if plan.plan.plan_id == "all_save"), baseline_evaluated[0])
    diagnostic_hints = ()
    if enable_diagnostic_hints:
        diagnostic_hints = tuple(
            hint
            for candidate in baseline_evaluated
            for hint in diagnose_plan(baseline, candidate).repair_hints
        )
    hints = repair_hints + diagnostic_hints
    base_candidates = select_save_candidates(ir, cost_provider=cost_provider, hints=repair_hints)
    hinted_candidates = select_save_candidates(ir, cost_provider=cost_provider, hints=hints)
    base_order = _candidate_order(base_candidates)
    hinted_order = _candidate_order(hinted_candidates)
    diagnostic_hint_targets = {target for hint in diagnostic_hints for target in hint.target_ids}
    candidate_storage_ids = {candidate.storage_id for candidate in base_candidates}
    plans.extend(
        _greedy_seed_plans(
            ir,
            budget_bytes,
            safety_margin_bytes,
            cost_provider,
            hints,
            max_candidates=max_greedy_candidates,
        )
    )
    searched = [evaluate_summary(plan) for plan in plans[len(baseline_evaluated) :]]
    from .repair import repair_to_budget_with_count

    repair_results = [
        repair_to_budget_with_count(
            ir,
            fixed_timeline,
            plan,
            hints=hints,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            evaluation_cache=evaluation_cache,
            materialize_event_trace=False,
        )
        for plan in searched
        if not plan.feasible
    ]
    repaired = [result.evaluated for result in repair_results]
    unique: dict[str, EvaluatedPlan] = {}
    for plan in select_pareto_topk(tuple(searched + repaired), top_k):
        unique.setdefault(plan.plan.plan_id, plan)
    evaluated = tuple(baseline_evaluated) + tuple(unique.values())
    diagnostics = SearchDiagnostics(
        diagnostic_hints_enabled=enable_diagnostic_hints,
        manual_hint_count=len(repair_hints),
        diagnostic_hint_count=len(diagnostic_hints),
        diagnostic_hint_kinds=tuple(sorted({hint.kind for hint in diagnostic_hints})),
        diagnostic_hint_candidate_match_count=len(diagnostic_hint_targets & candidate_storage_ids),
        diagnostic_hint_order_changed=base_order != hinted_order,
        diagnostic_hint_order_delta_count=_order_delta_count(base_order, hinted_order),
        greedy_plan_count=len(searched),
        feasible_before_repair_count=sum(1 for plan in searched if plan.feasible),
        repaired_candidate_count=len(repaired),
        repair_success_count=sum(1 for plan in repaired if plan.feasible),
        feasible_after_repair_count=sum(1 for plan in searched + repaired if plan.feasible),
        repaired_plan_ids=tuple(plan.plan.plan_id for plan in repaired),
        repair_evaluation_count=sum(result.evaluation_count for result in repair_results),
        evaluation_cache_hits=0 if evaluation_cache is None else evaluation_cache.hit_count,
        evaluation_cache_misses=0 if evaluation_cache is None else evaluation_cache.miss_count,
        summary_only_evaluation_count=(
            len(baseline_evaluated)
            + len(searched)
            + sum(result.evaluation_count for result in repair_results)
        ),
    )
    return evaluated, diagnostics


def search_plans_with_report(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    *,
    budget_bytes: int,
    safety_margin_bytes: int,
    manual_saved_value_ids: tuple[frozenset[int], ...] = (),
    cost_provider: CostProvider | None = None,
    repair_hints: tuple[RepairHint, ...] = (),
    enable_diagnostic_hints: bool = True,
    top_k: int = 3,
    max_greedy_candidates: int = 4,
) -> tuple[tuple[EvaluatedPlan, ...], EarlyStopReport | None]:
    evaluated = search_plans(
        ir,
        fixed_timeline,
        budget_bytes=budget_bytes,
        safety_margin_bytes=safety_margin_bytes,
        manual_saved_value_ids=manual_saved_value_ids,
        cost_provider=cost_provider,
        repair_hints=repair_hints,
        enable_diagnostic_hints=enable_diagnostic_hints,
        top_k=top_k,
        max_greedy_candidates=max_greedy_candidates,
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
