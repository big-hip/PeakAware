from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch
from torch import Tensor, nn

from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import AnalysisBundle, HardwareSpec, OptimizedTrainingResult, TrainingRequest
from peakaware.errors import InfeasibleBudgetError
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import (
    analyze_coarse_feasibility,
    analyze_refined_feasibility,
    build_optimizer_spec,
)
from peakaware.partition.aot import partition_joint_graph
from peakaware.partition.verifier import run_aot_eager_dry_run
from peakaware.runtime.executor import build_training_step_executor, make_measured_executable
from peakaware.search.engine import search_plans


def _hardware_spec(args: tuple[Any, ...], kwargs: dict[str, Any]) -> HardwareSpec:
    devices = [value.device for value in list(args) + list(kwargs.values()) if isinstance(value, Tensor)]
    device = str(devices[0]) if devices else ("cuda" if torch.cuda.is_available() else "cpu")
    total = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        total = int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    return HardwareSpec(device=device, cuda_available=torch.cuda.is_available(), total_memory_bytes=total)


def _request_key(model: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], budget: int) -> str:
    h = hashlib.sha256()
    h.update(model.__class__.__qualname__.encode("utf-8"))
    h.update(str(budget).encode("utf-8"))
    for value in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(value, Tensor):
            h.update(str(tuple(value.shape)).encode("utf-8"))
            h.update(str(value.dtype).encode("utf-8"))
            h.update(str(value.device).encode("utf-8"))
    return h.hexdigest()[:16]


def _validate_request(
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    memory_budget_bytes: int,
    config: PeakAwareConfig,
) -> None:
    del loss_fn
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple")
    model_params = {id(p) for p in model.parameters()}
    optim_params = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
    }
    if not optim_params:
        raise ValueError("optimizer has no parameters")
    if not optim_params.issubset(model_params):
        raise ValueError("optimizer contains parameters that are not owned by model")
    if any(isinstance(value, Tensor) and value.requires_grad for value in example_args):
        raise ValueError("M0 expects training inputs without requires_grad")
    config.validate()


def optimize_training(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    example_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    memory_budget_bytes: int,
    config: PeakAwareConfig | None = None,
) -> OptimizedTrainingResult:
    config = config or PeakAwareConfig()
    example_kwargs = dict(example_kwargs or {})
    _validate_request(model, example_args, example_kwargs, loss_fn, optimizer, memory_budget_bytes, config)
    model.train()

    optimizer_spec = build_optimizer_spec(optimizer, model)
    request = TrainingRequest(
        model=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=memory_budget_bytes,
        config=config,
        optimizer_spec=optimizer_spec,
        hardware=_hardware_spec(example_args, example_kwargs),
        request_key=_request_key(model, example_args, example_kwargs, memory_budget_bytes),
    )

    fixed_timeline, coarse = analyze_coarse_feasibility(model, optimizer, memory_budget_bytes)
    if coarse.status == "INFEASIBLE_BY_ACTIVATION_ONLY":
        raise InfeasibleBudgetError(coarse.explanations[0])

    capture = capture_joint_graph(request)
    ir, ir_report = build_joint_ir(capture)
    if not ir_report.valid:
        raise ValueError(f"invalid IR: {ir_report.errors}")

    feasibility = analyze_refined_feasibility(ir, fixed_timeline, memory_budget_bytes)
    if feasibility.status == "INFEASIBLE_BY_ACTIVATION_ONLY":
        raise InfeasibleBudgetError(feasibility.explanations[0])

    safety_margin = max(config.safety_margin_bytes, int(memory_budget_bytes * config.safety_margin_ratio))
    evaluated = search_plans(
        ir,
        fixed_timeline,
        budget_bytes=memory_budget_bytes,
        safety_margin_bytes=safety_margin,
        manual_saved_value_ids=config.manual_saved_value_ids,
        top_k=config.top_k,
    )
    if not evaluated:
        raise InfeasibleBudgetError("no plans were generated")
    selected = next((plan for plan in evaluated if plan.feasible), evaluated[0])

    lowered = partition_joint_graph(capture.joint_module, selected.plan, ir)
    dry_run = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=example_args,
        kwargs=example_kwargs,
        loss_fn=loss_fn,
        atol=config.atol,
        rtol=config.rtol,
    )
    if not (dry_run.abi_valid and dry_run.outputs_match and dry_run.gradients_match):
        raise InfeasibleBudgetError(f"selected plan failed dry-run: {dry_run.failure_reason}")

    executor = build_training_step_executor(model, optimizer, loss_fn, config)
    measured = make_measured_executable(
        selected.plan.plan_id,
        executor,
        example_args,
        example_kwargs,
        selected.simulation.estimated_peak_bytes,
    )
    fallback_ids = tuple(plan.plan.plan_id for plan in evaluated if plan.plan.plan_id != selected.plan.plan_id)
    analysis = AnalysisBundle(
        ir=ir,
        fixed_timeline=fixed_timeline,
        baseline_results=evaluated,
        analysis_key=f"{capture.capture_key}:{optimizer_spec.name}:{memory_budget_bytes}",
    )
    return OptimizedTrainingResult(
        selected_plan=selected.plan,
        executable=measured,
        executor=executor,
        feasibility=feasibility,
        fallback_plan_ids=fallback_ids,
        analysis=analysis,
        dry_run=dry_run,
    )
