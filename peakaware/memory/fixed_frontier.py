from __future__ import annotations

import torch
from torch import nn

from peakaware.contracts import (
    CoarseFeasibilityReport,
    FeasibilityReport,
    FixedTimeline,
    JointTrainingIR,
    OptimizerSpec,
)


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def estimate_parameter_bytes(model: nn.Module) -> int:
    return sum(_tensor_nbytes(p) for p in model.parameters())


def estimate_buffer_bytes(model: nn.Module) -> int:
    return sum(_tensor_nbytes(b) for b in model.buffers())


def estimate_gradient_timeline(model: nn.Module) -> int:
    return sum(_tensor_nbytes(p) for p in model.parameters() if p.requires_grad)


def estimate_optimizer_memory(optimizer: torch.optim.Optimizer, parameter_bytes: int) -> tuple[int, int]:
    state_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                state_bytes += _tensor_nbytes(value)
    name = optimizer.__class__.__name__.lower()
    temporary_bytes = 0
    if state_bytes == 0 and "adam" in name:
        state_bytes = 2 * parameter_bytes
    if "adam" in name:
        temporary_bytes = parameter_bytes
    elif "sgd" not in name:
        temporary_bytes = parameter_bytes
    return state_bytes, temporary_bytes


def build_optimizer_spec(optimizer: torch.optim.Optimizer, model: nn.Module) -> OptimizerSpec:
    parameter_count = sum(1 for _ in model.parameters())
    parameter_bytes = estimate_parameter_bytes(model)
    state_bytes, temporary_bytes = estimate_optimizer_memory(optimizer, parameter_bytes)
    modes = tuple(
        flag
        for flag in ("fused", "foreach", "capturable")
        if any(group.get(flag) for group in optimizer.param_groups)
    )
    name = optimizer.__class__.__name__
    if modes:
        name = f"{name}[{','.join(modes)}]"
    return OptimizerSpec(
        name=name,
        param_group_count=len(optimizer.param_groups),
        parameter_count=parameter_count,
        state_bytes=state_bytes,
        temporary_bytes=temporary_bytes,
    )


def build_fixed_timeline(model: nn.Module, optimizer: torch.optim.Optimizer) -> FixedTimeline:
    parameter_bytes = estimate_parameter_bytes(model)
    optimizer_state_bytes, optimizer_temporary_bytes = estimate_optimizer_memory(optimizer, parameter_bytes)
    return FixedTimeline(
        parameter_bytes=parameter_bytes,
        buffer_bytes=estimate_buffer_bytes(model),
        gradient_bytes=estimate_gradient_timeline(model),
        optimizer_state_bytes=optimizer_state_bytes,
        optimizer_temporary_bytes=optimizer_temporary_bytes,
    )


def _status_and_explanations(
    budget: int,
    fixed: int,
    activation_headroom: int,
    *,
    infeasible_lower_bound: int | None = None,
) -> tuple[str, tuple[str, ...]]:
    hard_lower_bound = fixed if infeasible_lower_bound is None else infeasible_lower_bound
    if hard_lower_bound > budget:
        return (
            "INFEASIBLE_BY_ACTIVATION_ONLY",
            (f"fixed lower bound {hard_lower_bound} bytes exceeds budget {budget} bytes",),
        )
    if activation_headroom < max(1 << 20, budget // 20):
        return (
            "LOW_ACTIVATION_HEADROOM",
            (f"activation headroom is low: {activation_headroom} bytes",),
        )
    return ("FEASIBLE", (f"activation headroom: {activation_headroom} bytes",))


def analyze_coarse_feasibility(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory_budget_bytes: int,
) -> tuple[FixedTimeline, CoarseFeasibilityReport]:
    fixed_timeline = build_fixed_timeline(model, optimizer)
    activation_budget = memory_budget_bytes - fixed_timeline.peak_lower_bound_bytes
    status, explanations = _status_and_explanations(
        memory_budget_bytes,
        fixed_timeline.peak_lower_bound_bytes,
        activation_budget,
        infeasible_lower_bound=fixed_timeline.steady_bytes,
    )
    return fixed_timeline, CoarseFeasibilityReport(
        user_budget_bytes=memory_budget_bytes,
        fixed_peak_lower_bound=fixed_timeline.peak_lower_bound_bytes,
        activation_budget_bytes=max(0, activation_budget),
        status=status,
        explanations=explanations,
    )


def analyze_refined_feasibility(
    ir: JointTrainingIR,
    fixed_timeline: FixedTimeline,
    memory_budget_bytes: int,
) -> FeasibilityReport:
    mandatory_bytes = sum(
        storage.physical_nbytes
        for storage in ir.storages
        if any(
            value.id in storage.value_ids and value.mandatory_save_reason
            for value in ir.values
        )
    )
    fixed_lower = fixed_timeline.peak_lower_bound_bytes + mandatory_bytes
    activation_budget = memory_budget_bytes - fixed_lower
    status, explanations = _status_and_explanations(memory_budget_bytes, fixed_lower, activation_budget)
    return FeasibilityReport(
        user_budget_bytes=memory_budget_bytes,
        fixed_peak_lower_bound=fixed_lower,
        activation_budget_bytes=max(0, activation_budget),
        dominant_phase="optimizer" if fixed_timeline.optimizer_temporary_bytes else "backward",
        status=status,
        explanations=explanations,
    )
