from __future__ import annotations

import torch
from torch import nn

from peakaware.api import optimize_training
from peakaware.config import PeakAwareConfig


def _run(capture_backend: str, search_algorithm: str = "pareto_beam"):
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return optimize_training(
        model,
        (torch.randn(4, 8),),
        loss_fn=lambda output: output.square().mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(
            capture_backend=capture_backend,
            search_algorithm=search_algorithm,
            beam_width=8,
            max_beam_candidates=32,
            validation_top_k=2,
            measurement_warmup_steps=0,
            measurement_repeats=1,
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            selection_objective="min_time_then_peak",
        ),
    )


def test_public_api_executes_integrated_beam_search() -> None:
    result = _run("aot")

    assert result.analysis.baseline_results
    assert all(item.plan.plan_id.startswith("beam_") for item in result.analysis.baseline_results)
    assert len({item.plan.plan_id for item in result.analysis.baseline_results}) == len(
        result.analysis.baseline_results
    )
    assert result.measured_candidates
    assert all(item.correctness_passed for item in result.measured_candidates)
    assert result.executor.selection_objective == "min_time_then_peak"
    selected = next(
        item
        for item in result.analysis.baseline_results
        if item.plan.plan_id == result.selected_plan.plan_id
    )
    assert selected.simulation.simulated_memory_event_trace
    components = selected.simulation.cost_breakdown["memory_components"]
    assert components["runtime_replica_bytes"] == 0
    assert components["runtime_replica_source"] == "none"
    assert sum(
        bool(item.simulation.simulated_memory_event_trace)
        for item in result.analysis.baseline_results
    ) == 1


def test_fx_fallback_does_not_checkpoint_hashed_all_save_plan() -> None:
    result = _run("fx")
    all_save = next(
        item for item in result.measured_candidates if item.plan_id.startswith("beam_all_save_")
    )

    assert all_save.phase_metrics.get("activation_checkpoint") == 0


def test_public_api_executes_lagrangian_beam_search() -> None:
    result = _run("aot", "lagrangian_beam")

    assert result.analysis.baseline_results
    assert result.measured_candidates
    assert result.selected_plan.plan_id.startswith("beam_")
    assert all(item.correctness_passed for item in result.measured_candidates)
