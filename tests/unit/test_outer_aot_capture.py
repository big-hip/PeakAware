from __future__ import annotations

import pytest
import torch

from peakaware import PeakAwareConfig, optimize_training
from peakaware.memory.fx_timeline import simulate_lowered_fx_l2_event_trace
from peakaware.models import TrainingTaskRegistry
from peakaware.partition.outer_aot import capture_outer_aot_partition


def test_outer_aot_partition_is_captured_before_candidate_execution() -> None:
    torch.manual_seed(0)
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)
    result = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            validation_top_k=0,
            capture_backend="aot",
        ),
    )
    parameter_versions = tuple(parameter._version for parameter in model.parameters())

    outer = capture_outer_aot_partition(
        result.executor.executable,
        args,
        kwargs,
        plan_id=result.selected_plan.plan_id,
    )

    assert tuple(parameter._version for parameter in model.parameters()) == parameter_versions
    assert tuple(outer.fw_graph.graph.find_nodes(op="output"))
    assert tuple(outer.bw_graph.graph.find_nodes(op="placeholder"))
    trace = simulate_lowered_fx_l2_event_trace(
        outer,
        result.analysis.fixed_timeline,
        align=1,
    )
    assert trace


def test_outer_aot_l3_refinement_is_used_without_candidate_measurement() -> None:
    torch.manual_seed(0)
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)

    result = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            validation_top_k=0,
            capture_backend="aot",
            enable_compile=True,
            compiler_refinement_top_k=1,
        ),
    )

    assert result.measured_candidates == ()
    assert result.candidate_attempts == ()
    assert result.optimization_metrics["candidate_validation_measurement_us"] == 0.0
    assert result.optimization_metrics["compiler_refinement_source"] == (
        "outer_aot_fx_l3_liveness"
    )
    assert result.executable.phase_metrics["candidate_gpu_measurements_used"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_outer_aot_capture_does_not_launch_candidate_cuda_graph() -> None:
    value = torch.randn(1024, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    capture_outer_aot_partition(
        lambda tensor: tensor.sin(),
        (value,),
        {},
        plan_id="cuda-no-execution",
    )

    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == baseline
