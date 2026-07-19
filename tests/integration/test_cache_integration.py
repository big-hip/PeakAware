import torch

import peakaware.api as api_module
from peakaware import PeakAwareConfig, optimize_training
from peakaware.models import TrainingTaskRegistry


def test_optimize_training_reuses_analysis_and_executable_cache(tmp_path, monkeypatch):
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    config = PeakAwareConfig(
        cache_root=tmp_path,
        safety_margin_bytes=0,
        safety_margin_ratio=0.0,
        top_k=1,
    )

    torch.manual_seed(0)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(1)
    first = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=config,
    )

    def fail_search(*args, **kwargs):
        del args, kwargs
        raise AssertionError("search should be served by analysis cache")

    def fail_measure(*args, **kwargs):
        del args, kwargs
        raise AssertionError("measurement should be served by executable cache")

    monkeypatch.setattr(api_module, "search_plans", fail_search)
    monkeypatch.setattr(api_module, "make_measured_executable", fail_measure)

    torch.manual_seed(0)
    cached_model = task.build_model()
    cached_optimizer = task.build_optimizer(cached_model)
    cached_args, cached_kwargs = task.build_batch(1)
    second = optimize_training(
        cached_model,
        cached_args,
        example_kwargs=cached_kwargs,
        loss_fn=task.loss_fn,
        optimizer=cached_optimizer,
        memory_budget_bytes=1 << 28,
        config=config,
    )

    assert first.selected_plan.plan_id == second.selected_plan.plan_id
    assert first.executable.measured_peak_bytes == second.executable.measured_peak_bytes
    assert second.dry_run is not None and second.dry_run.gradients_match
    assert second.cache_stats.layer_hits["analysis"] == 1
    assert second.cache_stats.layer_hits["executable"] == 1
    assert second.cache_stats.hit_rate is not None and second.cache_stats.hit_rate > 0
    assert (tmp_path / "analysis").exists()
    assert (tmp_path / "executable").exists()
