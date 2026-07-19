import copy

import pytest
import torch
from torch import nn

import peakaware.api as api_module
from peakaware import PeakAwareConfig, optimize_training
from peakaware.contracts import MeasuredExecutable
from peakaware.runtime.isolation import WorkerResult


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        residual = self.a(x).relu()
        hidden = self.b(residual).relu()
        return self.out(hidden + residual)


def squared_mean_loss(out):
    return out.pow(2).mean()


def test_measured_candidate_selection_prefers_peak_by_default():
    fast_high_peak = MeasuredExecutable("fast_high_peak", abs, 200, 1.0, True)
    slow_low_peak = MeasuredExecutable("slow_low_peak", abs, 100, 10.0, True)

    default_selected = api_module._select_measured_candidate(
        (fast_high_peak, slow_low_peak),
        memory_budget_bytes=1 << 20,
        selection_objective="min_peak_then_time",
    )
    time_selected = api_module._select_measured_candidate(
        (fast_high_peak, slow_low_peak),
        memory_budget_bytes=1 << 20,
        selection_objective="min_time_then_peak",
    )

    assert default_selected is slow_low_peak
    assert time_selected is fast_high_peak


def test_optimize_training_builds_executor_and_runs_step():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    result = optimize_training(
        model,
        (x,),
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )

    before = tuple(p.detach().clone() for p in model.parameters())
    step = result.executor.step(x)
    after = tuple(p.detach().clone() for p in model.parameters())

    assert step.optimizer_step_performed is True
    assert step.loss.ndim == 0
    assert result.executable.phase_metrics["step_us"] > 0
    assert {
        "fw_us",
        "bw_us",
        "optimizer_us",
        "overall_peak_bytes",
        "overall_reserved_peak_bytes",
        "step_us_median",
        "step_us_p10",
        "step_us_p90",
    }.issubset(result.executable.phase_metrics)
    assert len(result.measured_candidates) >= 2
    assert any(
        candidate.plan_id != "all_save"
        and (
            candidate.phase_metrics.get("activation_checkpoint") == 1
            or candidate.phase_metrics.get("aot_partition_runtime") == 1
        )
        for candidate in result.measured_candidates
    )
    assert result.executable.plan_id in {candidate.plan_id for candidate in result.measured_candidates}
    assert result.executor.current_plan_id == result.selected_plan.plan_id
    assert result.executor.activation_checkpoint == bool(result.executable.phase_metrics.get("activation_checkpoint", 0))
    assert result.executor.aot_partition_runtime == bool(result.executable.phase_metrics.get("aot_partition_runtime", 0))
    assert result.executor.selection_objective == "min_peak_then_time"
    assert result.executor.runtime_peak_threshold_bytes is not None
    assert result.executor.runtime_peak_threshold_bytes <= 1 << 28
    assert tuple(plan_id for plan_id, _ in result.executor.fallback_executables) == result.fallback_plan_ids
    measured_checkpoint_by_plan = {
        candidate.plan_id: bool(candidate.phase_metrics.get("activation_checkpoint", 0))
        for candidate in result.measured_candidates
    }
    assert result.executor.fallback_activation_checkpoints == {
        plan_id: measured_checkpoint_by_plan[plan_id]
        for plan_id in result.fallback_plan_ids
    }
    measured_aot_runtime_by_plan = {
        candidate.plan_id: bool(candidate.phase_metrics.get("aot_partition_runtime", 0))
        for candidate in result.measured_candidates
    }
    assert result.executor.fallback_aot_partition_runtimes == {
        plan_id: measured_aot_runtime_by_plan[plan_id]
        for plan_id in result.fallback_plan_ids
    }
    assert result.dry_run is not None and result.dry_run.gradients_match
    assert result.dry_run.replay_mode in {"lowered_aot", "eager_baseline"}
    assert result.analysis is not None and result.analysis.ir.values
    assert result.analysis is not None and result.analysis.ir.graph_key == result.selected_plan.graph_key
    assert any(not torch.equal(left, right) for left, right in zip(before, after))


def test_optimize_training_can_isolate_candidate_measurement():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    result = optimize_training(
        model,
        (x,),
        loss_fn=squared_mean_loss,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            enable_compile=False,
            top_k=2,
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            isolate_candidate_measurement=True,
            candidate_worker_timeout_s=30.0,
        ),
    )

    assert result.executable.correctness_passed
    assert result.dry_run is not None and result.dry_run.gradients_match
    assert result.measured_candidates
    assert result.executable.plan_id in {candidate.plan_id for candidate in result.measured_candidates}


def test_optimize_training_rejects_requires_grad_kwargs():
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="without requires_grad"):
        optimize_training(
            model,
            (x,),
            example_kwargs={"unused": torch.randn(4, 8, requires_grad=True)},
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False),
        )


def test_optimize_training_rejects_mixed_input_devices():
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="single device"):
        optimize_training(
            model,
            (x, {"unused": torch.empty(1, device="meta")}),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False),
        )


def test_optimize_training_rejects_mixed_optimizer_state_devices():
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    first_param = next(model.parameters())
    optimizer.state[first_param]["offloaded_state"] = torch.empty(1, device="meta")
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="single device"):
        optimize_training(
            model,
            (x,),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False),
        )


def test_optimize_training_rejects_non_configured_floating_dtype():
    model = TinyResidual().double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, dtype=torch.float64)

    with pytest.raises(ValueError, match="precision_dtype=float32"):
        optimize_training(
            model,
            (x,),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False),
        )


def test_optimize_training_rejects_autocast_and_grad_scaler_modes():
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="autocast"):
        optimize_training(
            model,
            (x,),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False, autocast_enabled=True),
        )
    with pytest.raises(ValueError, match="GradScaler"):
        optimize_training(
            model,
            (x,),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=PeakAwareConfig(enable_compile=False, grad_scaler_enabled=True),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (PeakAwareConfig(enable_compile=False, dynamic_shapes={"arg0": "batch"}), "dynamic shape"),
        (PeakAwareConfig(enable_compile=False, gradient_accumulation_steps=2), "gradient accumulation"),
        (PeakAwareConfig(enable_compile=False, fsdp_enabled=True), "FSDP"),
        (PeakAwareConfig(enable_compile=False, offload_enabled=True), "offload"),
        (PeakAwareConfig(enable_compile=False, selection_objective="unknown"), "selection_objective"),
    ),
)
def test_optimize_training_rejects_unsupported_execution_modes(config, message):
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match=message):
        optimize_training(
            model,
            (x,),
            loss_fn=squared_mean_loss,
            optimizer=optimizer,
            memory_budget_bytes=1 << 28,
            config=config,
        )


def test_request_key_includes_precision_configuration():
    model = TinyResidual()
    x = torch.randn(4, 8)

    default_key = api_module._request_key(
        model,
        (x,),
        {},
        1 << 28,
        PeakAwareConfig(enable_compile=False),
    )
    double_key = api_module._request_key(
        model,
        (x,),
        {},
        1 << 28,
        PeakAwareConfig(enable_compile=False, precision_dtype="float64"),
    )
    autocast_key = api_module._request_key(
        model,
        (x,),
        {},
        1 << 28,
        PeakAwareConfig(enable_compile=False, autocast_enabled=True),
    )

    assert len({default_key, double_key, autocast_key}) == 3


def test_isolated_candidate_failure_falls_back_to_next_candidate(monkeypatch):
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    calls = {"count": 0}

    def fake_worker(fn, payload, *, timeout_s):
        del timeout_s
        calls["count"] += 1
        if calls["count"] == 1:
            return WorkerResult(ok=False, error_type="RuntimeError", message="synthetic worker failure")
        return WorkerResult(ok=True, value=fn(payload))

    monkeypatch.setattr(api_module, "run_in_worker_process", fake_worker)

    result = optimize_training(
        model,
        (x,),
        loss_fn=squared_mean_loss,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            enable_compile=False,
            top_k=2,
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            isolate_candidate_measurement=True,
        ),
    )

    assert calls["count"] > 1
    assert result.executable.correctness_passed
    assert result.executable.plan_id != "all_save"


def test_optimize_training_does_not_advance_user_state_before_executor_step():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    before = tuple(p.detach().clone() for p in model.parameters())
    optimizer_before = optimizer.state_dict()

    result = optimize_training(
        model,
        (x,),
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )
    after_optimize = tuple(p.detach().clone() for p in model.parameters())

    assert result.executable.correctness_passed
    assert result.measured_candidates
    assert all(torch.equal(left, right) for left, right in zip(before, after_optimize))
    assert optimizer.state_dict()["state"] == optimizer_before["state"]


def test_executor_multistep_parameters_match_eager_training():
    torch.manual_seed(0)
    eager_model = TinyResidual()
    peakaware_model = copy.deepcopy(eager_model)
    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=0.01)
    peakaware_optimizer = torch.optim.SGD(peakaware_model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    result = optimize_training(
        peakaware_model,
        (x,),
        loss_fn=squared_mean_loss,
        optimizer=peakaware_optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )

    for _ in range(3):
        eager_optimizer.zero_grad(set_to_none=True)
        eager_loss = squared_mean_loss(eager_model(x))
        eager_loss.backward()
        eager_optimizer.step()

        step = result.executor.step(x)

        assert torch.allclose(step.loss, eager_loss.detach(), atol=1e-6, rtol=1e-5)
        for eager_param, peakaware_param in zip(eager_model.parameters(), peakaware_model.parameters()):
            assert torch.allclose(eager_param, peakaware_param, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Inductor steady-state validation")
def test_inductor_cuda_executor_runs_step_without_advancing_state_during_optimization():
    torch.manual_seed(0)
    model = TinyResidual().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, device="cuda")
    before = tuple(p.detach().cpu().clone() for p in model.parameters())

    result = optimize_training(
        model,
        (x,),
        loss_fn=squared_mean_loss,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            enable_compile=True,
            enable_inductor=True,
            top_k=1,
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            measurement_repeats=1,
        ),
    )
    after_optimize = tuple(p.detach().cpu().clone() for p in model.parameters())
    step = result.executor.step(x)

    assert result.executable.correctness_passed
    assert result.executable.phase_metrics["step_us"] > 0
    assert all(torch.equal(left, right) for left, right in zip(before, after_optimize))
    assert step.optimizer_step_performed is True
    assert step.metrics["plan_id"] == result.selected_plan.plan_id
    assert step.metrics["runtime_peak_bytes"] > 0


def test_optimize_training_restores_existing_adamw_optimizer_state():
    torch.manual_seed(0)
    model = TinyResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)

    optimizer.zero_grad(set_to_none=True)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    state_before = copy.deepcopy(optimizer.state_dict())
    params_before = tuple(p.detach().clone() for p in model.parameters())

    result = optimize_training(
        model,
        (x,),
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(enable_compile=False, top_k=3, safety_margin_bytes=0, safety_margin_ratio=0.0),
    )
    state_after = optimizer.state_dict()

    assert result.executable.phase_metrics["optimizer_us"] > 0
    assert all(torch.equal(left, right) for left, right in zip(params_before, model.parameters()))
    assert state_after["param_groups"] == state_before["param_groups"]
    for param_id, before_state in state_before["state"].items():
        after_state = state_after["state"][param_id]
        for name, before_value in before_state.items():
            after_value = after_state[name]
            if isinstance(before_value, torch.Tensor):
                assert torch.equal(before_value, after_value)
            else:
                assert before_value == after_value
