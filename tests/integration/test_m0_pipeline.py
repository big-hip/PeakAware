import copy

import pytest
import torch
from torch import nn

import peakaware.api as api_module
from peakaware import PeakAwareConfig, optimize_training
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
    assert result.executable.plan_id in {candidate.plan_id for candidate in result.measured_candidates}
    assert result.executor.current_plan_id == result.selected_plan.plan_id
    assert result.executor.runtime_peak_threshold_bytes is not None
    assert result.executor.runtime_peak_threshold_bytes <= 1 << 28
    assert tuple(plan_id for plan_id, _ in result.executor.fallback_executables) == result.fallback_plan_ids
    assert result.dry_run is not None and result.dry_run.gradients_match
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
