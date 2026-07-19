from types import SimpleNamespace

import pytest
import torch
from torch import nn

from peakaware import PeakAwareConfig
from peakaware.contracts import GuardSpec
from peakaware.runtime.executor import (
    EagerTrainingStepExecutor,
    build_training_step_executor,
    make_measured_executable,
    validate_runtime_guards,
)


def test_runtime_guards_validate_positional_and_keyword_tensors():
    args = (torch.ones(2, 3),)
    kwargs = {"mask": torch.ones(2, 3, dtype=torch.bool)}
    guards = (
        GuardSpec("arg0.shape", "(2, 3)"),
        GuardSpec("arg0.dtype", "torch.float32"),
        GuardSpec("arg0.device", "cpu"),
        GuardSpec("kw.mask.shape", "(2, 3)"),
        GuardSpec("kw.mask.dtype", "torch.bool"),
    )

    validate_runtime_guards(guards, args, kwargs)


def test_runtime_guards_reject_shape_drift():
    guards = (GuardSpec("arg0.shape", "(2, 3)"),)

    with pytest.raises(ValueError, match="runtime guard failed"):
        validate_runtime_guards(guards, (torch.ones(4, 3),), {})


def test_executor_rejects_guard_drift_before_zero_grad_or_step():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        guards=(GuardSpec("arg0.shape", "(2, 3)"),),
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    params_before = tuple(param.detach().clone() for param in model.parameters())
    grads_before = tuple(param.grad.detach().clone() for param in model.parameters())

    with pytest.raises(ValueError, match="runtime guard failed"):
        executor.step(torch.ones(4, 3))

    assert all(torch.equal(before, after) for before, after in zip(params_before, model.parameters()))
    assert all(torch.equal(before, param.grad) for before, param in zip(grads_before, model.parameters()))


def test_activation_checkpoint_executor_runs_and_reports_plan_marker():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        activation_checkpoint=True,
    )

    measured = make_measured_executable("checkpointed", executor, (torch.ones(2, 3),), {}, simulated_peak_bytes=1024)

    assert executor.activation_checkpoint is True
    assert measured.correctness_passed
    assert measured.phase_metrics["activation_checkpoint"] == 1


def test_activation_checkpoint_executor_preserves_object_outputs():
    torch.manual_seed(0)

    class LogitsModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x):
            return SimpleNamespace(logits=self.linear(x))

    model = LogitsModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.logits.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        activation_checkpoint=True,
    )

    step = executor.step(torch.ones(2, 3))

    assert step.optimizer_step_performed
    assert executor.activation_checkpoint is True


def test_executor_switches_to_verified_fallback_after_runtime_peak_breach():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = {"selected": 0, "fallback": 0}

    def selected(x):
        calls["selected"] += 1
        return model(x)

    def fallback(x):
        calls["fallback"] += 1
        return model(x)

    executor = EagerTrainingStepExecutor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        selected,
        PeakAwareConfig(enable_compile=False),
        plan_id="selected",
        fallback_executables=(("fallback", fallback),),
        runtime_peak_threshold_bytes=5,
        runtime_peak_observer=lambda: 10,
    )
    optimizer_id = id(executor.optimizer)

    first = executor.step(torch.ones(2, 3))
    executor.runtime_peak_observer = lambda: 0
    second = executor.step(torch.ones(2, 3))

    assert first.metrics["fallback_plan_id"] == "fallback"
    assert "exceeded threshold" in first.metrics["fallback_reason"]
    assert executor.current_plan_id == "fallback"
    assert calls == {"selected": 1, "fallback": 1}
    assert second.metrics["plan_id"] == "fallback"
    assert id(executor.optimizer) == optimizer_id
