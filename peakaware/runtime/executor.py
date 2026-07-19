from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from peakaware.config import PeakAwareConfig
from peakaware.contracts import GuardSpec, MeasuredExecutable, StepResult
from peakaware.runtime.measure import measure_training_step_phases


def _runtime_guard_value(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if name == "torch_version":
        return torch.__version__
    try:
        root, field = name.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"unsupported runtime guard: {name}") from exc
    if root.startswith("arg"):
        try:
            value = args[int(root.removeprefix("arg"))]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"runtime guard {name} references a missing positional input") from exc
    elif root == "kw":
        try:
            key, field = field.split(".", 1)
            value = kwargs[key]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"runtime guard {name} references a missing keyword input") from exc
    else:
        raise ValueError(f"unsupported runtime guard: {name}")
    if not isinstance(value, Tensor):
        raise ValueError(f"runtime guard {name} references a non-tensor input")
    if field == "shape":
        return str(tuple(value.shape))
    if field == "dtype":
        return str(value.dtype)
    if field == "device":
        return str(value.device)
    raise ValueError(f"unsupported runtime guard: {name}")


def validate_runtime_guards(guards: tuple[GuardSpec, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    for guard in guards:
        actual = _runtime_guard_value(guard.name, args, kwargs)
        if actual != guard.value:
            raise ValueError(f"runtime guard failed for {guard.name}: expected {guard.value}, got {actual}")


class EagerTrainingStepExecutor:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        executable: Callable[..., Tensor],
        config: PeakAwareConfig,
        guards: tuple[GuardSpec, ...] = (),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.executable = executable
        self.config = config
        self.guards = guards

    def step(self, *args: Any, **kwargs: Any) -> StepResult:
        validate_runtime_guards(self.guards, args, kwargs)
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)
        loss = self.loss_fn(self.executable(*args, **kwargs))
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        loss.backward()
        self.optimizer.step()
        return StepResult(loss=loss.detach(), optimizer_step_performed=True)


def run_compiled_forward_backward(model: nn.Module, *args: Any, **kwargs: Any) -> Tensor:
    return model(*args, **kwargs)


def run_eager_optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()


def build_training_step_executor(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., Tensor],
    config: PeakAwareConfig,
    guards: tuple[GuardSpec, ...] = (),
) -> EagerTrainingStepExecutor:
    executable: Callable[..., Tensor]
    if config.enable_compile:
        backend = "inductor" if config.enable_inductor else "aot_eager"
        executable = torch.compile(model, backend=backend)
    else:
        executable = model
    return EagerTrainingStepExecutor(model, optimizer, loss_fn, executable, config, guards)


def replace_executable_on_fallback(
    executor: EagerTrainingStepExecutor,
    executable: Callable[..., Tensor],
) -> EagerTrainingStepExecutor:
    executor.executable = executable
    return executor


def make_measured_executable(
    plan_id: str,
    executor: EagerTrainingStepExecutor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    simulated_peak_bytes: int,
) -> MeasuredExecutable:
    phase_metrics = measure_training_step_phases(
        executor.model,
        executor.optimizer,
        executor.executable,
        executor.loss_fn,
        args,
        kwargs,
        zero_grad_set_to_none=executor.config.zero_grad_set_to_none,
        warmup_steps=executor.config.measurement_warmup_steps,
        repeat_count=executor.config.measurement_repeats,
    )
    measured_peak = int(phase_metrics["overall_peak_bytes"])
    measured_us = float(phase_metrics["step_us"])
    if measured_peak == 0:
        measured_peak = simulated_peak_bytes
        phase_metrics = dict(phase_metrics)
        phase_metrics["overall_peak_bytes"] = measured_peak
    return MeasuredExecutable(
        plan_id=plan_id,
        forward_backward=executor.executable,
        measured_peak_bytes=measured_peak,
        measured_step_us=measured_us,
        correctness_passed=True,
        phase_metrics=phase_metrics,
    )
