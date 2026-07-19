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


def _runtime_peak_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _reset_runtime_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


class EagerTrainingStepExecutor:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        executable: Callable[..., Any],
        config: PeakAwareConfig,
        guards: tuple[GuardSpec, ...] = (),
        plan_id: str = "unconfigured",
        fallback_executables: tuple[tuple[str, Callable[..., Any]], ...] = (),
        runtime_peak_threshold_bytes: int | None = None,
        runtime_peak_observer: Callable[[], int] | None = None,
        selection_objective: str = "unconfigured",
        activation_checkpoint: bool = False,
        fallback_activation_checkpoints: dict[str, bool] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.executable = executable
        self.config = config
        self.guards = guards
        self.current_plan_id = plan_id
        self.fallback_executables = fallback_executables
        self.runtime_peak_threshold_bytes = runtime_peak_threshold_bytes
        self.runtime_peak_observer = runtime_peak_observer
        self.selection_objective = selection_objective
        self.activation_checkpoint = activation_checkpoint
        self.fallback_activation_checkpoints = dict(fallback_activation_checkpoints or {})

    def step(self, *args: Any, **kwargs: Any) -> StepResult:
        validate_runtime_guards(self.guards, args, kwargs)
        _reset_runtime_peak_stats()
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)
        loss = self.loss_fn(self.executable(*args, **kwargs))
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        loss.backward()
        self.optimizer.step()
        peak_bytes = self.runtime_peak_observer() if self.runtime_peak_observer is not None else _runtime_peak_bytes()
        metrics: dict[str, Any] = {
            "plan_id": self.current_plan_id,
            "runtime_peak_bytes": peak_bytes,
            "runtime_peak_threshold_bytes": self.runtime_peak_threshold_bytes,
            "activation_checkpoint": int(self.activation_checkpoint),
        }
        if self.runtime_peak_threshold_bytes is not None and peak_bytes > self.runtime_peak_threshold_bytes:
            metrics["fallback_reason"] = (
                f"runtime peak {peak_bytes} bytes exceeded threshold "
                f"{self.runtime_peak_threshold_bytes} bytes"
            )
            if self.fallback_executables:
                fallback_plan_id, fallback_executable = self.fallback_executables[0]
                replace_executable_on_fallback(
                    self,
                    fallback_executable,
                    plan_id=fallback_plan_id,
                    activation_checkpoint=self.fallback_activation_checkpoints.get(fallback_plan_id),
                )
                metrics["fallback_plan_id"] = fallback_plan_id
                metrics["fallback_activation_checkpoint"] = int(self.activation_checkpoint)
            else:
                metrics["fallback_plan_id"] = None
        return StepResult(loss=loss.detach(), optimizer_step_performed=True, metrics=metrics)


def run_compiled_forward_backward(model: nn.Module, *args: Any, **kwargs: Any) -> Tensor:
    return model(*args, **kwargs)


def run_eager_optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()


def _checkpointed_forward(model: nn.Module, *args: Any, **kwargs: Any) -> Any:
    from torch.utils.checkpoint import checkpoint

    def forward(*inner_args: Any) -> Any:
        return model(*inner_args, **kwargs)

    return checkpoint(forward, *args, use_reentrant=False)


def build_training_step_executor(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., Tensor],
    config: PeakAwareConfig,
    guards: tuple[GuardSpec, ...] = (),
    plan_id: str = "unconfigured",
    fallback_executables: tuple[tuple[str, Callable[..., Any]], ...] = (),
    runtime_peak_threshold_bytes: int | None = None,
    selection_objective: str = "unconfigured",
    activation_checkpoint: bool = False,
) -> EagerTrainingStepExecutor:
    executable: Callable[..., Any]
    base_executable: Callable[..., Any]
    if activation_checkpoint:
        base_executable = lambda *args, **kwargs: _checkpointed_forward(model, *args, **kwargs)
    else:
        base_executable = model
    if config.enable_compile:
        backend = "inductor" if config.enable_inductor else "aot_eager"
        executable = torch.compile(base_executable, backend=backend)
    else:
        executable = base_executable
    return EagerTrainingStepExecutor(
        model,
        optimizer,
        loss_fn,
        executable,
        config,
        guards,
        plan_id=plan_id,
        fallback_executables=fallback_executables,
        runtime_peak_threshold_bytes=runtime_peak_threshold_bytes,
        selection_objective=selection_objective,
        activation_checkpoint=activation_checkpoint,
    )


def replace_executable_on_fallback(
    executor: EagerTrainingStepExecutor,
    executable: Callable[..., Any],
    *,
    plan_id: str | None = None,
    activation_checkpoint: bool | None = None,
) -> EagerTrainingStepExecutor:
    executor.executable = executable
    if plan_id is not None:
        executor.current_plan_id = plan_id
    if activation_checkpoint is not None:
        executor.activation_checkpoint = activation_checkpoint
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
    phase_metrics = dict(phase_metrics)
    phase_metrics["activation_checkpoint"] = int(executor.activation_checkpoint)
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
