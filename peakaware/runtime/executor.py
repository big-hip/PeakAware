from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from peakaware.config import PeakAwareConfig
from peakaware.contracts import MeasuredExecutable, StepResult
from peakaware.runtime.measure import measure_training_step


class EagerTrainingStepExecutor:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        executable: Callable[..., Tensor],
        config: PeakAwareConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.executable = executable
        self.config = config

    def step(self, *args: Any, **kwargs: Any) -> StepResult:
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
) -> EagerTrainingStepExecutor:
    executable: Callable[..., Tensor]
    if config.enable_compile:
        backend = "inductor" if config.enable_inductor else "aot_eager"
        executable = torch.compile(model, backend=backend)
    else:
        executable = model
    return EagerTrainingStepExecutor(model, optimizer, loss_fn, executable, config)


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
    def one_forward(*step_args: Any, **step_kwargs: Any) -> Tensor:
        with torch.no_grad():
            output = executor.executable(*step_args, **step_kwargs)
            if isinstance(output, Tensor):
                return output.detach().sum()
            return torch.as_tensor(0)

    _, measured_peak, measured_us = measure_training_step(one_forward, *args, **kwargs)
    if measured_peak == 0:
        measured_peak = simulated_peak_bytes
    return MeasuredExecutable(
        plan_id=plan_id,
        forward_backward=executor.executable,
        measured_peak_bytes=measured_peak,
        measured_step_us=measured_us,
        correctness_passed=True,
    )
