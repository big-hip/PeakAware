from __future__ import annotations

import copy
import statistics
import time
from typing import Any, Callable

import torch
from torch import Tensor


def measure_training_step(fn: Callable[..., Tensor], *args: Any, **kwargs: Any) -> tuple[Tensor, int, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    return loss, peak, elapsed_us


def measure_phase_peaks(fn: Callable[..., Tensor], *args: Any, **kwargs: Any) -> dict[str, int | float]:
    loss, peak, elapsed = measure_training_step(fn, *args, **kwargs)
    del loss
    return {"overall_peak_bytes": peak, "step_us": elapsed}


def measure_compile_time(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1_000_000.0


def _cuda_allocated_peak_or_zero() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _cuda_reserved_peak_or_zero() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_reserved())


def _reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _clone_model_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _clone_grads(model: torch.nn.Module) -> tuple[Tensor | None, ...]:
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in model.parameters())


def _restore_grads(model: torch.nn.Module, grads: tuple[Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _aggregate_phase_metrics(samples: list[dict[str, int | float]], warmup_steps: int) -> dict[str, int | float]:
    if not samples:
        return {}
    keys = set().union(*(sample.keys() for sample in samples))
    aggregated: dict[str, int | float] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if key.endswith("_peak_bytes") or key == "overall_peak_bytes":
            aggregated[key] = int(max(values))
        else:
            aggregated[key] = float(statistics.median(values))
    step_values = [float(sample["step_us"]) for sample in samples]
    aggregated["step_us_median"] = float(statistics.median(step_values))
    aggregated["step_us_p10"] = _percentile(step_values, 0.10)
    aggregated["step_us_p90"] = _percentile(step_values, 0.90)
    aggregated["measurement_repeats"] = len(samples)
    aggregated["measurement_warmup_steps"] = warmup_steps
    return aggregated


def measure_training_step_phases(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
    warmup_steps: int = 0,
    repeat_count: int = 1,
) -> dict[str, int | float]:
    model_state = _clone_model_state(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    grad_state = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    warmup_steps = max(int(warmup_steps), 0)
    repeat_count = max(int(repeat_count), 1)

    def restore_state() -> None:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        _restore_grads(model, grad_state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def measure_once() -> dict[str, int | float]:
        metrics: dict[str, int | float] = {}
        restore_state()
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

        _reset_cuda_peak()
        start = time.perf_counter()
        output = executable(*args, **kwargs)
        loss = loss_fn(output)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["fw_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["fw_peak_bytes"] = _cuda_allocated_peak_or_zero()
        metrics["fw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero()

        _reset_cuda_peak()
        start = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["bw_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["bw_peak_bytes"] = _cuda_allocated_peak_or_zero()
        metrics["bw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero()

        _reset_cuda_peak()
        start = time.perf_counter()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["optimizer_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["optimizer_peak_bytes"] = _cuda_allocated_peak_or_zero()
        metrics["optimizer_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero()
        metrics["step_us"] = metrics["fw_us"] + metrics["bw_us"] + metrics["optimizer_us"]
        metrics["overall_peak_bytes"] = max(
            int(metrics["fw_peak_bytes"]),
            int(metrics["bw_peak_bytes"]),
            int(metrics["optimizer_peak_bytes"]),
        )
        metrics["overall_reserved_peak_bytes"] = max(
            int(metrics["fw_reserved_peak_bytes"]),
            int(metrics["bw_reserved_peak_bytes"]),
            int(metrics["optimizer_reserved_peak_bytes"]),
        )
        return metrics

    try:
        for _ in range(warmup_steps):
            measure_once()
        samples = [measure_once() for _ in range(repeat_count)]
        return _aggregate_phase_metrics(samples, warmup_steps)
    finally:
        restore_state()
