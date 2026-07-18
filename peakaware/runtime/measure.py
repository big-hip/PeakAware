from __future__ import annotations

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
