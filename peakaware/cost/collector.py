from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from peakaware.cost.base import OpSignature
from peakaware.cost.profile_db import ProfileDB, ProfileRecord


@dataclass(frozen=True)
class MicrobenchmarkResult:
    signature: OpSignature
    samples_us: tuple[float, ...]
    workspace_bytes: int
    record: ProfileRecord


@dataclass(frozen=True)
class ModelTraceEvent:
    name: str
    cpu_time_total_us: float
    self_cpu_time_total_us: float
    cuda_time_total_us: float
    call_count: int


def _percentile(samples: tuple[float, ...], percentile: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def summarize_samples(samples_us: tuple[float, ...], *, workspace_bytes: int = 0) -> ProfileRecord:
    if not samples_us:
        raise ValueError("samples_us must not be empty")
    return ProfileRecord(
        signature_hash="",
        sample_count=len(samples_us),
        p50_us=float(statistics.median(samples_us)),
        p90_us=_percentile(samples_us, 0.90),
        mean_us=sum(samples_us) / len(samples_us),
        workspace_bytes=int(workspace_bytes),
        source="microbenchmark",
    )


def measure_cuda_events(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> tuple[float, int]:
    kwargs = dict(kwargs or {})
    if not torch.cuda.is_available():
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        return elapsed_us, 0
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    return (
        float(start_event.elapsed_time(end_event) * 1000.0),
        max(0, peak_allocated - baseline_allocated),
    )


def collect_microbenchmark(
    signature: OpSignature,
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 1,
    repeats: int = 10,
    db: ProfileDB | None = None,
    output_allocation_bytes: int | None = None,
) -> MicrobenchmarkResult:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    kwargs = dict(kwargs or {})
    for _ in range(warmup):
        fn(*args, **kwargs)
    samples: list[float] = []
    workspace = 0
    expected_output_bytes = (
        max(int(signature.output_bytes), 0)
        if output_allocation_bytes is None
        else max(int(output_allocation_bytes), 0)
    )
    for _ in range(repeats):
        elapsed_us, peak_increment_bytes = measure_cuda_events(fn, args, kwargs)
        samples.append(elapsed_us)
        workspace = max(workspace, max(0, peak_increment_bytes - expected_output_bytes))
    record = summarize_samples(tuple(samples), workspace_bytes=workspace)
    if db is not None:
        db.upsert_profile(signature, record)
    return MicrobenchmarkResult(signature=signature, samples_us=tuple(samples), workspace_bytes=workspace, record=record)


def collect_model_trace(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> tuple[ModelTraceEvent, ...]:
    kwargs = dict(kwargs or {})
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        fn(*args, **kwargs)
    events = [
        ModelTraceEvent(
            name=item.key,
            cpu_time_total_us=float(item.cpu_time_total),
            self_cpu_time_total_us=float(item.self_cpu_time_total),
            cuda_time_total_us=float(getattr(item, "cuda_time_total", 0.0)),
            call_count=int(item.count),
        )
        for item in profiler.key_averages()
    ]
    events.sort(key=lambda event: (-event.cpu_time_total_us, event.name))
    return tuple(events)
