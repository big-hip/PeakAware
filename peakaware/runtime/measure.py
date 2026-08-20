from __future__ import annotations

import copy
import functools
import inspect
import shutil
import statistics
import subprocess
import threading
import time
import types
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils import _pytree

from .stable_callable import (
    StableCallableSnapshot,
    snapshot_registered_callable,
    validate_registered_callable,
)

DEFAULT_MEMORY_SAMPLER_INTERVAL_US = 500.0
DEFAULT_GPU_UTIL_SAMPLER_INTERVAL_US = 50_000.0


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


def _cuda_allocated_peak_or_zero(cuda_device: torch.device | None = None) -> int:
    if cuda_device is None:
        return 0
    return int(torch.cuda.max_memory_allocated(cuda_device))


def _cuda_reserved_peak_or_zero(cuda_device: torch.device | None = None) -> int:
    if cuda_device is None:
        return 0
    return int(torch.cuda.max_memory_reserved(cuda_device))


def _cuda_allocated_or_zero(cuda_device: torch.device | None = None) -> int:
    if cuda_device is None:
        return 0
    return int(torch.cuda.memory_allocated(cuda_device))


def _cuda_reserved_or_zero(cuda_device: torch.device | None = None) -> int:
    if cuda_device is None:
        return 0
    return int(torch.cuda.memory_reserved(cuda_device))


def _reset_cuda_peak(cuda_device: torch.device | None = None) -> None:
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)


class _CudaMemorySampler:
    def __init__(
        self,
        cuda_device: torch.device | None,
        *,
        interval_us: float = DEFAULT_MEMORY_SAMPLER_INTERVAL_US,
    ) -> None:
        self.cuda_device = cuda_device
        self.interval_s = max(float(interval_us), 1.0) / 1_000_000.0
        self._phase = "unknown"
        self._running = False
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._peak_bytes = 0

    def start(self, phase: str = "start") -> None:
        if self.cuda_device is None:
            return
        with torch.cuda.device(self.cuda_device):
            torch.cuda.synchronize(self.cuda_device)
        with self._lock:
            self._phase = phase
            self._running = True
            self._started_at = time.perf_counter()
        self._record_sample(event="sampler_start")
        self._thread = threading.Thread(target=self._run, name="peakaware-cuda-memory-sampler", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        if self.cuda_device is None:
            return
        self._record_sample(event="phase_end")
        with self._lock:
            self._phase = phase
        self._record_sample(event="phase_start")

    def stop(self) -> tuple[dict[str, Any], ...]:
        if self.cuda_device is None:
            return ()
        self._record_sample(event="sampler_stop_request")
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record_sample(event="sampler_stop")
        with self._lock:
            return tuple(self._samples)

    def _run(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            self._record_sample()
            time.sleep(self.interval_s)

    def _record_sample(self, *, event: str = "sample") -> None:
        if self.cuda_device is None:
            return
        try:
            allocated = int(torch.cuda.memory_allocated(self.cuda_device))
            reserved = int(torch.cuda.memory_reserved(self.cuda_device))
            allocator_peak = int(torch.cuda.max_memory_allocated(self.cuda_device))
            allocator_reserved_peak = int(torch.cuda.max_memory_reserved(self.cuda_device))
        except Exception:
            return
        with self._lock:
            phase = self._phase
            started_at = self._started_at
            self._peak_bytes = max(self._peak_bytes, allocated)
            sampled_peak = self._peak_bytes
        row = {
            "phase": phase,
            "event": event,
            "time_us": (time.perf_counter() - self._started_at) * 1_000_000.0,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "sampled_peak_bytes": sampled_peak,
            "peak_bytes": allocator_peak,
            "allocator_peak_bytes": allocator_peak,
            "allocator_reserved_peak_bytes": allocator_reserved_peak,
        }
        with self._lock:
            if started_at == self._started_at and (not self._samples or row != self._samples[-1]):
                self._samples.append(row)


class _GpuUtilizationSampler:
    def __init__(
        self,
        cuda_device: torch.device | None,
        *,
        interval_us: float = DEFAULT_GPU_UTIL_SAMPLER_INTERVAL_US,
    ) -> None:
        self.cuda_device = cuda_device
        self.interval_s = max(float(interval_us), 1_000.0) / 1_000_000.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._unavailable_reason: str | None = None
        self._binary = shutil.which("nvidia-smi")

    def start(self) -> None:
        if self.cuda_device is None:
            self._unavailable_reason = "cuda_device_unavailable"
            return
        if self._binary is None:
            self._unavailable_reason = "nvidia_smi_unavailable"
            return
        with self._lock:
            self._running = True
            self._started_at = time.perf_counter()
        self._record_sample(event="sampler_start")
        self._thread = threading.Thread(target=self._run, name="peakaware-gpu-util-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        self._record_sample(event="sampler_stop_request")
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record_sample(event="sampler_stop")
        with self._lock:
            trace = tuple(self._samples)
            reason = self._unavailable_reason
        return trace, _gpu_compute_summary(trace, unavailable_reason=reason)

    def _run(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            self._record_sample()
            time.sleep(self.interval_s)

    def _record_sample(self, *, event: str = "sample") -> None:
        if self.cuda_device is None or self._binary is None:
            return
        device_index = (
            self.cuda_device.index if self.cuda_device.index is not None else torch.cuda.current_device()
        )
        command = [
            self._binary,
            f"--id={device_index}",
            (
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                "power.draw,clocks.sm,clocks.mem"
            ),
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1.0)
            fields = [item.strip() for item in completed.stdout.strip().split(",")]
            if len(fields) < 6:
                raise ValueError("unexpected nvidia-smi output")
            row = {
                "event": event,
                "time_us": (time.perf_counter() - self._started_at) * 1_000_000.0,
                "device_index": int(device_index),
                "gpu_util_percent": _parse_float_or_none(fields[0]),
                "memory_util_percent": _parse_float_or_none(fields[1]),
                "memory_used_mib": _parse_float_or_none(fields[2]),
                "power_w": _parse_float_or_none(fields[3]),
                "sm_clock_mhz": _parse_float_or_none(fields[4]),
                "mem_clock_mhz": _parse_float_or_none(fields[5]),
                "source": "nvidia-smi",
            }
        except Exception as exc:
            with self._lock:
                if self._unavailable_reason is None:
                    self._unavailable_reason = f"nvidia_smi_query_failed:{type(exc).__name__}"
            return
        with self._lock:
            if not self._samples or row != self._samples[-1]:
                self._samples.append(row)


def _parse_float_or_none(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _max_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else max(present)


def _gpu_compute_summary(
    trace: tuple[dict[str, Any], ...],
    *,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    util_values = [point.get("gpu_util_percent") for point in trace]
    memory_values = [point.get("memory_util_percent") for point in trace]
    power_values = [point.get("power_w") for point in trace]
    return {
        "status": "ok" if trace else "unavailable",
        "source": "nvidia-smi" if trace else None,
        "sample_count": len(trace),
        "interval_us": DEFAULT_GPU_UTIL_SAMPLER_INTERVAL_US,
        "mean_gpu_util_percent": _mean_or_none(util_values),
        "max_gpu_util_percent": _max_or_none(util_values),
        "mean_memory_util_percent": _mean_or_none(memory_values),
        "max_memory_util_percent": _max_or_none(memory_values),
        "mean_power_w": _mean_or_none(power_values),
        "max_power_w": _max_or_none(power_values),
        "unavailable_reason": unavailable_reason if not trace else None,
    }


def _measurement_cuda_device(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.device | None:
    devices: set[torch.device] = set()

    leaves, _ = _pytree.tree_flatten(
        (
            tuple(model.parameters()),
            tuple(model.buffers()),
            optimizer.state_dict(),
            args,
            kwargs,
        )
    )
    for value in leaves:
        if isinstance(value, Tensor) and value.device.type == "cuda":
            index = value.device.index if value.device.index is not None else torch.cuda.current_device()
            devices.add(torch.device("cuda", index))
    if len(devices) > 1:
        raise ValueError("publication measurement requires a single CUDA device")
    return next(iter(devices), None)


def _clone_pytree(value: Any) -> Any:
    tensor_memo: dict[int, Tensor] = {}

    def clone_leaf(leaf: Any) -> Any:
        if isinstance(leaf, Tensor):
            cached = tensor_memo.get(id(leaf))
            if cached is not None:
                return cached
            cloned = leaf.detach().clone(memory_format=torch.preserve_format)
            cloned.requires_grad_(leaf.requires_grad)
            tensor_memo[id(leaf)] = cloned
            return cloned
        try:
            return copy.deepcopy(leaf)
        except Exception as exc:
            raise ValueError(f"measurement input leaf cannot be snapshotted: {type(leaf).__name__}") from exc

    return _pytree.tree_map(clone_leaf, value)


def _clone_model_state(model: torch.nn.Module) -> dict[str, Tensor]:
    # Restoration snapshots must not remain resident in the CUDA allocator
    # during the measured step. Keeping model/optimizer/grad clones on-device
    # inflates the reported peak by several complete parameter copies.
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def _clone_grads(model: torch.nn.Module) -> tuple[Tensor | None, ...]:
    return tuple(
        None if p.grad is None else p.grad.detach().to(device="cpu", copy=True)
        for p in model.parameters()
    )


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    def clone_leaf(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().to(device="cpu", copy=True)
        return copy.deepcopy(value)

    return _pytree.tree_map(clone_leaf, optimizer.state_dict())


def _restore_grads(model: torch.nn.Module, grads: tuple[Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = (
            None
            if grad is None
            else grad.detach().to(device=param.device, dtype=param.dtype, copy=True)
        )


@dataclass(frozen=True)
class _PythonAttributeState:
    module: torch.nn.Module
    name: str
    original: Any
    snapshot: Any


@dataclass(frozen=True)
class _TrainingState:
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    grads: tuple[Tensor | None, ...]
    training_modes: tuple[bool, ...]
    cpu_rng: Tensor
    cuda_rng: Tensor | None
    python_attributes: tuple[_PythonAttributeState, ...] = ()
    public_attribute_names: tuple[tuple[torch.nn.Module, frozenset[str]], ...] = ()


@dataclass(frozen=True)
class _PartialCallableSnapshot:
    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]


_IMMUTABLE_PYTHON_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _is_torch_stateless_callable(value: Any) -> bool:
    module_name = str(getattr(value, "__module__", ""))
    if not module_name.startswith("torch."):
        return False
    return (
        type(value) is types.BuiltinFunctionType
        or inspect.isclass(value)
    )


def _snapshot_python_value(value: Any, memo: dict[int, Any] | None = None) -> Any:
    if type(value) in _IMMUTABLE_PYTHON_TYPES:
        return value
    if _is_torch_stateless_callable(value):
        return value
    if memo is None:
        memo = {}
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if isinstance(value, Tensor):
        snapshot = value.detach().to(device="cpu", copy=True)
        snapshot.requires_grad_(value.requires_grad)
        memo[id(value)] = snapshot
        return snapshot
    callable_snapshot = snapshot_registered_callable(value)
    if callable_snapshot is not None:
        return callable_snapshot
    if isinstance(value, functools.partial) and _is_torch_stateless_callable(value.func):
        snapshot = _PartialCallableSnapshot(
            value.func,
            tuple(_snapshot_python_value(item, memo) for item in value.args),
            {
                str(key): _snapshot_python_value(item, memo)
                for key, item in (value.keywords or {}).items()
            },
        )
        memo[id(value)] = snapshot
        return snapshot
    if type(value) is list:
        snapshot_list: list[Any] = []
        memo[id(value)] = snapshot_list
        snapshot_list.extend(_snapshot_python_value(item, memo) for item in value)
        return snapshot_list
    if type(value) is dict:
        snapshot_dict: dict[Any, Any] = {}
        memo[id(value)] = snapshot_dict
        for key, item in value.items():
            snapshot_dict[_snapshot_python_value(key, memo)] = _snapshot_python_value(item, memo)
        return snapshot_dict
    if type(value) is set:
        snapshot_set: set[Any] = set()
        memo[id(value)] = snapshot_set
        snapshot_set.update(_snapshot_python_value(item, memo) for item in value)
        return snapshot_set
    if type(value) is tuple:
        snapshot_tuple = tuple(_snapshot_python_value(item, memo) for item in value)
        memo[id(value)] = snapshot_tuple
        return snapshot_tuple
    if is_dataclass(value) and not isinstance(value, type):
        snapshot_dataclass = object.__new__(type(value))
        memo[id(value)] = snapshot_dataclass
        for field in fields(value):
            object.__setattr__(
                snapshot_dataclass,
                field.name,
                _snapshot_python_value(getattr(value, field.name), memo),
            )
        return snapshot_dataclass
    raise ValueError(f"unsupported mutable Python attribute type: {type(value).__name__}")


def _restore_python_value(original: Any, snapshot: Any) -> Any:
    if type(snapshot) in _IMMUTABLE_PYTHON_TYPES:
        return snapshot
    if _is_torch_stateless_callable(snapshot):
        return snapshot
    if isinstance(snapshot, Tensor):
        if not isinstance(original, Tensor):
            return _snapshot_python_value(snapshot)
        with torch.no_grad():
            original.copy_(snapshot)
        return original
    if isinstance(snapshot, StableCallableSnapshot):
        validate_registered_callable(original, snapshot)
        return original
    if isinstance(snapshot, _PartialCallableSnapshot):
        return functools.partial(
            snapshot.func,
            *(_snapshot_python_value(item) for item in snapshot.args),
            **{
                key: _snapshot_python_value(item)
                for key, item in snapshot.keywords.items()
            },
        )
    if type(snapshot) is list:
        if type(original) is not list:
            return _snapshot_python_value(snapshot)
        restored = [
            _restore_python_value(original[index], item) if index < len(original) else _snapshot_python_value(item)
            for index, item in enumerate(snapshot)
        ]
        original[:] = restored
        return original
    if type(snapshot) is dict:
        if type(original) is not dict:
            return _snapshot_python_value(snapshot)
        restored_dict: dict[Any, Any] = {}
        for key, item in snapshot.items():
            restored_key = _snapshot_python_value(key)
            if key in original:
                restored_dict[restored_key] = _restore_python_value(original[key], item)
            else:
                restored_dict[restored_key] = _snapshot_python_value(item)
        original.clear()
        original.update(restored_dict)
        return original
    if type(snapshot) is set:
        if type(original) is not set:
            return _snapshot_python_value(snapshot)
        original.clear()
        original.update(_snapshot_python_value(item) for item in snapshot)
        return original
    if type(snapshot) is tuple:
        if type(original) is tuple and len(original) == len(snapshot):
            return tuple(_restore_python_value(left, right) for left, right in zip(original, snapshot))
        return _snapshot_python_value(snapshot)
    if is_dataclass(snapshot) and not isinstance(snapshot, type):
        target = original if type(original) is type(snapshot) else object.__new__(type(snapshot))
        for field in fields(snapshot):
            previous = getattr(target, field.name, None)
            object.__setattr__(
                target,
                field.name,
                _restore_python_value(previous, getattr(snapshot, field.name)),
            )
        return target
    raise ValueError(f"unsupported Python snapshot type: {type(snapshot).__name__}")


def _capture_python_attributes(model: torch.nn.Module) -> tuple[
    tuple[_PythonAttributeState, ...],
    tuple[tuple[torch.nn.Module, frozenset[str]], ...],
]:
    captured: list[_PythonAttributeState] = []
    names_by_module: list[tuple[torch.nn.Module, frozenset[str]]] = []
    for module_name, module in model.named_modules():
        names = frozenset(name for name in module.__dict__ if not name.startswith("_") and name != "training")
        names_by_module.append((module, names))
        for name in names:
            value = module.__dict__[name]
            try:
                snapshot = _snapshot_python_value(value)
            except Exception as exc:
                qualified_name = f"{module_name}.{name}" if module_name else name
                raise ValueError(
                    f"publication measurement cannot snapshot Python module attribute {qualified_name}"
                ) from exc
            captured.append(_PythonAttributeState(module, name, value, snapshot))
    return tuple(captured), tuple(names_by_module)


def _restore_python_attributes(state: _TrainingState) -> None:
    for module, original_names in state.public_attribute_names:
        current_names = {
            name for name in module.__dict__ if not name.startswith("_") and name != "training"
        }
        for name in current_names - original_names:
            delattr(module, name)
    for attribute in state.python_attributes:
        if isinstance(attribute.snapshot, StableCallableSnapshot):
            current = attribute.module.__dict__.get(attribute.name)
            validate_registered_callable(current, attribute.snapshot)
        restored = _restore_python_value(attribute.original, attribute.snapshot)
        setattr(attribute.module, attribute.name, restored)


def _capture_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cuda_device: torch.device | None,
    *,
    include_python_state: bool = False,
) -> _TrainingState:
    python_attributes, public_attribute_names = (
        _capture_python_attributes(model) if include_python_state else ((), ())
    )
    return _TrainingState(
        model=_clone_model_state(model),
        optimizer=_clone_optimizer_state(optimizer),
        grads=_clone_grads(model),
        training_modes=tuple(module.training for module in model.modules()),
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=torch.cuda.get_rng_state(cuda_device).clone() if cuda_device is not None else None,
        python_attributes=python_attributes,
        public_attribute_names=public_attribute_names,
    )


def _restore_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: _TrainingState,
    cuda_device: torch.device | None,
) -> None:
    model.load_state_dict(state.model)
    optimizer.load_state_dict(copy.deepcopy(state.optimizer))
    _restore_grads(model, state.grads)
    for module, training in zip(model.modules(), state.training_modes):
        module.training = training
    torch.set_rng_state(state.cpu_rng)
    if state.cuda_rng is not None:
        if cuda_device is None:
            raise RuntimeError("captured CUDA RNG state requires its original device")
        torch.cuda.set_rng_state(state.cuda_rng, cuda_device)
    _restore_python_attributes(state)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _median_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _timing_sample(
    prefix: str,
    wall_us: float,
    event_us: float | None,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> dict[str, float | bool | None]:
    if event_us is None:
        absolute_difference = None
        relative_difference = None
    else:
        absolute_difference = abs(wall_us - event_us)
        relative_difference = absolute_difference / event_us if event_us > 0 else None
    timing_qualified = (
        (
            relative_difference <= max_event_wall_relative_gap
            or absolute_difference <= max_event_wall_absolute_gap_us
        )
        if relative_difference is not None and absolute_difference is not None
        else None
    )
    return {
        f"{prefix}_wall_us": wall_us,
        f"{prefix}_event_us": event_us,
        f"{prefix}_event_wall_abs_diff_us": absolute_difference,
        f"{prefix}_event_wall_relative_diff": relative_difference,
        f"{prefix}_timing_qualified": timing_qualified,
    }


def _aggregate_timing_qualified(
    relative_difference: float | None,
    absolute_difference: float | None,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> bool | None:
    if relative_difference is None or absolute_difference is None:
        return None
    return (
        relative_difference <= max_event_wall_relative_gap
        or absolute_difference <= max_event_wall_absolute_gap_us
    )


def _relative_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = statistics.mean(values)
    if center <= 0:
        return None
    x_center = (len(values) - 1) / 2.0
    denominator = sum((index - x_center) ** 2 for index in range(len(values)))
    slope = sum((index - x_center) * (value - center) for index, value in enumerate(values)) / denominator
    return abs(slope) * (len(values) - 1) / center


def _strictly_decreasing(values: list[float]) -> bool | None:
    if len(values) < 5:
        return None
    return all(next_value < value for value, next_value in zip(values, values[1:]))


def _actual_memory_trace_from_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    fw_us = float(metrics.get("fw_us") or metrics.get("fw_wall_us") or 0.0)
    bw_us = float(metrics.get("bw_us") or metrics.get("bw_wall_us") or 0.0)
    optimizer_us = float(metrics.get("optimizer_us") or metrics.get("optimizer_wall_us") or 0.0)
    fw_end = fw_us
    bw_end = fw_us + bw_us
    optimizer_end = fw_us + bw_us + optimizer_us
    return (
        {
            "phase": "start",
            "time_us": 0.0,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_bytes": 0,
        },
        {
            "phase": "fw_peak",
            "time_us": fw_end,
            "allocated_bytes": int(metrics.get("after_fw_allocated_bytes", 0)),
            "reserved_bytes": int(metrics.get("after_fw_reserved_bytes", 0)),
            "peak_bytes": int(metrics.get("fw_peak_bytes", 0)),
        },
        {
            "phase": "after_fw",
            "time_us": fw_end,
            "allocated_bytes": int(metrics.get("after_fw_allocated_bytes", 0)),
            "reserved_bytes": int(metrics.get("after_fw_reserved_bytes", 0)),
            "peak_bytes": int(metrics.get("after_fw_allocated_bytes", 0)),
        },
        {
            "phase": "bw_peak",
            "time_us": bw_end,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_bytes": int(metrics.get("bw_peak_bytes", 0)),
        },
        {
            "phase": "optimizer_peak",
            "time_us": optimizer_end,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_bytes": int(metrics.get("optimizer_peak_bytes", 0)),
        },
        {
            "phase": "overall_peak",
            "time_us": optimizer_end,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_bytes": int(metrics.get("overall_peak_bytes", 0)),
        },
    )


def _measure_window(
    fn: Callable[[], Any],
    cuda_device: torch.device | None,
) -> tuple[Any, float, float | None]:
    if cuda_device is None:
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1_000_000.0, None

    with torch.cuda.device(cuda_device):
        torch.cuda.synchronize(cuda_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize(cuda_device)
        wall_us = (time.perf_counter() - wall_start) * 1_000_000.0
        event_us = float(start_event.elapsed_time(end_event)) * 1_000.0
    return result, wall_us, event_us


def _run_training_step(
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
) -> Tensor:
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
    output = executable(*args, **kwargs)
    loss = loss_fn(output)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    loss.backward()
    optimizer.step()
    return loss


def _measure_warmup_sample(
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
    warmup_index: int,
    cuda_device: torch.device | None,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> dict[str, Any]:
    _, wall_us, event_us = _measure_window(
        lambda: _run_training_step(
            optimizer,
            executable,
            loss_fn,
            args,
            kwargs,
            zero_grad_set_to_none=zero_grad_set_to_none,
        ),
        cuda_device,
    )
    sample: dict[str, Any] = {
        "warmup_index": warmup_index,
        "trajectory": "warmup",
        "trajectory_order": 0,
    }
    sample.update(
        _timing_sample(
            "warmup",
            wall_us,
            event_us,
            max_event_wall_relative_gap,
            max_event_wall_absolute_gap_us,
        )
    )
    return sample


def _measure_phase_sample(
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
    repeat_index: int,
    cuda_device: torch.device | None,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "repeat_index": repeat_index,
        "trajectory": "phase",
        "trajectory_order": 2,
    }
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
    sampler = _CudaMemorySampler(cuda_device)
    sampler.start("fw")

    try:
        _reset_cuda_peak(cuda_device)

        def forward() -> Tensor:
            output = executable(*args, **kwargs)
            loss = loss_fn(output)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            return loss

        loss, wall_us, event_us = _measure_window(forward, cuda_device)
        sampler.set_phase("after_fw")
        sample.update(_timing_sample("fw", wall_us, event_us, max_event_wall_relative_gap, max_event_wall_absolute_gap_us))
        sample["fw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        sample["fw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
        sample["after_fw_allocated_bytes"] = _cuda_allocated_or_zero(cuda_device)
        sample["after_fw_reserved_bytes"] = _cuda_reserved_or_zero(cuda_device)

        sampler.set_phase("bw")
        _reset_cuda_peak(cuda_device)
        _, wall_us, event_us = _measure_window(loss.backward, cuda_device)
        sample.update(_timing_sample("bw", wall_us, event_us, max_event_wall_relative_gap, max_event_wall_absolute_gap_us))
        sample["bw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        sample["bw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)

        sampler.set_phase("optimizer")
        _reset_cuda_peak(cuda_device)
        _, wall_us, event_us = _measure_window(optimizer.step, cuda_device)
        sample.update(
            _timing_sample(
                "optimizer",
                wall_us,
                event_us,
                max_event_wall_relative_gap,
                max_event_wall_absolute_gap_us,
            )
        )
        sample["optimizer_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        sample["optimizer_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
    finally:
        sample["actual_sampled_memory_trace"] = sampler.stop()
    sample["phase_step_wall_us"] = sum(float(sample[f"{phase}_wall_us"]) for phase in ("fw", "bw", "optimizer"))
    phase_events = [sample[f"{phase}_event_us"] for phase in ("fw", "bw", "optimizer")]
    sample["phase_step_event_us"] = (
        sum(float(value) for value in phase_events) if all(value is not None for value in phase_events) else None
    )
    sample["actual_memory_trace"] = _actual_memory_trace_from_metrics(sample)
    return sample


def _measure_overall_sample(
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
    repeat_index: int,
    cuda_device: torch.device | None,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> dict[str, Any]:
    # This is deliberately a separate full-step allocation window. No phase
    # reset occurs between forward, backward, and optimizer execution.
    _reset_cuda_peak(cuda_device)
    sampler = _CudaMemorySampler(cuda_device)
    gpu_sampler = _GpuUtilizationSampler(cuda_device)
    sampler.start("overall")
    gpu_sampler.start()
    try:
        _, wall_us, event_us = _measure_window(
            lambda: _run_training_step(
                optimizer,
                executable,
                loss_fn,
                args,
                kwargs,
                zero_grad_set_to_none=zero_grad_set_to_none,
            ),
            cuda_device,
        )
    finally:
        sampled_trace = sampler.stop()
        gpu_util_trace, gpu_compute_summary = gpu_sampler.stop()
    sample: dict[str, Any] = {
        "repeat_index": repeat_index,
        "trajectory": "overall",
        "trajectory_order": 1,
    }
    sample.update(_timing_sample("overall", wall_us, event_us, max_event_wall_relative_gap, max_event_wall_absolute_gap_us))
    sample["overall_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
    sample["overall_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
    if sampled_trace:
        sample["actual_overall_sampled_memory_trace"] = sampled_trace
    if gpu_util_trace:
        sample["gpu_util_trace"] = gpu_util_trace
    sample["gpu_compute_summary"] = gpu_compute_summary
    return sample


def _aggregate_measurements(
    warmup_samples: list[dict[str, Any]],
    phase_samples: list[dict[str, Any]],
    overall_samples: list[dict[str, Any]],
    warmup_steps: int,
    warmup_trend_relative_tolerance: float,
    max_event_wall_relative_gap: float,
    max_event_wall_absolute_gap_us: float,
) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}

    phase_names = ("fw", "bw", "optimizer")
    for phase in phase_names:
        wall_value = _median_or_none([sample[f"{phase}_wall_us"] for sample in phase_samples])
        event_value = _median_or_none([sample[f"{phase}_event_us"] for sample in phase_samples])
        aggregated[f"{phase}_wall_us"] = wall_value
        aggregated[f"{phase}_event_us"] = event_value
        aggregated[f"{phase}_us"] = wall_value
        aggregated[f"{phase}_event_wall_abs_diff_us"] = _median_or_none(
            [sample[f"{phase}_event_wall_abs_diff_us"] for sample in phase_samples]
        )
        aggregated[f"{phase}_event_wall_relative_diff"] = _median_or_none(
            [sample[f"{phase}_event_wall_relative_diff"] for sample in phase_samples]
        )
        aggregated[f"{phase}_timing_qualified"] = _aggregate_timing_qualified(
            aggregated[f"{phase}_event_wall_relative_diff"],
            aggregated[f"{phase}_event_wall_abs_diff_us"],
            max_event_wall_relative_gap,
            max_event_wall_absolute_gap_us,
        )
        for memory_kind in ("peak_bytes", "reserved_peak_bytes"):
            key = f"{phase}_{memory_kind}"
            aggregated[key] = max(int(sample[key]) for sample in phase_samples)

    aggregated["after_fw_allocated_bytes"] = max(
        int(sample["after_fw_allocated_bytes"]) for sample in phase_samples
    )
    aggregated["after_fw_reserved_bytes"] = max(
        int(sample["after_fw_reserved_bytes"]) for sample in phase_samples
    )

    aggregated["phase_step_wall_us"] = _median_or_none(
        [sample["phase_step_wall_us"] for sample in phase_samples]
    )
    aggregated["phase_step_event_us"] = _median_or_none(
        [sample["phase_step_event_us"] for sample in phase_samples]
    )

    for key in (
        "overall_wall_us",
        "overall_event_us",
        "overall_event_wall_abs_diff_us",
        "overall_event_wall_relative_diff",
    ):
        aggregated[key] = _median_or_none([sample[key] for sample in overall_samples])
    aggregated["overall_timing_qualified"] = _aggregate_timing_qualified(
        aggregated["overall_event_wall_relative_diff"],
        aggregated["overall_event_wall_abs_diff_us"],
        max_event_wall_relative_gap,
        max_event_wall_absolute_gap_us,
    )
    aggregated["overall_peak_bytes"] = max(int(sample["overall_peak_bytes"]) for sample in overall_samples)
    aggregated["overall_reserved_peak_bytes"] = max(
        int(sample["overall_reserved_peak_bytes"]) for sample in overall_samples
    )

    # Preserve the legacy step_us contract as the sum of phase wall clocks.
    # Publication consumers use overall_wall_us for the independent full step.
    step_values = [float(sample["phase_step_wall_us"]) for sample in phase_samples]
    aggregated["step_us"] = float(aggregated["phase_step_wall_us"])
    aggregated["step_us_median"] = float(statistics.median(step_values))
    aggregated["step_us_p10"] = _percentile(step_values, 0.10)
    aggregated["step_us_p90"] = _percentile(step_values, 0.90)
    aggregated["measurement_repeats"] = len(overall_samples)
    aggregated["measurement_warmup_steps"] = warmup_steps
    last_warmups = warmup_samples[-5:]
    wall_slope = _relative_slope([float(sample["warmup_wall_us"]) for sample in last_warmups])
    event_values = [
        float(sample["warmup_event_us"]) for sample in last_warmups if sample["warmup_event_us"] is not None
    ]
    event_slope = _relative_slope(event_values)
    wall_strictly_decreasing = _strictly_decreasing(
        [float(sample["warmup_wall_us"]) for sample in last_warmups]
    )
    event_strictly_decreasing = _strictly_decreasing(event_values)
    aggregated["warmup_last5_wall_relative_slope"] = wall_slope
    aggregated["warmup_last5_event_relative_slope"] = event_slope
    aggregated["warmup_last5_wall_strictly_decreasing"] = wall_strictly_decreasing
    aggregated["warmup_last5_event_strictly_decreasing"] = event_strictly_decreasing
    aggregated["warmup_wall_trend_stable"] = (
        not wall_strictly_decreasing if wall_strictly_decreasing is not None else None
    )
    aggregated["warmup_event_trend_stable"] = (
        not event_strictly_decreasing if event_strictly_decreasing is not None else None
    )
    aggregated["max_event_wall_relative_gap"] = max_event_wall_relative_gap
    aggregated["max_event_wall_absolute_gap_us"] = max_event_wall_absolute_gap_us
    aggregated["warmup_trend_relative_tolerance"] = warmup_trend_relative_tolerance
    aggregated["warmup_samples"] = warmup_samples
    aggregated["phase_samples"] = phase_samples
    aggregated["overall_samples"] = overall_samples
    aggregated["actual_memory_trace"] = _actual_memory_trace_from_metrics(aggregated)
    overall_sampled_traces = [
        tuple(sample.get("actual_overall_sampled_memory_trace") or ())
        for sample in overall_samples
        if sample.get("actual_overall_sampled_memory_trace")
    ]
    if overall_sampled_traces:
        overall_sampled_trace = max(
            overall_sampled_traces,
            key=lambda trace: max((int(point.get("allocator_peak_bytes", point.get("peak_bytes", 0))) for point in trace), default=0),
        )
        aggregated["actual_overall_sampled_memory_trace"] = overall_sampled_trace
        aggregated["actual_overall_sampled_memory_trace_kind"] = "sampled_cuda_memory_overall"
        aggregated["actual_overall_sampled_memory_trace_interval_us"] = DEFAULT_MEMORY_SAMPLER_INTERVAL_US
    gpu_util_traces = [
        tuple(sample.get("gpu_util_trace") or ())
        for sample in overall_samples
        if sample.get("gpu_util_trace")
    ]
    if gpu_util_traces:
        gpu_util_trace = max(gpu_util_traces, key=len)
        aggregated["gpu_util_trace"] = gpu_util_trace
        aggregated["gpu_util_trace_kind"] = "nvidia_smi_overall_step"
    gpu_summaries = [dict(sample.get("gpu_compute_summary") or {}) for sample in overall_samples]
    ok_gpu_summaries = [summary for summary in gpu_summaries if summary.get("status") == "ok"]
    if ok_gpu_summaries:
        aggregated["gpu_compute_summary"] = max(
            ok_gpu_summaries,
            key=lambda summary: int(summary.get("sample_count") or 0),
        )
    elif gpu_summaries:
        aggregated["gpu_compute_summary"] = gpu_summaries[-1]
    sampled_traces = [
        tuple(sample.get("actual_sampled_memory_trace") or ())
        for sample in phase_samples
        if sample.get("actual_sampled_memory_trace")
    ]
    if sampled_traces:
        sampled_trace = max(
            sampled_traces,
            key=lambda trace: max((int(point.get("allocator_peak_bytes", point.get("peak_bytes", 0))) for point in trace), default=0),
        )
        aggregated["actual_sampled_memory_trace"] = sampled_trace
        aggregated["actual_sampled_memory_trace_kind"] = "sampled_cuda_memory_phase"
        aggregated["actual_sampled_memory_trace_interval_us"] = DEFAULT_MEMORY_SAMPLER_INTERVAL_US
    aggregated["actual_memory_trace_kind"] = "phase_boundary_anchor"
    aggregated["actual_memory_trace_sampled"] = int(aggregated["overall_peak_bytes"]) > 0
    aggregated["raw_samples"] = [
        {
            "repeat_index": index,
            "trajectory_order": ["overall", "phase"],
            "overall": overall_samples[index],
            "phase": phase_samples[index],
        }
        for index in range(len(overall_samples))
    ]
    aggregated["trajectory_order"] = ["warmup", "overall", "phase"]
    aggregated["callable_state_assumption"] = "state_dict_owned_or_registry_identity_bound"
    aggregated["python_module_state_policy"] = (
        "public_attributes_snapshotted_registered_callables_identity_checked_or_fail_closed"
    )
    aggregated["callable_device_contract"] = "single_device_model_inputs_optimizer_state"
    aggregated["cuda_event_scope"] = "current_stream_and_dependency_ordered_work_only"
    aggregated["process_isolation"] = "external_qualification_runner_required"
    return aggregated


def _aggregate_legacy_phase_metrics(
    samples: list[dict[str, Any]],
    warmup_steps: int,
) -> dict[str, Any]:
    keys = set().union(*(sample.keys() for sample in samples))
    aggregated: dict[str, Any] = {}
    for key in keys:
        if not all(isinstance(sample.get(key), (int, float)) for sample in samples):
            continue
        values = [float(sample[key]) for sample in samples]
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
    aggregated["actual_memory_trace"] = _actual_memory_trace_from_metrics(aggregated)
    sampled_traces = [
        tuple(sample.get("actual_sampled_memory_trace") or ())
        for sample in samples
        if sample.get("actual_sampled_memory_trace")
    ]
    if sampled_traces:
        sampled_trace = max(
            sampled_traces,
            key=lambda trace: max((int(point.get("allocated_bytes", 0)) for point in trace), default=0),
        )
        aggregated["actual_sampled_memory_trace"] = sampled_trace
        aggregated["actual_sampled_memory_trace_kind"] = "sampled_cuda_memory"
        aggregated["actual_sampled_memory_trace_interval_us"] = DEFAULT_MEMORY_SAMPLER_INTERVAL_US
    aggregated["actual_memory_trace_kind"] = "phase_boundary_anchor"
    aggregated["actual_memory_trace_sampled"] = bool(max(int(sample.get("overall_peak_bytes", 0)) for sample in samples))
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
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    warmup_steps = int(warmup_steps)
    repeat_count = int(repeat_count)
    model_state = _clone_model_state(model)
    optimizer_state = _clone_optimizer_state(optimizer)
    grad_state = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_device = _measurement_cuda_device(model, optimizer, args, kwargs)
    cuda_rng = torch.cuda.get_rng_state_all() if cuda_device is not None else None

    def restore_state() -> None:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        _restore_grads(model, grad_state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def measure_once() -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        restore_state()
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        sampler = _CudaMemorySampler(cuda_device)
        sampler.start("fw")

        try:
            _reset_cuda_peak(cuda_device)
            start = time.perf_counter()
            output = executable(*args, **kwargs)
            loss = loss_fn(output)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            sampler.set_phase("after_fw")
            metrics["fw_us"] = (time.perf_counter() - start) * 1_000_000.0
            metrics["fw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
            metrics["fw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
            metrics["after_fw_allocated_bytes"] = _cuda_allocated_or_zero(cuda_device)
            metrics["after_fw_reserved_bytes"] = _cuda_reserved_or_zero(cuda_device)

            sampler.set_phase("bw")
            _reset_cuda_peak(cuda_device)
            start = time.perf_counter()
            loss.backward()
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            metrics["bw_us"] = (time.perf_counter() - start) * 1_000_000.0
            metrics["bw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
            metrics["bw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)

            sampler.set_phase("optimizer")
            _reset_cuda_peak(cuda_device)
            start = time.perf_counter()
            optimizer.step()
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            metrics["optimizer_us"] = (time.perf_counter() - start) * 1_000_000.0
            metrics["optimizer_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
            metrics["optimizer_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
        finally:
            metrics["actual_sampled_memory_trace"] = sampler.stop()
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
        metrics["actual_memory_trace"] = _actual_memory_trace_from_metrics(metrics)
        return metrics

    try:
        for _ in range(warmup_steps):
            measure_once()
        samples = [measure_once() for _ in range(repeat_count)]
        return _aggregate_legacy_phase_metrics(samples, warmup_steps)
    finally:
        restore_state()


def _measure_publication_training_step_phases(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    zero_grad_set_to_none: bool,
    warmup_steps: int,
    repeat_count: int = 1,
    max_event_wall_relative_gap: float = 0.25,
    max_event_wall_absolute_gap_us: float = 2_000.0,
    warmup_trend_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    if max_event_wall_relative_gap < 0:
        raise ValueError("max_event_wall_relative_gap must be non-negative")
    if max_event_wall_absolute_gap_us < 0:
        raise ValueError("max_event_wall_absolute_gap_us must be non-negative")
    if warmup_trend_relative_tolerance < 0:
        raise ValueError("warmup_trend_relative_tolerance must be non-negative")
    warmup_steps = int(warmup_steps)
    repeat_count = int(repeat_count)
    cuda_device = _measurement_cuda_device(model, optimizer, args, kwargs)
    initial_state = _capture_training_state(model, optimizer, cuda_device, include_python_state=True)
    initial_args, initial_kwargs = _clone_pytree((args, kwargs))

    try:
        warmup_args, warmup_kwargs = _clone_pytree((initial_args, initial_kwargs))
        warmup_samples: list[dict[str, Any]] = []
        for index in range(30):
            warmup_samples.append(
                _measure_warmup_sample(
                    optimizer,
                    executable,
                    loss_fn,
                    warmup_args,
                    warmup_kwargs,
                    zero_grad_set_to_none=zero_grad_set_to_none,
                    warmup_index=index,
                    cuda_device=cuda_device,
                    max_event_wall_relative_gap=max_event_wall_relative_gap,
                    max_event_wall_absolute_gap_us=max_event_wall_absolute_gap_us,
                )
            )
            if len(warmup_samples) < warmup_steps:
                continue
            last_five = warmup_samples[-5:]
            wall_decreasing = _strictly_decreasing(
                [float(sample["warmup_wall_us"]) for sample in last_five]
            )
            event_decreasing = _strictly_decreasing(
                [
                    float(sample["warmup_event_us"])
                    for sample in last_five
                    if sample["warmup_event_us"] is not None
                ]
            )
            warmup_samples[-1]["last5_wall_strictly_decreasing"] = wall_decreasing
            warmup_samples[-1]["last5_event_strictly_decreasing"] = event_decreasing
            if wall_decreasing is not True and event_decreasing is not True:
                break
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        warm_state = _capture_training_state(model, optimizer, cuda_device, include_python_state=True)
        warm_args, warm_kwargs = _clone_pytree((warmup_args, warmup_kwargs))

        # The publication-primary overall trajectory runs before diagnostic
        # phase instrumentation can perturb allocator or compiler state.
        _restore_training_state(model, optimizer, warm_state, cuda_device)
        overall_args, overall_kwargs = _clone_pytree((warm_args, warm_kwargs))
        overall_samples = [
            _measure_overall_sample(
                optimizer,
                executable,
                loss_fn,
                overall_args,
                overall_kwargs,
                zero_grad_set_to_none=zero_grad_set_to_none,
                repeat_index=index,
                cuda_device=cuda_device,
                max_event_wall_relative_gap=max_event_wall_relative_gap,
                max_event_wall_absolute_gap_us=max_event_wall_absolute_gap_us,
            )
            for index in range(repeat_count)
        ]

        _restore_training_state(model, optimizer, warm_state, cuda_device)
        phase_args, phase_kwargs = _clone_pytree((warm_args, warm_kwargs))
        phase_samples = [
            _measure_phase_sample(
                optimizer,
                executable,
                loss_fn,
                phase_args,
                phase_kwargs,
                zero_grad_set_to_none=zero_grad_set_to_none,
                repeat_index=index,
                cuda_device=cuda_device,
                max_event_wall_relative_gap=max_event_wall_relative_gap,
                max_event_wall_absolute_gap_us=max_event_wall_absolute_gap_us,
            )
            for index in range(repeat_count)
        ]
        return _aggregate_measurements(
            warmup_samples,
            phase_samples,
            overall_samples,
            len(warmup_samples),
            warmup_trend_relative_tolerance,
            max_event_wall_relative_gap,
            max_event_wall_absolute_gap_us,
        )
    finally:
        try:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
        finally:
            _restore_training_state(model, optimizer, initial_state, cuda_device)


def measure_publication_training_step_phases(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    executable: Callable[..., Any],
    loss_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    backend: str,
    zero_grad_set_to_none: bool,
    warmup_steps: int,
    repeat_count: int,
    max_event_wall_relative_gap: float = 0.25,
    max_event_wall_absolute_gap_us: float = 2_000.0,
    warmup_trend_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Run the strict publication protocol for one single-device callable.

    Mutable state must live in ``state_dict``. Public Python attributes on named
    modules are snapshotted or rejected; compiler-owned callables additionally
    require an active identity-bound registry token. State hidden in unknown
    closures or external objects cannot be restored. CUDA Event timing covers the current
    stream and dependency-ordered work; unrelated side-stream work is not
    claimed and should fail the Event/wall-gap qualification.
    """
    if backend not in {"aot_eager", "inductor"}:
        raise ValueError("backend must be 'aot_eager' or 'inductor'")
    minimum_warmup = 5 if backend == "aot_eager" else 10
    if warmup_steps != minimum_warmup:
        raise ValueError(f"{backend} publication measurement starts with warmup_steps == {minimum_warmup}")
    if repeat_count < 20:
        raise ValueError("publication measurement requires repeat_count >= 20")

    metrics = _measure_publication_training_step_phases(
        model,
        optimizer,
        executable,
        loss_fn,
        args,
        kwargs,
        zero_grad_set_to_none=zero_grad_set_to_none,
        warmup_steps=warmup_steps,
        repeat_count=repeat_count,
        max_event_wall_relative_gap=max_event_wall_relative_gap,
        max_event_wall_absolute_gap_us=max_event_wall_absolute_gap_us,
        warmup_trend_relative_tolerance=warmup_trend_relative_tolerance,
    )
    phase_window_qualifications = [
        metrics[f"{window}_timing_qualified"] for window in ("fw", "bw", "optimizer")
    ]
    warmup_timing = [sample["warmup_timing_qualified"] for sample in metrics["warmup_samples"]]
    phase_timing_qualified = all(value is True for value in phase_window_qualifications)
    warmup_timing_qualified = all(value is True for value in warmup_timing)
    timing_qualified = metrics["overall_timing_qualified"] is True
    event_trend = metrics["warmup_event_trend_stable"]
    trend_qualified = metrics["warmup_wall_trend_stable"] is True and event_trend in {True, None}

    reasons: list[str] = []
    if not timing_qualified:
        reasons.append("event_wall_gap_or_event_unavailable")
    if not trend_qualified:
        reasons.append("warmup_last5_trend_unstable")
    metrics["publication_backend"] = backend
    metrics["warmup_initial_steps"] = minimum_warmup
    metrics["warmup_auto_extended_steps"] = len(metrics["warmup_samples"]) - minimum_warmup
    metrics["warmup_reached_max_steps"] = len(metrics["warmup_samples"]) == 30
    metrics["timing_qualified"] = timing_qualified
    metrics["phase_timing_qualified"] = phase_timing_qualified
    metrics["warmup_timing_qualified"] = warmup_timing_qualified
    metrics["publication_timing_scope"] = "overall_step"
    metrics["phase_timing_scope"] = "diagnostic_only"
    metrics["warmup_timing_scope"] = "diagnostic_only"
    metrics["warmup_trend_qualified"] = trend_qualified
    metrics["publication_qualified"] = timing_qualified and trend_qualified
    if metrics["publication_qualified"]:
        metrics["publication_status"] = "qualified"
    elif not timing_qualified:
        metrics["publication_status"] = "timing_unqualified"
    else:
        metrics["publication_status"] = "warmup_unqualified"
    metrics["publication_unqualified_reasons"] = reasons
    return metrics
