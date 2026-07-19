from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils import _pytree


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
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _clone_grads(model: torch.nn.Module) -> tuple[Tensor | None, ...]:
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in model.parameters())


def _restore_grads(model: torch.nn.Module, grads: tuple[Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


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


_IMMUTABLE_PYTHON_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _snapshot_python_value(value: Any, memo: dict[int, Any] | None = None) -> Any:
    if type(value) in _IMMUTABLE_PYTHON_TYPES:
        return value
    if memo is None:
        memo = {}
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if isinstance(value, Tensor):
        snapshot = value.detach().clone(memory_format=torch.preserve_format)
        snapshot.requires_grad_(value.requires_grad)
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
    if isinstance(snapshot, Tensor):
        if not isinstance(original, Tensor):
            return _snapshot_python_value(snapshot)
        with torch.no_grad():
            original.copy_(snapshot)
        return original
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
        optimizer=copy.deepcopy(optimizer.state_dict()),
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
) -> dict[str, float | bool | None]:
    if event_us is None:
        absolute_difference = None
        relative_difference = None
    else:
        absolute_difference = abs(wall_us - event_us)
        relative_difference = absolute_difference / event_us if event_us > 0 else None
    timing_qualified = (
        relative_difference <= max_event_wall_relative_gap if relative_difference is not None else None
    )
    return {
        f"{prefix}_wall_us": wall_us,
        f"{prefix}_event_us": event_us,
        f"{prefix}_event_wall_abs_diff_us": absolute_difference,
        f"{prefix}_event_wall_relative_diff": relative_difference,
        f"{prefix}_timing_qualified": timing_qualified,
    }


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
    sample.update(_timing_sample("warmup", wall_us, event_us, max_event_wall_relative_gap))
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
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "repeat_index": repeat_index,
        "trajectory": "phase",
        "trajectory_order": 2,
    }
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

    _reset_cuda_peak(cuda_device)

    def forward() -> Tensor:
        output = executable(*args, **kwargs)
        loss = loss_fn(output)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        return loss

    loss, wall_us, event_us = _measure_window(forward, cuda_device)
    sample.update(_timing_sample("fw", wall_us, event_us, max_event_wall_relative_gap))
    sample["fw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
    sample["fw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
    sample["after_fw_allocated_bytes"] = _cuda_allocated_or_zero(cuda_device)
    sample["after_fw_reserved_bytes"] = _cuda_reserved_or_zero(cuda_device)

    _reset_cuda_peak(cuda_device)
    _, wall_us, event_us = _measure_window(loss.backward, cuda_device)
    sample.update(_timing_sample("bw", wall_us, event_us, max_event_wall_relative_gap))
    sample["bw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
    sample["bw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)

    _reset_cuda_peak(cuda_device)
    _, wall_us, event_us = _measure_window(optimizer.step, cuda_device)
    sample.update(_timing_sample("optimizer", wall_us, event_us, max_event_wall_relative_gap))
    sample["optimizer_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
    sample["optimizer_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
    sample["phase_step_wall_us"] = sum(float(sample[f"{phase}_wall_us"]) for phase in ("fw", "bw", "optimizer"))
    phase_events = [sample[f"{phase}_event_us"] for phase in ("fw", "bw", "optimizer")]
    sample["phase_step_event_us"] = (
        sum(float(value) for value in phase_events) if all(value is not None for value in phase_events) else None
    )
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
) -> dict[str, Any]:
    # This is deliberately a separate full-step allocation window. No phase
    # reset occurs between forward, backward, and optimizer execution.
    _reset_cuda_peak(cuda_device)
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
        "repeat_index": repeat_index,
        "trajectory": "overall",
        "trajectory_order": 1,
    }
    sample.update(_timing_sample("overall", wall_us, event_us, max_event_wall_relative_gap))
    sample["overall_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
    sample["overall_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
    return sample


def _aggregate_measurements(
    warmup_samples: list[dict[str, Any]],
    phase_samples: list[dict[str, Any]],
    overall_samples: list[dict[str, Any]],
    warmup_steps: int,
    warmup_trend_relative_tolerance: float,
    max_event_wall_relative_gap: float,
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
        qualifications = [sample[f"{phase}_timing_qualified"] for sample in phase_samples]
        aggregated[f"{phase}_timing_qualified"] = (
            all(bool(value) for value in qualifications) if all(value is not None for value in qualifications) else None
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
    overall_qualifications = [sample["overall_timing_qualified"] for sample in overall_samples]
    aggregated["overall_timing_qualified"] = (
        all(bool(value) for value in overall_qualifications)
        if all(value is not None for value in overall_qualifications)
        else None
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
    aggregated["warmup_trend_relative_tolerance"] = warmup_trend_relative_tolerance
    aggregated["warmup_samples"] = warmup_samples
    aggregated["phase_samples"] = phase_samples
    aggregated["overall_samples"] = overall_samples
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
    aggregated["callable_state_assumption"] = "pure_or_state_dict_owned"
    aggregated["python_module_state_policy"] = "public_attributes_snapshotted_or_fail_closed"
    aggregated["callable_device_contract"] = "single_device_model_inputs_optimizer_state"
    aggregated["cuda_event_scope"] = "current_stream_and_dependency_ordered_work_only"
    aggregated["process_isolation"] = "external_qualification_runner_required"
    return aggregated


def _aggregate_legacy_phase_metrics(
    samples: list[dict[str, int | float]],
    warmup_steps: int,
) -> dict[str, int | float]:
    keys = set().union(*(sample.keys() for sample in samples))
    aggregated: dict[str, int | float] = {}
    for key in keys:
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
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    warmup_steps = int(warmup_steps)
    repeat_count = int(repeat_count)
    model_state = _clone_model_state(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    grad_state = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cuda_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None

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

        _reset_cuda_peak(cuda_device)
        start = time.perf_counter()
        output = executable(*args, **kwargs)
        loss = loss_fn(output)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        metrics["fw_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["fw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        metrics["fw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)

        _reset_cuda_peak(cuda_device)
        start = time.perf_counter()
        loss.backward()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        metrics["bw_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["bw_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        metrics["bw_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)

        _reset_cuda_peak(cuda_device)
        start = time.perf_counter()
        optimizer.step()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        metrics["optimizer_us"] = (time.perf_counter() - start) * 1_000_000.0
        metrics["optimizer_peak_bytes"] = _cuda_allocated_peak_or_zero(cuda_device)
        metrics["optimizer_reserved_peak_bytes"] = _cuda_reserved_peak_or_zero(cuda_device)
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
    warmup_trend_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    if max_event_wall_relative_gap < 0:
        raise ValueError("max_event_wall_relative_gap must be non-negative")
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
    warmup_trend_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Run the strict publication protocol for one single-device callable.

    The callable must be pure or keep mutable state in ``state_dict``. Public
    Python attributes on named modules are snapshotted or rejected, but state
    hidden in closures or external objects cannot be restored. CUDA Event timing covers the current
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
        warmup_trend_relative_tolerance=warmup_trend_relative_tolerance,
    )
    window_qualifications = [
        metrics[f"{window}_timing_qualified"] for window in ("fw", "bw", "optimizer", "overall")
    ]
    warmup_timing = [sample["warmup_timing_qualified"] for sample in metrics["warmup_samples"]]
    timing_qualified = all(value is True for value in (*window_qualifications, *warmup_timing))
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
