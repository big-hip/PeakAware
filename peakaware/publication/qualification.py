from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import random
import signal
import tempfile
import time
import traceback
import types
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils import _pytree

from peakaware.api import optimize_training
from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import (
    ExecutionSpec,
    FrozenConfig,
    HardwareSpec,
    TrainingRequest,
    TrainingTaskSpec,
    WorkloadSpec,
)
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.models.registry import TrainingTaskRegistry
from peakaware.publication.baselines import (
    PreparedMethod,
    prepare_all_save,
    prepare_aot_min_cut,
    prepare_block_activation_checkpoint,
    prepare_selective_activation_checkpoint,
)
from peakaware.runtime import measure_publication_training_step_phases
from peakaware.workload_manifest import (
    attempt_fingerprint,
    build_workload_manifest,
    canonical_json,
    case_id,
    execution_config_fingerprint,
    execution_spec_to_dict,
    render_t1_markdown,
    replicate_fingerprint,
    validate_record_manifest_reference,
    workload_fingerprint,
    workload_spec_to_dict,
)


QUALIFICATION_SCHEMA_VERSION = "1.0"
SUPPORTED_STATUSES = frozenset(
    {
        "ok",
        "budget_violation",
        "oom",
        "timeout",
        "correctness_failure",
        "compile_failure",
        "runtime_failure",
        "unsupported",
        "infra_failure",
    }
)
PUBLICATION_METHODS = ("all_save", "pytorch_min_cut", "block_ac", "sac", "peakaware")
PUBLICATION_BACKENDS = ("aot_eager", "inductor")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _thaw(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _json_or_none(value: Any | None) -> str | None:
    return None if value is None else canonical_json(value)


def _decode_or_none(value: str | None) -> Any | None:
    return None if value is None else json.loads(value)


def _all_cuda_device_indexes() -> tuple[int, ...]:
    if not torch.cuda.is_available():
        return ()
    return tuple(range(torch.cuda.device_count()))


def _preserve_all_rng(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        state = _capture_rng_state(_all_cuda_device_indexes())
        try:
            return fn(*args, **kwargs)
        finally:
            _restore_rng_state(state)

    return wrapped


def _tensor_binding(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "requires_grad": value.requires_grad,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _binding_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_binding(value)
    if isinstance(value, Mapping):
        return {
            str(key): _binding_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_binding_value(item) for item in value],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, types.CodeType):
        return {
            "kind": "code",
            "name": value.co_name,
            "bytecode_sha256": hashlib.sha256(value.co_code).hexdigest(),
            "constants": _binding_value(value.co_consts),
            "names": list(value.co_names),
        }
    if callable(value):
        return {
            "kind": "callable_reference",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        }
    return {
        "kind": "python_object",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "state": _binding_value(vars(value)) if hasattr(value, "__dict__") else repr(value),
    }


def _callable_binding(value: Callable[..., Any]) -> dict[str, Any]:
    target = value
    partial_payload = None
    if isinstance(value, functools.partial):
        target = value.func
        partial_payload = {
            "args": _binding_value(value.args),
            "keywords": _binding_value(value.keywords or {}),
        }
    code = getattr(target, "__code__", None)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = None
    closure = getattr(target, "__closure__", None)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "module": getattr(target, "__module__", type(target).__module__),
        "qualname": getattr(target, "__qualname__", type(target).__qualname__),
        "code_sha256": None if code is None else hashlib.sha256(code.co_code).hexdigest(),
        "constants": None if code is None else _binding_value(code.co_consts),
        "source_sha256": None if source is None else hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "defaults": _binding_value(getattr(target, "__defaults__", None)),
        "kwdefaults": _binding_value(getattr(target, "__kwdefaults__", None)),
        "closure": None if closure is None else _binding_value(tuple(cell.cell_contents for cell in closure)),
        "partial": partial_payload,
        "object_state": _binding_value(vars(value)) if hasattr(value, "__dict__") else None,
    }


@_preserve_all_rng
def _task_binding(task: TrainingTaskSpec, *, microbatch_size: int, seed: int) -> dict[str, Any]:
    _seed_all(seed)
    model = task.build_model().cpu().train()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(microbatch_size)
    initial_model_state = _binding_value(model.state_dict())
    initial_batch = _binding_value((args, kwargs))
    initial_optimizer_state = _binding_value(optimizer.state_dict())
    with torch.no_grad():
        output = model(*args, **kwargs)
        loss = task.loss_fn(output)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError(f"task {task.name!r} loss probe did not return a scalar tensor")
    model_structure = [
        {
            "name": name,
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
            "extra_repr": module.extra_repr(),
        }
        for name, module in model.named_modules()
    ]
    optimizer_groups = []
    for group in optimizer.param_groups:
        optimizer_groups.append(
            _binding_value({key: value for key, value in group.items() if key != "params"})
        )
    payload = {
        "protocol": "qualification_task_binding_v1",
        "task_name": task.name,
        "seed": seed,
        "microbatch_size": microbatch_size,
        "factories": {
            "model": _callable_binding(task.build_model),
            "batch": _callable_binding(task.build_batch),
            "optimizer": _callable_binding(task.build_optimizer),
            "loss": _callable_binding(task.loss_fn),
        },
        "probe": {
            "model_structure": model_structure,
            "model_initial_state": initial_model_state,
            "batch_initial_content": initial_batch,
            "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "optimizer_defaults": _binding_value(optimizer.defaults),
            "optimizer_param_groups": optimizer_groups,
            "optimizer_initial_state": initial_optimizer_state,
            "model_output": _binding_value(output),
            "loss": _tensor_binding(loss),
            "rng_after_probe": _rng_binding(_capture_rng_state(_all_cuda_device_indexes())),
        },
    }
    return {**payload, "fingerprint": _sha256(payload)}


@dataclass(frozen=True)
class QualificationSlot:
    schema_version: str
    run_id: str
    slot_id: str
    case_id: str
    replicate_id: str
    attempt_id: str
    manifest_entry_id: str
    workload_fingerprint: str
    execution_fingerprint: str
    task_name: str
    display_name: str
    workload_config: FrozenConfig
    execution_config: FrozenConfig
    backend: str
    method: str
    memory_budget_bytes: int
    microbatch_size: int
    seed: int
    replicate_index: int
    attempt_index: int
    pairing_block_id: str
    execution_order_index: int

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported qualification schema version")
        if self.backend not in PUBLICATION_BACKENDS:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.method not in PUBLICATION_METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        if self.memory_budget_bytes <= 0 or self.microbatch_size <= 0:
            raise ValueError("memory budget and microbatch size must be positive")
        if self.replicate_index < 0 or self.attempt_index < 0 or self.execution_order_index < 0:
            raise ValueError("replicate and attempt indexes must be non-negative")
        if not isinstance(self.workload_config, FrozenConfig):
            object.__setattr__(self, "workload_config", FrozenConfig(self.workload_config))
        if not isinstance(self.execution_config, FrozenConfig):
            object.__setattr__(self, "execution_config", FrozenConfig(self.execution_config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "slot_id": self.slot_id,
            "case_id": self.case_id,
            "replicate_id": self.replicate_id,
            "attempt_id": self.attempt_id,
            "manifest_entry_id": self.manifest_entry_id,
            "workload_fingerprint": self.workload_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "task_name": self.task_name,
            "display_name": self.display_name,
            "workload_config": _thaw(self.workload_config),
            "execution_config": _thaw(self.execution_config),
            "backend": self.backend,
            "method": self.method,
            "memory_budget_bytes": self.memory_budget_bytes,
            "microbatch_size": self.microbatch_size,
            "seed": self.seed,
            "replicate_index": self.replicate_index,
            "attempt_index": self.attempt_index,
            "pairing_block_id": self.pairing_block_id,
            "execution_order_index": self.execution_order_index,
        }


@dataclass(frozen=True)
class QualificationRecord:
    slot: QualificationSlot
    process_id: int | None
    status: str
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    runtime_identity: str | None = None
    correctness_report: str | None = None
    measurement_raw: str | None = None
    measurement_aggregate: str | None = None
    environment: str | None = None
    elapsed_seconds: float | None = None
    last_progress_stage: str | None = None

    def __post_init__(self) -> None:
        if self.status not in SUPPORTED_STATUSES:
            raise ValueError(f"invalid qualification status: {self.status}")
        failed = self.status != "ok" and self.status != "budget_violation"
        if failed and not all((self.error_stage, self.error_type, self.error_message)):
            raise ValueError("failed records require error stage, type, and message")
        if not failed and any((self.error_stage, self.error_type, self.error_message)):
            raise ValueError("successful and budget-violation records cannot contain error fields")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.status in {"ok", "budget_violation"}:
            identity = _decode_or_none(self.runtime_identity)
            correctness = _decode_or_none(self.correctness_report)
            raw = _decode_or_none(self.measurement_raw)
            aggregate = _decode_or_none(self.measurement_aggregate)
            environment = _decode_or_none(self.environment)
            if not isinstance(identity, Mapping) or identity.get("status") != "ready":
                raise ValueError("qualified records require a ready runtime identity")
            if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
                raise ValueError("qualified records require passed correctness")
            if not isinstance(raw, Mapping) or not raw.get("overall_samples") or not raw.get("phase_samples"):
                raise ValueError("qualified records require non-empty raw measurement samples")
            if not isinstance(aggregate, Mapping) or aggregate.get("publication_qualified") is not True:
                raise ValueError("qualified records require a qualified measurement aggregate")
            if not isinstance(environment, Mapping) or not environment:
                raise ValueError("qualified records require an environment snapshot")
        if self.status == "unsupported":
            identity = _decode_or_none(self.runtime_identity)
            if (
                not isinstance(identity, Mapping)
                or identity.get("status") != "unsupported"
                or not identity.get("fallback_reason")
            ):
                raise ValueError("unsupported records require an explicit fallback runtime identity")

    def to_dict(self) -> dict[str, Any]:
        payload = self.slot.to_dict()
        payload.update(
            {
                "process_id": self.process_id,
                "status": self.status,
                "error_stage": self.error_stage,
                "error_type": self.error_type,
                "error_message": self.error_message,
                "runtime_identity": _decode_or_none(self.runtime_identity),
                "correctness_report": _decode_or_none(self.correctness_report),
                "measurement_raw": _decode_or_none(self.measurement_raw),
                "measurement_aggregate": _decode_or_none(self.measurement_aggregate),
                "environment": _decode_or_none(self.environment),
                "elapsed_seconds": self.elapsed_seconds,
                "last_progress_stage": self.last_progress_stage,
            }
        )
        return payload


def _execution_spec(backend: str, device: str) -> ExecutionSpec:
    compiler_protocol = "aot_eager_publication_v1" if backend == "aot_eager" else "inductor_publication_v1"
    return ExecutionSpec(
        schema_version="1.0",
        backend=backend,
        device=device,
        compiler_protocol=compiler_protocol,
        precision_protocol="fp32_no_autocast",
        measurement_protocol="publication_phase_overall_v1",
    )


def _pairing_block_id(
    workload_fingerprint_value: str,
    execution_fingerprint_value: str,
    *,
    memory_budget_bytes: int,
    replicate_id: str,
) -> str:
    return _sha256(
        {
            "identity_kind": "qualification_pairing_block",
            "workload_fingerprint": workload_fingerprint_value,
            "execution_fingerprint": execution_fingerprint_value,
            "memory_budget_bytes": memory_budget_bytes,
            "replicate_id": replicate_id,
        }
    )


def _slot_id(run_id: str, case_id_value: str, replicate_id: str, attempt_id: str) -> str:
    return _sha256(
        {
            "identity_kind": "qualification_slot",
            "run_id": run_id,
            "case_id": case_id_value,
            "replicate_id": replicate_id,
            "attempt_id": attempt_id,
        }
    )


@_preserve_all_rng
def build_qualification_slots(
    tasks: Sequence[TrainingTaskSpec],
    *,
    run_id: str,
    backends: Sequence[str],
    methods: Sequence[str],
    memory_budgets_bytes: Sequence[int],
    replicates: int,
    microbatch_size: int,
    device: str,
    base_seed: int = 1337,
    attempt_index: int = 0,
    repeat_count: int = 20,
    timeout_s: float = 900.0,
    warmup_steps_by_backend: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], tuple[QualificationSlot, ...]]:
    if not run_id:
        raise ValueError("run_id must not be empty")
    if not tasks or not backends or not methods or not memory_budgets_bytes:
        raise ValueError("qualification matrix dimensions must not be empty")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if repeat_count < 20 or timeout_s <= 0:
        raise ValueError("qualification requires repeat_count >= 20 and timeout_s > 0")
    if len(set(backends)) != len(backends) or len(set(methods)) != len(methods):
        raise ValueError("backends and methods must not contain duplicates")
    for backend in backends:
        if backend not in PUBLICATION_BACKENDS:
            raise ValueError(f"unsupported backend: {backend}")
    for method in methods:
        if method not in PUBLICATION_METHODS:
            raise ValueError(f"unsupported method: {method}")
    missing_workloads = [task.name for task in tasks if task.workload is None]
    if missing_workloads:
        raise ValueError(f"tasks have no workload identity: {missing_workloads}")
    registry_keys = [task.workload.registry_key for task in tasks if task.workload is not None]
    workload_fingerprints = [
        workload_fingerprint(task.workload, microbatch_size=microbatch_size)
        for task in tasks
        if task.workload is not None
    ]
    if len(registry_keys) != len(set(registry_keys)):
        raise ValueError("qualification tasks contain duplicate workload registry keys")
    if len(workload_fingerprints) != len(set(workload_fingerprints)):
        raise ValueError("qualification tasks contain duplicate workload fingerprints")
    warmups = dict(warmup_steps_by_backend or {backend: 5 if backend == "aot_eager" else 10 for backend in backends})
    if set(warmups) != set(backends):
        raise ValueError("warmup_steps_by_backend must cover exactly the registered backends")
    for backend, steps in warmups.items():
        expected = 5 if backend == "aot_eager" else 10
        if steps != expected:
            raise ValueError(f"{backend} publication warmup must start at {expected}")
    manifest = build_workload_manifest(
        tasks,
        microbatch_size=microbatch_size,
        seed=base_seed,
        compiler_mode=",".join(backends),
    )
    manifest["task_bindings"] = {
        task.name: _task_binding(task, microbatch_size=microbatch_size, seed=base_seed)
        for task in sorted(tasks, key=lambda item: item.name)
    }
    entries = {entry["workload_fingerprint"]: entry for entry in manifest["workloads"]}
    slots: list[QualificationSlot] = []
    for task in sorted(tasks, key=lambda item: item.name):
        if task.workload is None:
            raise ValueError(f"task {task.name!r} has no workload identity")
        task_fingerprint = workload_fingerprint(task.workload, microbatch_size=microbatch_size)
        entry = entries[task_fingerprint]
        for backend in sorted(backends):
            execution = _execution_spec(backend, device)
            execution_fingerprint = execution_config_fingerprint(execution)
            for budget in sorted(memory_budgets_bytes):
                for replicate_index in range(replicates):
                    seed = base_seed + replicate_index
                    replicate_id = replicate_fingerprint(
                        entry["workload_fingerprint"],
                        execution_fingerprint,
                        memory_budget_bytes=budget,
                        seed=seed,
                        replicate_index=replicate_index,
                    )
                    attempt_id = attempt_fingerprint(replicate_id, attempt_id=attempt_index)
                    for method in sorted(methods):
                        method_case_id = case_id(
                            entry["workload_fingerprint"],
                            execution_fingerprint,
                            memory_budget_bytes=budget,
                            strategy=method,
                        )
                        block_id = _pairing_block_id(
                            entry["workload_fingerprint"],
                            execution_fingerprint,
                            memory_budget_bytes=budget,
                            replicate_id=replicate_id,
                        )
                        slots.append(
                            QualificationSlot(
                                schema_version=QUALIFICATION_SCHEMA_VERSION,
                                run_id=run_id,
                                slot_id=_slot_id(run_id, method_case_id, replicate_id, attempt_id),
                                case_id=method_case_id,
                                replicate_id=replicate_id,
                                attempt_id=attempt_id,
                                manifest_entry_id=entry["manifest_entry_id"],
                                workload_fingerprint=entry["workload_fingerprint"],
                                execution_fingerprint=execution_fingerprint,
                                task_name=task.name,
                                display_name=task.workload.display_name,
                                workload_config=FrozenConfig(workload_spec_to_dict(task.workload)),
                                execution_config=FrozenConfig(execution_spec_to_dict(execution)),
                                backend=backend,
                                method=method,
                                memory_budget_bytes=budget,
                                microbatch_size=microbatch_size,
                                seed=seed,
                                replicate_index=replicate_index,
                                attempt_index=attempt_index,
                                pairing_block_id=block_id,
                                execution_order_index=0,
                            )
                        )
    blocks: dict[str, list[QualificationSlot]] = {}
    for slot in slots:
        blocks.setdefault(slot.pairing_block_id, []).append(slot)
    ordered_slots: list[QualificationSlot] = []
    execution_blocks: list[dict[str, Any]] = []
    for block_id in sorted(blocks):
        block_slots = sorted(blocks[block_id], key=lambda item: item.method)
        shuffle_seed = int(_sha256({"base_seed": base_seed, "pairing_block_id": block_id})[:16], 16)
        random.Random(shuffle_seed).shuffle(block_slots)
        execution_blocks.append(
            {
                "pairing_block_id": block_id,
                "shuffle_seed": shuffle_seed,
                "slot_ids": [slot.slot_id for slot in block_slots],
                "method_order": [slot.method for slot in block_slots],
            }
        )
        ordered_slots.extend(block_slots)
    slots = [replace(slot, execution_order_index=index) for index, slot in enumerate(ordered_slots)]
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("qualification slot identity collision")
    manifest["qualification_run"] = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "run_id": run_id,
        "matrix": {
            "tasks": sorted(task.name for task in tasks),
            "backends": sorted(backends),
            "methods": sorted(methods),
            "memory_budgets_bytes": sorted(memory_budgets_bytes),
            "replicates": replicates,
            "microbatch_size": microbatch_size,
            "device": device,
            "base_seed": base_seed,
            "attempt_index": attempt_index,
            "repeat_count": repeat_count,
            "timeout_s": float(timeout_s),
            "warmup_steps_by_backend": warmups,
            "spawn_start_interruptible": False,
            "timeout_deadline_origin": "immediately_before_multiprocessing.Process.start",
            "spawn_start_boundary": (
                "Python multiprocessing.Process.start cannot be safely interrupted; elapsed start time is "
                "charged to the deadline and checked immediately after start returns"
            ),
            "worker_process_group_protocol": "POSIX setsid + process-group TERM/KILL; process fallback otherwise",
        },
        "expected_slot_count": len(slots),
        "expected_slot_ids": sorted(slot.slot_id for slot in slots),
        "execution_order": [slot.slot_id for slot in slots],
        "pairing_blocks": execution_blocks,
    }
    for slot in slots:
        validate_qualification_slot(slot, manifest)
    return manifest, tuple(slots)


def validate_qualification_slot(slot: QualificationSlot, manifest: Mapping[str, Any]) -> None:
    _validate_qualification_preregistration(manifest)
    workload = _thaw(slot.workload_config)
    execution = _thaw(slot.execution_config)
    workload_spec = WorkloadSpec(
        schema_version=workload["schema_version"],
        registry_key=workload["registry_key"],
        display_name=workload["display_name"],
        model_family=workload["model_family"],
        implementation=workload["implementation"],
        model_config=workload["model"],
        input_config=workload["input"],
        optimizer_config=workload["optimizer"],
        loss_config=workload["loss"],
        compute_dtype=workload["compute_dtype"],
        parameter_dtype=workload["parameter_dtype"],
    )
    execution_spec = ExecutionSpec(
        schema_version=execution["schema_version"],
        backend=execution["backend"],
        device=execution["device"],
        compiler_protocol=execution["compiler_protocol"],
        precision_protocol=execution["precision_protocol"],
        measurement_protocol=execution["measurement_protocol"],
    )
    expected_workload = workload_fingerprint(workload_spec, microbatch_size=slot.microbatch_size)
    expected_execution = execution_config_fingerprint(execution_spec)
    expected_case = case_id(
        expected_workload,
        expected_execution,
        memory_budget_bytes=slot.memory_budget_bytes,
        strategy=slot.method,
    )
    expected_replicate = replicate_fingerprint(
        expected_workload,
        expected_execution,
        memory_budget_bytes=slot.memory_budget_bytes,
        seed=slot.seed,
        replicate_index=slot.replicate_index,
    )
    expected_attempt = attempt_fingerprint(expected_replicate, attempt_id=slot.attempt_index)
    expected_block = _pairing_block_id(
        expected_workload,
        expected_execution,
        memory_budget_bytes=slot.memory_budget_bytes,
        replicate_id=expected_replicate,
    )
    expected = {
        "workload_fingerprint": expected_workload,
        "execution_fingerprint": expected_execution,
        "case_id": expected_case,
        "replicate_id": expected_replicate,
        "attempt_id": expected_attempt,
        "pairing_block_id": expected_block,
        "slot_id": _slot_id(slot.run_id, expected_case, expected_replicate, expected_attempt),
        "manifest_entry_id": expected_workload,
        "display_name": workload_spec.display_name,
        "task_name": workload_spec.registry_key,
        "backend": execution_spec.backend,
    }
    for field_name, expected_value in expected.items():
        if getattr(slot, field_name) != expected_value:
            raise ValueError(
                f"qualification slot {field_name} mismatch: "
                f"recorded={getattr(slot, field_name)!r}, expected={expected_value!r}"
            )
    run = manifest.get("qualification_run")
    if not isinstance(run, Mapping) or run.get("run_id") != slot.run_id:
        raise ValueError("qualification slot run_id is not registered in the manifest")
    matrix = run.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("qualification manifest matrix is missing")
    matrix_checks = {
        "task": slot.task_name in matrix.get("tasks", ()),
        "backend": slot.backend in matrix.get("backends", ()),
        "method": slot.method in matrix.get("methods", ()),
        "memory budget": slot.memory_budget_bytes in matrix.get("memory_budgets_bytes", ()),
        "microbatch": slot.microbatch_size == matrix.get("microbatch_size"),
        "device": execution_spec.device == matrix.get("device"),
        "attempt": slot.attempt_index == matrix.get("attempt_index"),
        "replicate": 0 <= slot.replicate_index < matrix.get("replicates", 0),
        "seed": slot.seed == matrix.get("base_seed") + slot.replicate_index,
    }
    failed_checks = [name for name, passed in matrix_checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"qualification slot disagrees with manifest matrix: {failed_checks}")
    expected_ids = run.get("expected_slot_ids")
    execution_order = run.get("execution_order")
    if not isinstance(expected_ids, list) or slot.slot_id not in expected_ids:
        raise ValueError("qualification slot is not pre-registered in expected_slot_ids")
    if not isinstance(execution_order, list) or slot.execution_order_index >= len(execution_order):
        raise ValueError("qualification slot execution order index is invalid")
    if execution_order[slot.execution_order_index] != slot.slot_id:
        raise ValueError("qualification slot execution order does not match the manifest")
    blocks = run.get("pairing_blocks")
    matches = [block for block in blocks or () if block.get("pairing_block_id") == slot.pairing_block_id]
    if len(matches) != 1 or slot.slot_id not in matches[0].get("slot_ids", ()):
        raise ValueError("qualification slot pairing block is not pre-registered")
    validate_record_manifest_reference(manifest, slot.to_dict())


def validate_worker_task_binding(
    task: TrainingTaskSpec,
    slot: QualificationSlot,
    manifest: Mapping[str, Any],
) -> None:
    validate_qualification_slot(slot, manifest)
    if task.workload is None:
        raise ValueError("worker registry task has no WorkloadSpec")
    actual_config = workload_spec_to_dict(task.workload)
    if canonical_json(actual_config) != canonical_json(_thaw(slot.workload_config)):
        raise ValueError("worker registry WorkloadSpec does not match the preregistered slot")
    actual_fingerprint = workload_fingerprint(task.workload, microbatch_size=slot.microbatch_size)
    if actual_fingerprint != slot.workload_fingerprint:
        raise ValueError("worker registry workload fingerprint does not match the slot")
    bindings = manifest.get("task_bindings")
    expected_binding = bindings.get(slot.task_name) if isinstance(bindings, Mapping) else None
    if not isinstance(expected_binding, Mapping):
        raise ValueError("manifest has no preregistered task binding for the worker task")
    matrix = manifest["qualification_run"]["matrix"]
    actual_binding = _task_binding(
        task,
        microbatch_size=slot.microbatch_size,
        seed=matrix["base_seed"],
    )
    if canonical_json(actual_binding) != canonical_json(expected_binding):
        raise ValueError("worker registry task binding does not match the preregistered factories and probe")
    validate_record_manifest_reference(
        manifest,
        {
            "manifest_entry_id": slot.manifest_entry_id,
            "workload_fingerprint": actual_fingerprint,
            "display_name": task.workload.display_name,
            "workload_config": actual_config,
        },
    )


def _validate_qualification_preregistration(manifest: Mapping[str, Any]) -> None:
    run = manifest.get("qualification_run")
    if not isinstance(run, Mapping) or not run.get("run_id"):
        raise ValueError("manifest has no valid qualification_run preregistration")
    matrix = run.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("qualification manifest matrix is missing")
    dimensions = (
        matrix.get("tasks"),
        matrix.get("backends"),
        matrix.get("methods"),
        matrix.get("memory_budgets_bytes"),
    )
    if any(not isinstance(values, list) or not values for values in dimensions):
        raise ValueError("qualification matrix dimensions must be non-empty lists")
    if any(len(values) != len({canonical_json(value) for value in values}) for values in dimensions):
        raise ValueError("qualification matrix dimensions must not contain duplicates")
    if any(
        not isinstance(value, str) or not value
        for field_name in ("tasks", "backends", "methods")
        for value in matrix[field_name]
    ):
        raise ValueError("qualification task/backend/method dimensions must contain non-empty strings")
    integer_fields = {
        "microbatch_size": 1,
        "base_seed": None,
        "attempt_index": 0,
        "repeat_count": 20,
    }
    for field_name, minimum in integer_fields.items():
        value = matrix.get(field_name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or (minimum is not None and value < minimum)
        ):
            raise ValueError(f"qualification matrix {field_name} is invalid")
    timeout_s = matrix.get("timeout_s")
    if not _finite_non_negative_number(timeout_s) or timeout_s == 0:
        raise ValueError("qualification matrix timeout_s is invalid")
    if not isinstance(matrix.get("device"), str) or not matrix["device"]:
        raise ValueError("qualification matrix device is invalid")
    if any(not _non_negative_int(budget) or budget == 0 for budget in matrix["memory_budgets_bytes"]):
        raise ValueError("qualification matrix memory budgets are invalid")
    if any(backend not in PUBLICATION_BACKENDS for backend in matrix["backends"]):
        raise ValueError("qualification matrix contains an unsupported backend")
    if any(method not in PUBLICATION_METHODS for method in matrix["methods"]):
        raise ValueError("qualification matrix contains an unsupported method")
    warmups = matrix.get("warmup_steps_by_backend")
    if not isinstance(warmups, Mapping) or set(warmups) != set(matrix["backends"]):
        raise ValueError("qualification warmups must cover exactly the matrix backends")
    for backend, warmup in warmups.items():
        if warmup != (5 if backend == "aot_eager" else 10):
            raise ValueError("qualification backend warmup protocol is invalid")
    if matrix.get("spawn_start_interruptible") is not False:
        raise ValueError("qualification spawn start boundary must remain explicitly non-interruptible")
    boundary = matrix.get("spawn_start_boundary")
    if not isinstance(boundary, str) or "cannot be safely interrupted" not in boundary:
        raise ValueError("qualification spawn start boundary disclosure is invalid")
    bindings = manifest.get("task_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(matrix["tasks"]):
        raise ValueError("qualification task bindings must cover exactly the preregistered tasks")
    for task_name, binding in bindings.items():
        if (
            not isinstance(binding, Mapping)
            or binding.get("protocol") != "qualification_task_binding_v1"
            or binding.get("task_name") != task_name
            or binding.get("seed") != matrix.get("base_seed")
            or binding.get("microbatch_size") != matrix.get("microbatch_size")
            or binding.get("fingerprint")
            != _sha256({key: value for key, value in binding.items() if key != "fingerprint"})
        ):
            raise ValueError(f"qualification task binding is invalid: {task_name}")
    expected_product = 1
    for values in dimensions:
        expected_product *= len(values)
    replicates = matrix.get("replicates")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("qualification matrix replicates must be positive")
    expected_product *= replicates
    expected_ids = run.get("expected_slot_ids")
    execution_order = run.get("execution_order")
    if (
        run.get("expected_slot_count") != expected_product
        or not isinstance(expected_ids, list)
        or len(expected_ids) != expected_product
        or len(set(expected_ids)) != expected_product
        or not isinstance(execution_order, list)
        or len(execution_order) != expected_product
        or set(execution_order) != set(expected_ids)
    ):
        raise ValueError("qualification expected slots do not equal the matrix Cartesian product")
    blocks = run.get("pairing_blocks")
    expected_block_count = (
        len(matrix["tasks"])
        * len(matrix["backends"])
        * len(matrix["memory_budgets_bytes"])
        * replicates
    )
    if not isinstance(blocks, list) or len(blocks) != expected_block_count:
        raise ValueError("qualification pairing block count does not match the matrix")
    block_slot_ids: list[str] = []
    for block in blocks:
        if (
            not isinstance(block, Mapping)
            or set(block.get("method_order", ())) != set(matrix["methods"])
            or len(block.get("slot_ids", ())) != len(matrix["methods"])
        ):
            raise ValueError("qualification pairing block does not contain every method exactly once")
        block_slot_ids.extend(block["slot_ids"])
    if len(block_slot_ids) != len(set(block_slot_ids)) or set(block_slot_ids) != set(expected_ids):
        raise ValueError("qualification pairing blocks do not partition expected slots")


def _move_pytree(value: Any, device: torch.device) -> Any:
    return _pytree.tree_map(lambda item: item.to(device) if isinstance(item, torch.Tensor) else item, value)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_seeded_model_optimizer(
    task: TrainingTaskSpec,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    _seed_all(seed)
    model = task.build_model().to(device).train()
    optimizer = task.build_optimizer(model)
    return model, optimizer


def _build_seeded_batch(
    task: TrainingTaskSpec,
    microbatch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _seed_all(seed)
    args, kwargs = task.build_batch(microbatch_size)
    return _move_pytree((args, kwargs), device)


def _clone_pytree(value: Any) -> Any:
    return _pytree.tree_map(
        lambda item: item.detach().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item),
        value,
    )


def _value_is_finite(value: Any) -> bool:
    leaves, _ = _pytree.tree_flatten(value)
    return all(
        not isinstance(item, torch.Tensor) or bool(torch.isfinite(item).all())
        for item in leaves
    )


def _related_cuda_devices(*values: Any) -> tuple[int, ...]:
    indexes: set[int] = set()
    for value in values:
        leaves, _ = _pytree.tree_flatten(value)
        for item in leaves:
            if isinstance(item, nn.Module):
                tensors = tuple(item.parameters()) + tuple(item.buffers())
            elif isinstance(item, torch.optim.Optimizer):
                tensors = tuple(
                    state_value
                    for state in item.state.values()
                    for state_value in state.values()
                    if isinstance(state_value, torch.Tensor)
                )
            elif isinstance(item, torch.Tensor):
                tensors = (item,)
            else:
                tensors = ()
            for tensor in tensors:
                if tensor.device.type == "cuda":
                    indexes.add(tensor.device.index if tensor.device.index is not None else torch.cuda.current_device())
    return tuple(sorted(indexes))


def _capture_rng_state(cuda_devices: Sequence[int]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        numpy_state = None
    else:
        raw = np.random.get_state()
        numpy_state = (raw[0], raw[1].copy(), raw[2], raw[3], raw[4])
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": {index: torch.cuda.get_rng_state(index).clone() for index in cuda_devices},
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    if state["numpy"] is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    for index, cuda_state in state["torch_cuda"].items():
        torch.cuda.set_rng_state(cuda_state, index)


def _rng_binding(state: Mapping[str, Any]) -> dict[str, Any]:
    numpy_state = state["numpy"]
    if numpy_state is None:
        numpy_binding = None
    else:
        numpy_binding = {
            "algorithm": numpy_state[0],
            "keys_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        }
    return {
        "python_sha256": hashlib.sha256(repr(state["python"]).encode("utf-8")).hexdigest(),
        "numpy": numpy_binding,
        "torch_cpu": _tensor_binding(state["torch_cpu"]),
        "torch_cuda": {
            str(index): _tensor_binding(cuda_state)
            for index, cuda_state in sorted(state["torch_cuda"].items())
        },
    }


def _compare_rng_states(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, bool | None]:
    python_match = reference["python"] == candidate["python"]
    if reference["numpy"] is None or candidate["numpy"] is None:
        numpy_match = reference["numpy"] is candidate["numpy"]
    else:
        import numpy as np

        left = reference["numpy"]
        right = candidate["numpy"]
        numpy_match = (
            left[0] == right[0]
            and np.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )
    torch_cpu_match = torch.equal(reference["torch_cpu"], candidate["torch_cpu"])
    cuda_match = set(reference["torch_cuda"]) == set(candidate["torch_cuda"]) and all(
        torch.equal(reference["torch_cuda"][index], candidate["torch_cuda"][index])
        for index in reference["torch_cuda"]
    )
    return {
        "python_random_match": python_match,
        "numpy_random_match": numpy_match,
        "torch_cpu_rng_match": torch_cpu_match,
        "torch_cuda_rng_match": cuda_match,
    }


@dataclass
class _Comparison:
    passed: bool = True
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatch: str | None = None

    def fail(self, path: str) -> None:
        if self.passed:
            self.passed = False
            self.mismatch = path


def _compare_values(reference: Any, candidate: Any, result: _Comparison, path: str, atol: float, rtol: float) -> None:
    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        left = reference.detach().cpu()
        right = candidate.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            result.fail(f"{path}: tensor metadata")
            return
        if left.is_floating_point() or left.is_complex():
            if left.numel():
                absolute = (left - right).abs()
                result.max_abs_error = max(result.max_abs_error, float(absolute.max().item()))
                denominator = left.abs().clamp_min(torch.finfo(left.real.dtype).eps)
                result.max_rel_error = max(result.max_rel_error, float((absolute / denominator).max().item()))
            if not torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False):
                result.fail(path)
        elif not torch.equal(left, right):
            result.fail(path)
        return
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        if set(reference) != set(candidate):
            result.fail(f"{path}: mapping keys")
            return
        for key in sorted(reference, key=str):
            _compare_values(reference[key], candidate[key], result, f"{path}.{key}", atol, rtol)
        return
    if isinstance(reference, (tuple, list)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            result.fail(f"{path}: sequence length")
            return
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _compare_values(left, right, result, f"{path}[{index}]", atol, rtol)
        return
    if reference != candidate:
        result.fail(path)


def _run_three_steps(
    model: nn.Module,
    executable: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = executable(*args, **kwargs)
        loss = loss_fn(output)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        loss.backward()
        gradients: dict[str, Any] = {}
        gradients_finite = True
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            gradients[name] = None if gradient is None else gradient.detach().cpu().clone()
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                gradients_finite = False
        steps.append(
            {
                "output": _clone_pytree(output),
                "loss": loss.detach().cpu().clone(),
                "gradients": gradients,
                "gradients_finite": gradients_finite,
            }
        )
        optimizer.step()
    return {
        "steps": steps,
        "parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()},
        "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
        "optimizer": _clone_pytree(optimizer.state_dict()),
    }


def qualify_three_step_correctness(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    candidate_executable: Callable[..., Any],
    reference_optimizer: torch.optim.Optimizer,
    candidate_optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    device: torch.device,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    cuda_devices = set(
        _related_cuda_devices(
            reference_model,
            candidate_model,
            reference_optimizer,
            candidate_optimizer,
            args,
            kwargs,
        )
    )
    if device.type == "cuda":
        cuda_devices.add(device.index if device.index is not None else torch.cuda.current_device())
    caller_rng = _capture_rng_state(sorted(cuda_devices))
    try:
        _restore_rng_state(caller_rng)
        reference = _run_three_steps(
            reference_model,
            reference_model,
            reference_optimizer,
            loss_fn,
            *_clone_pytree((args, kwargs)),
        )
        reference_rng = _capture_rng_state(sorted(cuda_devices))
        _restore_rng_state(caller_rng)
        candidate = _run_three_steps(
            candidate_model,
            candidate_executable,
            candidate_optimizer,
            loss_fn,
            *_clone_pytree((args, kwargs)),
        )
        candidate_rng = _capture_rng_state(sorted(cuda_devices))
    finally:
        _restore_rng_state(caller_rng)

    comparison = _Comparison()
    _compare_values(reference, candidate, comparison, "training", atol, rtol)
    rng_components = _compare_rng_states(reference_rng, candidate_rng)
    rng_match = all(value is True for value in rng_components.values())
    gradients_finite = all(step["gradients_finite"] for run in (reference, candidate) for step in run["steps"])
    parameters_finite = _value_is_finite(candidate["parameters"])
    buffers_finite = _value_is_finite(candidate["buffers"])
    optimizer_state_finite = _value_is_finite(candidate["optimizer"])
    passed = (
        comparison.passed
        and rng_match
        and gradients_finite
        and parameters_finite
        and buffers_finite
        and optimizer_state_finite
    )
    return {
        "protocol": "three_optimizer_steps_v1",
        "steps": 3,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
        "values_match": comparison.passed,
        "rng_match": rng_match,
        **rng_components,
        "gradients_finite": gradients_finite,
        "parameters_finite": parameters_finite,
        "buffers_finite": buffers_finite,
        "optimizer_state_finite": optimizer_state_finite,
        "max_abs_error": comparison.max_abs_error,
        "max_rel_error": comparison.max_rel_error,
        "first_mismatch": comparison.mismatch or (None if rng_match else "rng"),
    }


def _hardware(device: torch.device) -> HardwareSpec:
    if device.type != "cuda":
        return HardwareSpec(str(device), False, None)
    return HardwareSpec(str(device), True, int(torch.cuda.get_device_properties(device).total_memory))


def _capture(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task: TrainingTaskSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: int,
    device: torch.device,
) -> Any:
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=budget,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=_hardware(device),
        request_key=f"publication:{task.name}:{budget}",
    )
    return capture_joint_graph(request)


class _Unsupported(RuntimeError):
    def __init__(self, message: str, identity: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.identity = None if identity is None else dict(identity)


def _unsupported_runtime_identity(
    slot: QualificationSlot,
    reason: str,
    *,
    api: str | None = None,
    policy: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity_provenance = {
        "requested_method": slot.method,
        "requested_backend": slot.backend,
        **dict(provenance or {}),
    }
    return {
        "method_id": slot.method,
        "status": "unsupported",
        "is_real": False,
        "api": api,
        "policy": policy,
        "region_paths": [],
        "compiler_protocol": f"{slot.backend}:unavailable",
        "fallback_reason": reason,
        "provenance": identity_provenance,
    }


def convert_physical_budget_to_activation_memory_budget(
    physical_budget_bytes: int,
    *,
    fixed_physical_bytes: int | None,
    maximum_saved_activation_bytes: int | None,
) -> dict[str, Any]:
    """Convert only when both physical frontier and activation range are proven."""
    evidence = {
        "physical_budget_bytes": physical_budget_bytes,
        "fixed_physical_bytes": fixed_physical_bytes,
        "maximum_saved_activation_bytes": maximum_saved_activation_bytes,
        "formula": "(physical_budget_bytes-fixed_physical_bytes)/maximum_saved_activation_bytes",
    }
    if physical_budget_bytes <= 0:
        raise ValueError("physical_budget_bytes must be positive")
    if fixed_physical_bytes is None or maximum_saved_activation_bytes is None:
        return {
            **evidence,
            "status": "unavailable",
            "ratio": None,
            "reason": (
                "physical peak includes allocator, workspace, gradient, optimizer, and compiler effects; "
                "the slot has no frozen fixed-frontier and maximum-saved-activation calibration"
            ),
        }
    if fixed_physical_bytes < 0 or maximum_saved_activation_bytes <= 0:
        raise ValueError("conversion evidence must contain non-negative fixed bytes and positive activation bytes")
    ratio = (physical_budget_bytes - fixed_physical_bytes) / maximum_saved_activation_bytes
    if not 0.0 <= ratio <= 1.0:
        return {
            **evidence,
            "status": "out_of_range",
            "ratio": None,
            "reason": "the calibrated absolute budget maps outside PyTorch's [0, 1] activation ratio",
        }
    return {**evidence, "status": "converted", "ratio": ratio, "reason": None}


def _compile_all_save(model: nn.Module, backend: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    if backend != "aot_eager":
        raise _Unsupported(f"all_save has no qualified {backend} adapter")
    try:
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch.partitioners import default_partition
    except (ImportError, AttributeError) as exc:
        raise _Unsupported(f"all-save AOT APIs are unavailable: {exc}") from exc

    observations: dict[str, Any] = {
        "fw_compile_count": 0,
        "bw_compile_count": 0,
        "fw_graph_sha256": [],
        "bw_graph_sha256": [],
    }

    def compile_graph(graph_module: torch.fx.GraphModule, _inputs: list[Any], phase: str):
        observations[f"{phase}_compile_count"] += 1
        observations[f"{phase}_graph_sha256"].append(
            hashlib.sha256(str(graph_module.graph).encode("utf-8")).hexdigest()
        )
        return make_boxed_func(graph_module.forward)

    compiler = aot_autograd(
        fw_compiler=lambda graph, inputs: compile_graph(graph, inputs, "fw"),
        bw_compiler=lambda graph, inputs: compile_graph(graph, inputs, "bw"),
        partition_fn=default_partition,
        keep_inference_input_mutations=True,
    )
    executable = torch.compile(model, backend=compiler)
    prepared = prepare_all_save(model).require_supported()
    identity = prepared.identity.to_dict()
    identity["api"] = "torch.compile+aot_autograd(default_partition)"
    identity["compiler_protocol"] = "aot_eager:custom_aot_autograd_default_partition"
    identity["policy"] = "default_partition_without_rematerialization"
    identity["provenance"] = {
        **dict(identity.get("provenance", {})),
        "partition_fn": "torch._functorch.partitioners.default_partition",
        "partition_fn_source_sha256": hashlib.sha256(
            inspect.getsource(default_partition).encode("utf-8")
        ).hexdigest(),
        "all_save_naming_scope": (
            "no-rematerialization AOTAutograd partition; all values required by backward are returned "
            "by the forward according to default_partition, excluding dead/non-required values"
        ),
    }
    setattr(executable, "_peakaware_runtime_observations", observations)
    return executable, identity


def _identity_with_runtime_observations(
    identity: Mapping[str, Any],
    executable: Callable[..., Any],
) -> dict[str, Any]:
    result = dict(identity)
    observations = getattr(executable, "_peakaware_runtime_observations", None)
    if observations is not None:
        result["runtime_observations"] = _thaw(observations)
    return result


def _prepare_method(
    slot: QualificationSlot,
    task: TrainingTaskSpec,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    if slot.backend == "inductor":
        reason = f"{slot.method} has no qualified Inductor adapter"
        raise _Unsupported(reason, _unsupported_runtime_identity(slot, reason))
    if slot.method == "all_save":
        return _compile_all_save(model, slot.backend)
    elif slot.method == "pytorch_min_cut":
        conversion = convert_physical_budget_to_activation_memory_budget(
            slot.memory_budget_bytes,
            fixed_physical_bytes=None,
            maximum_saved_activation_bytes=None,
        )
        if conversion["status"] != "converted":
            reason = f"absolute-to-activation budget conversion unavailable: {conversion['reason']}"
            raise _Unsupported(
                reason,
                _unsupported_runtime_identity(
                    slot,
                    reason,
                    api="torch._functorch.partitioners.min_cut_rematerialization_partition",
                    policy=None,
                    provenance={"budget_conversion": conversion},
                ),
            )
        capture = _capture(model, optimizer, task, args, kwargs, slot.memory_budget_bytes, device)
        prepared = prepare_aot_min_cut(
            model,
            capture,
            example_args=args,
            example_kwargs=kwargs,
            loss_fn=task.loss_fn,
            activation_memory_budget=float(conversion["ratio"]),
            execution_backend="aot_lowered_graphmodule_eager",
        )
    elif slot.method == "block_ac":
        reason = "block_ac adapter is eager and is not integrated with the requested aot_eager compiler protocol"
        raise _Unsupported(reason, _unsupported_runtime_identity(slot, reason))
    elif slot.method == "sac":
        reason = "sac adapter is eager and is not integrated with the requested aot_eager compiler protocol"
        raise _Unsupported(reason, _unsupported_runtime_identity(slot, reason))
    elif slot.method == "peakaware":
        config = PeakAwareConfig(
            capture_backend="aot",
            enable_compile=True,
            enable_inductor=slot.backend == "inductor",
            measurement_warmup_steps=0,
            measurement_repeats=1,
            rng_seed=slot.seed,
        )
        result = optimize_training(
            model,
            args,
            example_kwargs=kwargs,
            loss_fn=task.loss_fn,
            optimizer=optimizer,
            memory_budget_bytes=slot.memory_budget_bytes,
            config=config,
        )
        metrics = result.executable.phase_metrics
        activation_checkpoint = bool(metrics.get("activation_checkpoint", 0))
        aot_runtime = bool(metrics.get("aot_partition_runtime", 0))
        identity = {
            "method_id": "peakaware",
            "status": "ready",
            "is_real": True,
            "api": "peakaware.optimize_training",
            "policy": result.selected_plan.plan_id,
            "compiler_protocol": f"{slot.backend}:peakaware_aot_candidate_runtime",
            "fallback_reason": None,
            "plan_id": result.selected_plan.plan_id,
            "plan_provenance": result.selected_plan.strategy_expectation_provenance,
            "activation_checkpoint": activation_checkpoint,
            "aot_partition_runtime": aot_runtime,
            "dry_run_replay_mode": None if result.dry_run is None else result.dry_run.replay_mode,
            "provenance": {
                "requested_method": slot.method,
                "requested_backend": slot.backend,
                "selected_plan_strategy": result.selected_plan.strategy,
            },
        }
        dry_run = result.dry_run
        identity["dry_run"] = None if dry_run is None else {
            "abi_valid": dry_run.abi_valid,
            "outputs_match": dry_run.outputs_match,
            "gradients_match": dry_run.gradients_match,
            "rng_match": dry_run.rng_match,
            "replay_mode": dry_run.replay_mode,
        }
        identity_valid = (
            result.selected_plan.plan_id != "all_save"
            and not activation_checkpoint
            and aot_runtime
            and result.executable.correctness_passed
            and dry_run is not None
            and dry_run.abi_valid
            and dry_run.outputs_match
            and dry_run.gradients_match
            and dry_run.rng_match
        )
        if not identity_valid:
            identity["status"] = "unsupported"
            identity["fallback_reason"] = "selected plan did not execute through its lowered AOT partition"
            raise _Unsupported(identity["fallback_reason"], identity)
        return result.executable.forward_backward, identity
    else:  # pragma: no cover - guarded by QualificationSlot
        reason = f"unknown method {slot.method}"
        raise _Unsupported(reason, _unsupported_runtime_identity(slot, reason))

    if not isinstance(prepared, PreparedMethod) or not prepared.supported or prepared.executable is None:
        identity = prepared.identity.to_dict()
        raise _Unsupported(prepared.identity.fallback_reason or "adapter preparation failed", identity)
    identity = prepared.identity.to_dict()
    if slot.method == "pytorch_min_cut":
        identity["provenance"] = {
            **dict(identity.get("provenance", {})),
            "budget_conversion": conversion,
        }
    return prepared.executable, identity


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "session_id": os.getsid(0) if hasattr(os, "getsid") else None,
        "process_group_id": os.getpgrp() if hasattr(os, "getpgrp") else None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu_name"] = properties.name
        result["gpu_uuid"] = str(getattr(properties, "uuid", None))
    else:
        result["gpu_name"] = None
        result["gpu_uuid"] = None
    return result


def _record_failure(
    slot: QualificationSlot,
    *,
    process_id: int | None,
    status: str,
    stage: str,
    error: BaseException,
    runtime_identity: Mapping[str, Any] | None = None,
    correctness_report: Mapping[str, Any] | None = None,
    measurement: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    last_progress_stage: str | None = None,
) -> QualificationRecord:
    raw = None
    aggregate = None
    if measurement is not None:
        raw_keys = ("warmup_samples", "phase_samples", "overall_samples", "raw_samples")
        raw = {key: measurement.get(key) for key in raw_keys if key in measurement}
        aggregate = {key: value for key, value in measurement.items() if key not in raw_keys}
        overall_samples = raw.get("overall_samples")
        if overall_samples:
            aggregate["per_run_max_overall_allocated_bytes"] = max(
                int(sample["overall_peak_bytes"]) for sample in overall_samples
            )
    return QualificationRecord(
        slot=slot,
        process_id=process_id,
        status=status,
        error_stage=stage,
        error_type=type(error).__name__,
        error_message=str(error),
        runtime_identity=_json_or_none(runtime_identity),
        correctness_report=_json_or_none(correctness_report),
        measurement_raw=_json_or_none(raw),
        measurement_aggregate=_json_or_none(aggregate),
        environment=_json_or_none(environment),
        elapsed_seconds=elapsed_seconds,
        last_progress_stage=last_progress_stage or stage,
    )


def _worker_record(
    slot: QualificationSlot,
    manifest: Mapping[str, Any],
    warmup_steps: int,
    repeat_count: int,
    progress: Callable[[str, float], None] | None = None,
) -> QualificationRecord:
    started = time.monotonic()
    process_id = os.getpid()
    stage = "worker_setup"
    runtime_identity: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    measurement: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None

    def mark(next_stage: str) -> None:
        nonlocal stage
        stage = next_stage
        if progress is not None:
            progress(stage, time.monotonic() - started)

    try:
        mark("worker_setup")
        _seed_all(slot.seed)
        device = torch.device(_thaw(slot.execution_config)["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(slot.seed)
        environment = _environment(device)
        registry = TrainingTaskRegistry.with_defaults()
        task = registry.get(slot.task_name)
        validate_worker_task_binding(task, slot, manifest)

        mark("model_build")
        reference_model, reference_optimizer = _build_seeded_model_optimizer(task, device, slot.seed)
        candidate_model, candidate_optimizer = _build_seeded_model_optimizer(task, device, slot.seed)
        args, kwargs = _build_seeded_batch(task, slot.microbatch_size, device, slot.seed)

        mark("method_prepare")
        executable, runtime_identity = _prepare_method(
            slot,
            task,
            candidate_model,
            candidate_optimizer,
            args,
            kwargs,
            device,
        )

        mark("correctness")
        correctness = qualify_three_step_correctness(
            reference_model,
            candidate_model,
            executable,
            reference_optimizer,
            candidate_optimizer,
            task.loss_fn,
            args,
            kwargs,
            device=device,
        )
        if not correctness["passed"]:
            raise _CorrectnessFailure(correctness["first_mismatch"] or "three-step correctness mismatch")
        runtime_identity = _identity_with_runtime_observations(runtime_identity, executable)
        if slot.method == "all_save":
            observed = runtime_identity.get("runtime_observations", {})
            if observed.get("fw_compile_count", 0) < 1 or observed.get("bw_compile_count", 0) < 1:
                raise _CorrectnessFailure("all-save compiler callbacks were not observed for forward and backward")

        # Correctness qualification advances state. Rebuild a clean, independent
        # measurement instance so the measured trajectory starts from its own seed.
        mark("measurement_setup")
        measurement_model, measurement_optimizer = _build_seeded_model_optimizer(task, device, slot.seed)
        measurement_args, measurement_kwargs = _build_seeded_batch(
            task,
            slot.microbatch_size,
            device,
            slot.seed,
        )
        measurement_executable, measurement_identity = _prepare_method(
            slot,
            task,
            measurement_model,
            measurement_optimizer,
            measurement_args,
            measurement_kwargs,
            device,
        )
        identity_without_observations = {
            key: value for key, value in runtime_identity.items() if key != "runtime_observations"
        }
        if canonical_json(identity_without_observations) != canonical_json(measurement_identity):
            raise _CorrectnessFailure("runtime identity changed between qualification and measurement preparation")

        mark("measurement")
        measurement = measure_publication_training_step_phases(
            measurement_model,
            measurement_optimizer,
            measurement_executable,
            task.loss_fn,
            measurement_args,
            measurement_kwargs,
            backend=slot.backend,
            zero_grad_set_to_none=True,
            warmup_steps=warmup_steps,
            repeat_count=repeat_count,
        )
        runtime_identity = _identity_with_runtime_observations(measurement_identity, measurement_executable)
        if not measurement.get("publication_qualified", False):
            raise _RuntimeQualificationFailure(str(measurement.get("publication_status", "unqualified")))
        overall_samples = measurement.get("overall_samples", ())
        if len(overall_samples) != repeat_count:
            raise _RuntimeQualificationFailure("measurement did not preserve every overall repeat")
        per_run_peak = max(int(sample["overall_peak_bytes"]) for sample in overall_samples)
        status = "ok" if per_run_peak <= slot.memory_budget_bytes else "budget_violation"
        raw_keys = ("warmup_samples", "phase_samples", "overall_samples", "raw_samples")
        raw = {key: measurement.get(key) for key in raw_keys if key in measurement}
        aggregate = {key: value for key, value in measurement.items() if key not in raw_keys}
        aggregate["per_run_max_overall_allocated_bytes"] = per_run_peak
        return QualificationRecord(
            slot=slot,
            process_id=process_id,
            status=status,
            runtime_identity=_json_or_none(runtime_identity),
            correctness_report=_json_or_none(correctness),
            measurement_raw=_json_or_none(raw),
            measurement_aggregate=_json_or_none(aggregate),
            environment=_json_or_none(environment),
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )
    except _Unsupported as exc:
        return _record_failure(
            slot,
            process_id=process_id,
            status="unsupported",
            stage=stage,
            error=exc,
            runtime_identity=exc.identity or runtime_identity,
            correctness_report=correctness,
            measurement=measurement,
            environment=environment,
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )
    except _CorrectnessFailure as exc:
        return _record_failure(
            slot,
            process_id=process_id,
            status="correctness_failure",
            stage=stage,
            error=exc,
            runtime_identity=runtime_identity,
            correctness_report=correctness,
            measurement=measurement,
            environment=environment,
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )
    except _RuntimeQualificationFailure as exc:
        return _record_failure(
            slot,
            process_id=process_id,
            status="runtime_failure",
            stage=stage,
            error=exc,
            runtime_identity=runtime_identity,
            correctness_report=correctness,
            measurement=measurement,
            environment=environment,
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _record_failure(
            slot,
            process_id=process_id,
            status="oom",
            stage=stage,
            error=exc,
            runtime_identity=runtime_identity,
            correctness_report=correctness,
            measurement=measurement,
            environment=environment,
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )
    except Exception as exc:
        compile_stages = {"method_prepare", "measurement_setup"}
        status = "compile_failure" if stage in compile_stages else "runtime_failure"
        if stage in {"worker_setup", "model_build"}:
            status = "infra_failure"
        return _record_failure(
            slot,
            process_id=process_id,
            status=status,
            stage=stage,
            error=exc,
            runtime_identity=runtime_identity,
            correctness_report=correctness,
            measurement=measurement,
            environment=environment,
            elapsed_seconds=time.monotonic() - started,
            last_progress_stage=stage,
        )


class _CorrectnessFailure(RuntimeError):
    pass


class _RuntimeQualificationFailure(RuntimeError):
    pass


def _worker_entry(
    result_channel: Any,
    slot: QualificationSlot,
    manifest: Mapping[str, Any],
    process_marker: str,
    warmup_steps: int,
    repeat_count: int,
) -> None:
    try:
        os.environ["PEAKAWARE_QUALIFICATION_MARKER"] = process_marker
        if os.name == "posix":
            os.setsid()

        def progress(stage: str, elapsed_seconds: float) -> None:
            result_channel.send(
                {
                    "kind": "progress",
                    "slot_id": slot.slot_id,
                    "process_id": os.getpid(),
                    "stage": stage,
                    "elapsed_seconds": elapsed_seconds,
                }
            )

        record = _worker_record(slot, manifest, warmup_steps, repeat_count, progress)
        result_channel.send({"kind": "record", "payload": record.to_dict()})
    except BaseException as exc:  # ensure the parent receives a diagnosable record
        try:
            result_channel.send(
                {
                    "worker_transport_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        result_channel.close()


def _record_from_dict(slot: QualificationSlot, payload: Mapping[str, Any]) -> QualificationRecord:
    return QualificationRecord(
        slot=slot,
        process_id=payload.get("process_id"),
        status=str(payload["status"]),
        error_stage=payload.get("error_stage"),
        error_type=payload.get("error_type"),
        error_message=payload.get("error_message"),
        runtime_identity=_json_or_none(payload.get("runtime_identity")),
        correctness_report=_json_or_none(payload.get("correctness_report")),
        measurement_raw=_json_or_none(payload.get("measurement_raw")),
        measurement_aggregate=_json_or_none(payload.get("measurement_aggregate")),
        environment=_json_or_none(payload.get("environment")),
        elapsed_seconds=payload.get("elapsed_seconds"),
        last_progress_stage=payload.get("last_progress_stage"),
    )


def _slot_from_dict(payload: Mapping[str, Any]) -> QualificationSlot:
    field_names = set(QualificationSlot.__dataclass_fields__)
    missing = field_names - set(payload)
    if missing:
        raise ValueError(f"qualification record is missing slot fields: {sorted(missing)}")
    values = {name: payload[name] for name in field_names}
    values["workload_config"] = FrozenConfig(values["workload_config"])
    values["execution_config"] = FrozenConfig(values["execution_config"])
    return QualificationSlot(**values)


def _artifact_record_from_dict(payload: Mapping[str, Any]) -> QualificationRecord:
    expected_fields = set(QualificationSlot.__dataclass_fields__) | {
        name for name in QualificationRecord.__dataclass_fields__ if name != "slot"
    }
    if set(payload) != expected_fields:
        raise ValueError("qualification JSONL record fields do not match the schema")
    return _record_from_dict(_slot_from_dict(payload), payload)


def _marker_process_ids(process_marker: str) -> tuple[int, ...]:
    if os.name != "posix" or not Path("/proc").is_dir():
        return ()
    marker = f"PEAKAWARE_QUALIFICATION_MARKER={process_marker}".encode("utf-8")
    matches: list[int] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        if pid == os.getpid():
            continue
        try:
            environ = (candidate / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in environ:
            matches.append(pid)
    return tuple(sorted(matches))


def _cleanup_marker_processes(process_marker: str) -> None:
    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        pids = _marker_process_ids(process_marker)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal_value)
            except ProcessLookupError:
                pass
        for _ in range(100):
            if not _marker_process_ids(process_marker):
                return
            time.sleep(0.02)
    remaining = _marker_process_ids(process_marker)
    if remaining:
        raise RuntimeError(f"qualification marker processes still exist: {remaining}")


def _terminate_process(process: mp.Process, process_marker: str | None = None) -> None:
    process.join(0.0)
    process_group: int | None = None
    if os.name == "posix" and process.pid is not None:
        try:
            candidate = os.getpgid(process.pid)
        except ProcessLookupError:
            candidate = None
        if candidate == process.pid and candidate != os.getpgrp():
            process_group = candidate
    if process_group is not None:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.join(2.0)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join(3.0)
        for _ in range(100):
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError(f"qualification worker process group {process_group} still exists")
    else:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(3.0)
    if process.is_alive():
        raise RuntimeError(f"failed to terminate qualification worker pid={process.pid}")
    if process_marker is not None:
        _cleanup_marker_processes(process_marker)


def _timeout_stage(last_stage: str) -> str:
    if last_stage in {"method_prepare", "measurement_setup"}:
        return "compile_timeout"
    if last_stage in {"correctness", "measurement"}:
        return "runtime_timeout"
    return "infra_timeout"


_IDENTITY_METHOD_IDS = {
    "all_save": {"all_save"},
    "pytorch_min_cut": {"pytorch_min_cut", "pytorch_aot_min_cut"},
    "block_ac": {"block_ac", "block_activation_checkpoint"},
    "sac": {"sac", "selective_activation_checkpoint"},
    "peakaware": {"peakaware"},
}


def _identity_matches_backend(identity: Mapping[str, Any], backend: str) -> bool:
    protocol = identity.get("compiler_protocol")
    if not isinstance(protocol, str) or not protocol:
        return False
    if protocol.startswith(f"{backend}:"):
        return True
    return backend == "aot_eager" and protocol == "aot_lowered_graphmodule_eager"


def _finite_non_negative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and torch.isfinite(torch.tensor(float(value))).item()
        and value >= 0
    )


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _timing_field_names(prefix: str) -> set[str]:
    return {
        f"{prefix}_wall_us",
        f"{prefix}_event_us",
        f"{prefix}_event_wall_abs_diff_us",
        f"{prefix}_event_wall_relative_diff",
        f"{prefix}_timing_qualified",
    }


def _validate_timing_fields(sample: Mapping[str, Any], prefix: str, max_gap: float) -> None:
    wall = sample.get(f"{prefix}_wall_us")
    event = sample.get(f"{prefix}_event_us")
    absolute = sample.get(f"{prefix}_event_wall_abs_diff_us")
    relative = sample.get(f"{prefix}_event_wall_relative_diff")
    qualified = sample.get(f"{prefix}_timing_qualified")
    if not _finite_non_negative_number(wall):
        raise ValueError(f"{prefix} wall timing is invalid")
    if event is None:
        if absolute is not None or relative is not None or qualified is not None:
            raise ValueError(f"{prefix} event timing nullability is inconsistent")
    else:
        if not all(_finite_non_negative_number(value) for value in (event, absolute)):
            raise ValueError(f"{prefix} event timing is invalid")
        expected_relative = absolute / event if event > 0 else None
        expected_qualified = expected_relative <= max_gap if expected_relative is not None else None
        if (
            relative != expected_relative
            or qualified != expected_qualified
            or (expected_qualified is not None and type(qualified) is not bool)
        ):
            raise ValueError(f"{prefix} event timing derivation is inconsistent")


def _validate_measurement_samples(
    raw: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    slot: QualificationSlot,
    matrix: Mapping[str, Any],
) -> int:
    repeat_count = matrix["repeat_count"]
    warmup = raw.get("warmup_samples")
    overall = raw.get("overall_samples")
    phase = raw.get("phase_samples")
    combined = raw.get("raw_samples")
    max_gap = aggregate.get("max_event_wall_relative_gap")
    trend_tolerance = aggregate.get("warmup_trend_relative_tolerance")
    if not _finite_non_negative_number(max_gap) or not _finite_non_negative_number(trend_tolerance):
        raise ValueError("measurement aggregate tolerances are invalid")
    if set(raw) != {"warmup_samples", "overall_samples", "phase_samples", "raw_samples"}:
        raise ValueError("measurement raw data has an invalid structure")
    if not all(isinstance(samples, list) for samples in (warmup, overall, phase, combined)):
        raise ValueError("measurement raw trajectories must be lists")
    registered_warmup = matrix["warmup_steps_by_backend"][slot.backend]
    if not registered_warmup <= len(warmup) <= 30:
        raise ValueError("measurement warmup sample count is invalid")
    if len(overall) != repeat_count or len(phase) != repeat_count:
        raise ValueError("measurement raw sample count does not match manifest repeat_count")
    if len(combined) != repeat_count:
        raise ValueError("measurement raw combined sample count is invalid")
    for index, sample in enumerate(warmup):
        if not isinstance(sample, Mapping):
            raise ValueError("warmup sample must be an object")
        if sample.get("warmup_index") != index or sample.get("trajectory") != "warmup":
            raise ValueError("warmup sample identity is invalid")
        if sample.get("trajectory_order") != 0:
            raise ValueError("warmup sample trajectory order is invalid")
        allowed_fields = {
            "warmup_index",
            "trajectory",
            "trajectory_order",
            "last5_wall_strictly_decreasing",
            "last5_event_strictly_decreasing",
        } | _timing_field_names("warmup")
        if not _timing_field_names("warmup") <= set(sample) or set(sample) - allowed_fields:
            raise ValueError("warmup sample fields are invalid")
        _validate_timing_fields(sample, "warmup", max_gap)
        for optional in ("last5_wall_strictly_decreasing", "last5_event_strictly_decreasing"):
            if optional in sample and sample[optional] is not None and type(sample[optional]) is not bool:
                raise ValueError(f"warmup sample field {optional} is invalid")
    for index, sample in enumerate(overall):
        if not isinstance(sample, Mapping) or sample.get("repeat_index") != index:
            raise ValueError("overall repeat indexes are incomplete")
        if sample.get("trajectory") != "overall" or sample.get("trajectory_order") != 1:
            raise ValueError("overall sample trajectory is invalid")
        expected_fields = {
            "repeat_index",
            "trajectory",
            "trajectory_order",
            "overall_peak_bytes",
            "overall_reserved_peak_bytes",
        } | _timing_field_names("overall")
        if set(sample) != expected_fields:
            raise ValueError("overall sample fields are invalid")
        _validate_timing_fields(sample, "overall", max_gap)
        for key in ("overall_peak_bytes", "overall_reserved_peak_bytes"):
            if not _non_negative_int(sample.get(key)):
                raise ValueError(f"overall sample byte field {key} must be an integer")
    for index, sample in enumerate(phase):
        if not isinstance(sample, Mapping) or sample.get("repeat_index") != index:
            raise ValueError("phase repeat indexes are incomplete")
        if sample.get("trajectory") != "phase" or sample.get("trajectory_order") != 2:
            raise ValueError("phase sample trajectory is invalid")
        expected_fields = {
            "repeat_index",
            "trajectory",
            "trajectory_order",
            "after_fw_allocated_bytes",
            "after_fw_reserved_bytes",
            "phase_step_wall_us",
            "phase_step_event_us",
        }
        for phase_name in ("fw", "bw", "optimizer"):
            expected_fields |= _timing_field_names(phase_name)
            expected_fields |= {f"{phase_name}_peak_bytes", f"{phase_name}_reserved_peak_bytes"}
        if set(sample) != expected_fields:
            raise ValueError("phase sample fields are invalid")
        for phase_name in ("fw", "bw", "optimizer"):
            _validate_timing_fields(sample, phase_name, max_gap)
            for suffix in ("peak_bytes", "reserved_peak_bytes"):
                key = f"{phase_name}_{suffix}"
                if not _non_negative_int(sample.get(key)):
                    raise ValueError(f"phase sample byte field {key} must be an integer")
        for key in ("after_fw_allocated_bytes", "after_fw_reserved_bytes"):
            if not _non_negative_int(sample.get(key)):
                raise ValueError(f"phase sample byte field {key} must be an integer")
        expected_wall = sum(sample[f"{name}_wall_us"] for name in ("fw", "bw", "optimizer"))
        if sample.get("phase_step_wall_us") != expected_wall:
            raise ValueError("phase step wall aggregate is inconsistent")
        events = [sample[f"{name}_event_us"] for name in ("fw", "bw", "optimizer")]
        expected_event = sum(events) if all(value is not None for value in events) else None
        if sample.get("phase_step_event_us") != expected_event:
            raise ValueError("phase step event aggregate is inconsistent")
    expected_combined = [
        {
            "repeat_index": index,
            "trajectory_order": ["overall", "phase"],
            "overall": overall[index],
            "phase": phase[index],
        }
        for index in range(repeat_count)
    ]
    if canonical_json(combined) != canonical_json(expected_combined):
        raise ValueError("measurement raw_samples do not match overall/phase trajectories")

    if aggregate.get("publication_backend") != slot.backend:
        raise ValueError("measurement backend does not match the slot")
    if aggregate.get("measurement_repeats") != repeat_count:
        raise ValueError("measurement aggregate repeats do not match the manifest")
    if aggregate.get("warmup_initial_steps") != registered_warmup:
        raise ValueError("measurement aggregate warmup does not match the manifest")
    if aggregate.get("measurement_warmup_steps") != len(warmup):
        raise ValueError("measurement aggregate warmup sample count is inconsistent")
    from peakaware.runtime.measure import _aggregate_measurements

    rebuilt = _aggregate_measurements(
        list(warmup),
        list(phase),
        list(overall),
        len(warmup),
        trend_tolerance,
        max_gap,
    )
    window_qualified = [rebuilt[f"{name}_timing_qualified"] for name in ("fw", "bw", "optimizer", "overall")]
    warmup_timing = [sample["warmup_timing_qualified"] for sample in warmup]
    timing_qualified = all(value is True for value in (*window_qualified, *warmup_timing))
    event_trend = rebuilt["warmup_event_trend_stable"]
    trend_qualified = rebuilt["warmup_wall_trend_stable"] is True and event_trend in {True, None}
    reasons = []
    if not timing_qualified:
        reasons.append("event_wall_gap_or_event_unavailable")
    if not trend_qualified:
        reasons.append("warmup_last5_trend_unstable")
    rebuilt.update(
        {
            "publication_backend": slot.backend,
            "warmup_initial_steps": registered_warmup,
            "warmup_auto_extended_steps": len(warmup) - registered_warmup,
            "warmup_reached_max_steps": len(warmup) == 30,
            "timing_qualified": timing_qualified,
            "warmup_trend_qualified": trend_qualified,
            "publication_qualified": timing_qualified and trend_qualified,
            "publication_status": (
                "qualified" if timing_qualified and trend_qualified
                else "timing_unqualified" if not timing_qualified
                else "warmup_unqualified"
            ),
            "publication_unqualified_reasons": reasons,
        }
    )
    for key in ("warmup_samples", "phase_samples", "overall_samples", "raw_samples"):
        rebuilt.pop(key)
    measured_peak = max(sample["overall_peak_bytes"] for sample in overall)
    rebuilt["per_run_max_overall_allocated_bytes"] = measured_peak
    if canonical_json(rebuilt) != canonical_json(aggregate):
        raise ValueError("measurement aggregate does not match the raw trajectories and manifest")
    return measured_peak


def _validate_ready_runtime_provenance(
    slot: QualificationSlot,
    identity: Mapping[str, Any],
) -> None:
    provenance = identity["provenance"]
    if slot.method == "all_save":
        observations = identity.get("runtime_observations")
        if (
            provenance.get("partition_fn") != "torch._functorch.partitioners.default_partition"
            or len(str(provenance.get("partition_fn_source_sha256", ""))) != 64
            or not isinstance(observations, Mapping)
            or observations.get("fw_compile_count", 0) < 1
            or observations.get("bw_compile_count", 0) < 1
        ):
            raise ValueError("all-save identity lacks default_partition runtime evidence")
    elif slot.method == "pytorch_min_cut":
        conversion = provenance.get("budget_conversion")
        if (
            not isinstance(conversion, Mapping)
            or conversion.get("status") != "converted"
            or conversion.get("physical_budget_bytes") != slot.memory_budget_bytes
        ):
            raise ValueError("min-cut identity lacks an auditable absolute-budget conversion")
    elif slot.method == "block_ac":
        if (
            not identity.get("region_paths")
            or provenance.get("checkpoint_call_count", 0) < 1
            or provenance.get("recompute_count", 0) < 1
        ):
            raise ValueError("block AC identity lacks region and recomputation provenance")
    elif slot.method == "sac":
        if (
            not identity.get("region_paths")
            or provenance.get("must_save_count", 0) < 1
            or provenance.get("prefer_recompute_count", 0) < 1
            or not provenance.get("policy_hash")
        ):
            raise ValueError("SAC identity lacks policy decision provenance")
    elif slot.method == "peakaware":
        if (
            provenance.get("requested_method") != "peakaware"
            or provenance.get("requested_backend") != slot.backend
            or not identity.get("plan_id")
            or identity.get("aot_partition_runtime") is not True
        ):
            raise ValueError("PeakAware identity lacks selected lowered-runtime provenance")


def validate_qualification_record(
    record: QualificationRecord,
    manifest: Mapping[str, Any],
) -> None:
    validate_qualification_slot(record.slot, manifest)
    validate_record_manifest_reference(manifest, record.to_dict())
    identity = _decode_or_none(record.runtime_identity)
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise ValueError("runtime identity must be an object")
        if identity.get("method_id") not in _IDENTITY_METHOD_IDS[record.slot.method]:
            raise ValueError("runtime identity method does not match the slot method")
        if not _identity_matches_backend(identity, record.slot.backend):
            raise ValueError("runtime identity compiler protocol does not match the slot backend")
        provenance = identity.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("runtime identity requires non-empty provenance")

    correctness_payload = _decode_or_none(record.correctness_report)
    if correctness_payload is not None:
        if not isinstance(correctness_payload, Mapping):
            raise ValueError("correctness report must be an object")
        if (
            correctness_payload.get("protocol") != "three_optimizer_steps_v1"
            or correctness_payload.get("steps") != 3
            or correctness_payload.get("atol") != 1e-5
            or correctness_payload.get("rtol") != 1e-4
        ):
            raise ValueError("correctness report protocol or tolerances are invalid")
        boolean_fields = (
            "passed",
            "values_match",
            "rng_match",
            "gradients_finite",
            "parameters_finite",
            "buffers_finite",
            "optimizer_state_finite",
            "python_random_match",
            "numpy_random_match",
            "torch_cpu_rng_match",
            "torch_cuda_rng_match",
        )
        if any(type(correctness_payload.get(key)) is not bool for key in boolean_fields):
            raise ValueError("correctness report boolean verdicts are incomplete")
        for key in ("max_abs_error", "max_rel_error"):
            if not _finite_non_negative_number(correctness_payload.get(key)):
                raise ValueError(f"correctness numeric field {key} is invalid")

    raw_payload = _decode_or_none(record.measurement_raw)
    aggregate_payload = _decode_or_none(record.measurement_aggregate)
    measured_peak: int | None = None
    if (raw_payload is None) != (aggregate_payload is None):
        raise ValueError("measurement raw and aggregate fields must appear together")
    if raw_payload is not None:
        if not isinstance(raw_payload, Mapping) or not isinstance(aggregate_payload, Mapping):
            raise ValueError("measurement payloads must be objects")
        provisional_overall = raw_payload.get("overall_samples")
        if record.status in {"ok", "budget_violation"} and isinstance(provisional_overall, list):
            provisional_peaks = [sample.get("overall_peak_bytes") for sample in provisional_overall]
            if provisional_peaks and all(_non_negative_int(value) for value in provisional_peaks):
                provisional_status = (
                    "ok" if max(provisional_peaks) <= record.slot.memory_budget_bytes else "budget_violation"
                )
                if record.status != provisional_status:
                    raise ValueError("record budget status does not match the measured per-run peak")
        matrix = manifest["qualification_run"]["matrix"]
        measured_peak = _validate_measurement_samples(
            raw_payload,
            aggregate_payload,
            record.slot,
            matrix,
        )
        if type(aggregate_payload.get("publication_qualified")) is not bool:
            raise ValueError("measurement aggregate qualification verdict is invalid")
        if aggregate_payload.get("per_run_max_overall_allocated_bytes") != measured_peak:
            raise ValueError("measurement aggregate peak does not match raw overall samples")

    if record.status == "unsupported":
        if identity is None or identity.get("status") != "unsupported" or not identity.get("fallback_reason"):
            raise ValueError("unsupported record has no explicit fallback identity")
        provenance = identity["provenance"]
        if provenance.get("requested_method") != record.slot.method:
            raise ValueError("unsupported identity requested_method does not match the slot")
        if provenance.get("requested_backend") != record.slot.backend:
            raise ValueError("unsupported identity requested_backend does not match the slot")
        return

    if identity is not None and identity.get("status") == "ready":
        _validate_ready_runtime_provenance(record.slot, identity)

    if record.status not in {"ok", "budget_violation"}:
        return
    if identity is None or identity.get("status") != "ready" or identity.get("fallback_reason") is not None:
        raise ValueError("qualified record runtime identity is not ready")
    correctness = correctness_payload
    required_correctness = {
        "protocol": "three_optimizer_steps_v1",
        "steps": 3,
        "passed": True,
        "values_match": True,
        "rng_match": True,
        "gradients_finite": True,
        "parameters_finite": True,
        "buffers_finite": True,
        "optimizer_state_finite": True,
        "python_random_match": True,
        "numpy_random_match": True,
        "torch_cpu_rng_match": True,
        "torch_cuda_rng_match": True,
        "atol": 1e-5,
        "rtol": 1e-4,
        "first_mismatch": None,
    }
    if not isinstance(correctness, Mapping):
        raise ValueError("qualified record correctness report is missing")
    for key, expected in required_correctness.items():
        if correctness.get(key) != expected:
            raise ValueError(f"correctness field {key} is not qualified")
    for key in ("max_abs_error", "max_rel_error"):
        if not _finite_non_negative_number(correctness.get(key)):
            raise ValueError(f"correctness numeric field {key} is invalid")
    raw = raw_payload
    aggregate = aggregate_payload
    if not isinstance(raw, Mapping) or not isinstance(aggregate, Mapping):
        raise ValueError("qualified record measurement is missing")
    if aggregate.get("publication_qualified") is not True:
        raise ValueError("measurement aggregate is not publication-qualified")
    assert measured_peak is not None
    expected_status = "ok" if measured_peak <= record.slot.memory_budget_bytes else "budget_violation"
    if record.status != expected_status:
        raise ValueError("record budget status does not match the measured per-run peak")
    environment = _decode_or_none(record.environment)
    if not isinstance(environment, Mapping) or environment.get("pid") != record.process_id:
        raise ValueError("record environment PID does not match the worker process")


def _run_qualification_slot(
    context: Any,
    slot: QualificationSlot,
    manifest: Mapping[str, Any],
    *,
    process_marker: str,
    warmup_steps: int,
    repeat_count: int,
    timeout_s: float,
) -> QualificationRecord:
    result_channel = None
    worker_channel = None
    process = None
    started = time.monotonic()
    try:
        result_channel, worker_channel = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_entry,
            args=(worker_channel, slot, manifest, process_marker, warmup_steps, repeat_count),
            name=f"peakaware-qualification-{slot.slot_id[:12]}",
        )
        try:
            process.start()
        except Exception as exc:
            record = _record_failure(
                slot,
                process_id=process.pid,
                status="infra_failure",
                stage="spawn_start",
                error=exc,
                elapsed_seconds=time.monotonic() - started,
                last_progress_stage="spawn",
            )
            validate_qualification_record(record, manifest)
            return record
        worker_channel.close()
        worker_channel = None
        process_id = process.pid
        deadline = started + timeout_s
        payload: Mapping[str, Any] | None = None
        channel_eof = False
        last_stage = "spawn"
        last_worker_elapsed = 0.0
        while payload is None and not channel_eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not result_channel.poll(remaining):
                break
            try:
                message = result_channel.recv()
            except EOFError:
                channel_eof = True
                break
            if not isinstance(message, Mapping):
                payload = {
                    "worker_transport_error": {
                        "type": "MessageTypeError",
                        "message": "worker message is not an object",
                    }
                }
                break
            if message.get("kind") == "progress":
                if message.get("slot_id") != slot.slot_id or message.get("process_id") != process_id:
                    payload = {
                        "worker_transport_error": {
                            "type": "ProgressIdentityError",
                            "message": "worker progress slot identity or PID mismatch",
                        }
                    }
                    break
                last_stage = str(message["stage"])
                last_worker_elapsed = float(message["elapsed_seconds"])
                continue
            payload = message.get("payload") if message.get("kind") == "record" else message

        if channel_eof:
            record = _record_failure(
                slot,
                process_id=process_id,
                status="infra_failure",
                stage="worker_transport",
                error=RuntimeError(f"worker closed its channel with exit code {process.exitcode}"),
                elapsed_seconds=time.monotonic() - started,
                last_progress_stage=last_stage,
            )
        elif payload is None:
            process.join(0.0)
            if process.is_alive():
                record = _record_failure(
                    slot,
                    process_id=process_id,
                    status="timeout",
                    stage=_timeout_stage(last_stage),
                    error=TimeoutError(
                        f"slot exceeded {timeout_s:g} seconds; last stage={last_stage}, "
                        f"worker elapsed={last_worker_elapsed:g}"
                    ),
                    elapsed_seconds=time.monotonic() - started,
                    last_progress_stage=last_stage,
                )
            else:
                record = _record_failure(
                    slot,
                    process_id=process_id,
                    status="infra_failure",
                    stage="worker_transport",
                    error=RuntimeError(f"worker exited with code {process.exitcode} without a record"),
                    elapsed_seconds=time.monotonic() - started,
                    last_progress_stage=last_stage,
                )
        else:
            process.join(5.0)
            if process.is_alive():
                record = _record_failure(
                    slot,
                    process_id=process_id,
                    status="infra_failure",
                    stage="worker_transport",
                    error=RuntimeError("worker sent a record but did not exit"),
                    elapsed_seconds=time.monotonic() - started,
                    last_progress_stage=last_stage,
                )
            elif "worker_transport_error" in payload:
                error = payload["worker_transport_error"]
                record = _record_failure(
                    slot,
                    process_id=process_id,
                    status="infra_failure",
                    stage="worker_transport",
                    error=RuntimeError(f"{error['type']}: {error['message']}"),
                    elapsed_seconds=time.monotonic() - started,
                    last_progress_stage=last_stage,
                )
            elif payload.get("slot_id") != slot.slot_id or payload.get("process_id") != process_id:
                record = _record_failure(
                    slot,
                    process_id=process_id,
                    status="infra_failure",
                    stage="worker_transport",
                    error=RuntimeError("worker record slot identity or PID mismatch"),
                    elapsed_seconds=time.monotonic() - started,
                    last_progress_stage=last_stage,
                )
            else:
                record = _record_from_dict(slot, payload)
        validate_qualification_record(record, manifest)
        return record
    finally:
        for channel in (worker_channel, result_channel):
            if channel is not None:
                channel.close()
        if process is not None and process.pid is not None:
            _terminate_process(process, process_marker)
        else:
            _cleanup_marker_processes(process_marker)


def run_qualification_slots(
    slots: Sequence[QualificationSlot],
    manifest: Mapping[str, Any],
    *,
    timeout_s: float,
    warmup_steps: int | None = None,
    repeat_count: int = 20,
) -> tuple[QualificationRecord, ...]:
    _validate_qualification_preregistration(manifest)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if repeat_count < 20:
        raise ValueError("publication qualification requires at least 20 repeats")
    run = manifest.get("qualification_run")
    matrix = None if not isinstance(run, Mapping) else run.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("manifest has no preregistered qualification matrix")
    if repeat_count != matrix.get("repeat_count") or float(timeout_s) != matrix.get("timeout_s"):
        raise ValueError("runtime repeat_count/timeout_s do not match manifest preregistration")
    registered_warmups = matrix.get("warmup_steps_by_backend")
    if not isinstance(registered_warmups, Mapping):
        raise ValueError("manifest has no preregistered backend warmup protocol")
    if warmup_steps is not None:
        invalid_backends = {
            slot.backend
            for slot in slots
            if warmup_steps != (5 if slot.backend == "aot_eager" else 10)
        }
        if invalid_backends:
            expected = sorted(5 if backend == "aot_eager" else 10 for backend in invalid_backends)
            raise ValueError(f"warmup_steps must match backend publication minimum(s): {expected}")
    provided_ids = [slot.slot_id for slot in slots]
    if len(provided_ids) != len(set(provided_ids)):
        raise ValueError("qualification slots contain duplicate slot IDs")
    for slot in slots:
        validate_qualification_slot(slot, manifest)
    execution_order = manifest["qualification_run"]["execution_order"]
    order_index = {slot_id: index for index, slot_id in enumerate(execution_order)}
    ordered_slots = sorted(slots, key=lambda item: order_index[item.slot_id])
    context = mp.get_context("spawn")
    invocation_token = uuid.uuid4().hex
    records = []
    for slot in ordered_slots:
        slot_warmup = warmup_steps if warmup_steps is not None else registered_warmups[slot.backend]
        if slot_warmup != registered_warmups[slot.backend]:
            raise ValueError("runtime warmup does not match manifest preregistration")
        process_marker = f"{slot.run_id}:{slot.slot_id}:{slot.attempt_id}:{invocation_token}"
        records.append(
            _run_qualification_slot(
                context,
                slot,
                manifest,
                process_marker=process_marker,
                warmup_steps=slot_warmup,
                repeat_count=repeat_count,
                timeout_s=timeout_s,
            )
        )
    return tuple(records)


def summarize_qualification(
    records: Sequence[QualificationRecord],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_qualification_preregistration(manifest)
    run = manifest.get("qualification_run")
    if not isinstance(run, Mapping):
        raise ValueError("manifest has no qualification_run preregistration")
    expected_ids = list(run.get("expected_slot_ids", ()))
    expected_count = run.get("expected_slot_count")
    if expected_count != len(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("manifest expected slot preregistration is inconsistent")
    actual_ids = [record.slot.slot_id for record in records]
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    status_counts = {status: 0 for status in sorted(SUPPORTED_STATUSES)}
    method_counts: dict[str, dict[str, dict[str, int]]] = {
        backend: {
            method: {"qualified": 0, "unsupported": 0, "failure": 0}
            for method in run["matrix"]["methods"]
        }
        for backend in run["matrix"]["backends"]
    }
    for record in records:
        validate_qualification_record(record, manifest)
        status_counts[record.status] += 1
        category = (
            "qualified"
            if record.status in {"ok", "budget_violation"}
            else "unsupported"
            if record.status == "unsupported"
            else "failure"
        )
        method_counts[record.slot.backend][record.slot.method][category] += 1
    order_positions = {slot_id: index for index, slot_id in enumerate(run["execution_order"])}
    relative_order_valid = actual_ids == sorted(actual_ids, key=order_positions.__getitem__)
    expected_per_method = (
        len(run["matrix"]["tasks"])
        * len(run["matrix"]["memory_budgets_bytes"])
        * run["matrix"]["replicates"]
    )
    required_method_coverage = {
        backend: {
            method: counts["qualified"] == expected_per_method
            for method, counts in methods.items()
        }
        for backend, methods in method_counts.items()
    }
    complete_slot_coverage = (
        len(actual_ids) == expected_count
        and len(actual_ids) == len(actual_set)
        and actual_set == expected_set
    )
    qualification_passed = (
        complete_slot_coverage
        and relative_order_valid
        and all(
            covered
            for methods in required_method_coverage.values()
            for covered in methods.values()
        )
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "record_count": len(records),
        "expected_slot_count": expected_count,
        "status_counts": status_counts,
        "complete_slot_coverage": complete_slot_coverage,
        "record_order_valid": relative_order_valid,
        "method_qualification": method_counts,
        "required_method_coverage": required_method_coverage,
        "qualification_passed": qualification_passed,
        "missing_slot_ids": sorted(expected_set - actual_set),
        "unexpected_slot_ids": sorted(actual_set - expected_set),
        "duplicate_slot_ids": sorted(
            slot_id for slot_id in actual_set if actual_ids.count(slot_id) > 1
        ),
        "run_ids": sorted({record.slot.run_id for record in records}),
    }


def write_qualification_artifacts(
    records: Sequence[QualificationRecord],
    manifest: Mapping[str, Any],
    *,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    t1_output: str | Path,
) -> Path:
    output_path = Path(output_jsonl)
    manifest_path = Path(manifest_json)
    t1_path = Path(t1_output)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    targets = (output_path, manifest_path, t1_path, summary_path)
    resolved_targets = [path.resolve(strict=False) for path in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("publication artifact paths must not alias each other")
    resolved_parents = {path.parent.resolve(strict=False) for path in targets}
    if len(resolved_parents) != 1:
        raise ValueError("publication artifacts must share one parent directory for commit-marker semantics")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        committed = summary_path.exists()
        state = "committed" if committed else "uncommitted partial"
        raise FileExistsError(f"publication artifact paths contain {state} files: {existing}")
    summary = summarize_qualification(records, manifest)
    if not summary["complete_slot_coverage"] or not summary["record_order_valid"]:
        raise ValueError("artifact writer requires the complete record set in manifest execution order")
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    data_payloads = (
        (output_path, "".join(canonical_json(record.to_dict()) + "\n" for record in records)),
        (manifest_path, canonical_json(manifest, indent=2) + "\n"),
        (t1_path, render_t1_markdown(manifest)),
    )
    committed_summary = {
        **summary,
        "artifact_commit": {
            "protocol": "same_directory_data_then_commit_marker_v1",
            "committed": True,
            "files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "size_bytes": len(payload.encode("utf-8")),
                }
                for path, payload in data_payloads
            ],
            "crash_semantics": (
                "data files without this commit marker are uncommitted partial output and must not be consumed"
            ),
        },
    }
    payloads = (*data_payloads, (summary_path, canonical_json(committed_summary, indent=2) + "\n"))
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for path, payload in payloads:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            for index, (temporary, (target, _payload)) in enumerate(zip(temporary_paths, payloads)):
                os.link(temporary, target)
                published_paths.append(target)
                if index == len(data_payloads) - 1 or index == len(payloads) - 1:
                    os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for published in reversed(published_paths):
            try:
                published.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return summary_path


def validate_qualification_artifact_bundle(
    *,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    t1_output: str | Path,
) -> Mapping[str, Any]:
    output_path = Path(output_jsonl)
    manifest_path = Path(manifest_json)
    t1_path = Path(t1_output)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    paths = (output_path, manifest_path, t1_path, summary_path)
    if len({path.parent.resolve(strict=False) for path in paths}) != 1:
        raise ValueError("artifact bundle paths do not share one parent directory")
    if len({path.resolve(strict=False) for path in paths}) != len(paths):
        raise ValueError("artifact bundle paths alias each other")
    if not summary_path.exists():
        raise ValueError("artifact bundle is uncommitted because the commit marker is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("artifact commit marker must be an object")
    commit = summary.get("artifact_commit")
    if not isinstance(commit, Mapping) or commit.get("committed") is not True:
        raise ValueError("artifact commit marker is invalid")
    file_entries = commit.get("files")
    if (
        commit.get("protocol") != "same_directory_data_then_commit_marker_v1"
        or not isinstance(file_entries, list)
        or len(file_entries) != 3
        or any(not isinstance(entry, Mapping) for entry in file_entries)
    ):
        raise ValueError("artifact commit marker file manifest is invalid")
    expected_files = {entry.get("name"): entry for entry in file_entries}
    if set(expected_files) != {output_path.name, manifest_path.name, t1_path.name}:
        raise ValueError("artifact commit marker file set is invalid")
    for path in (output_path, manifest_path, t1_path):
        entry = expected_files.get(path.name)
        if entry is None or not path.exists():
            raise ValueError(f"committed artifact file is missing from marker or disk: {path.name}")
        payload = path.read_bytes()
        if len(payload) != entry.get("size_bytes"):
            raise ValueError(f"committed artifact size mismatch: {path.name}")
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            raise ValueError(f"committed artifact hash mismatch: {path.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("qualification manifest artifact must be an object")
    _validate_qualification_preregistration(manifest)
    record_lines = output_path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in record_lines):
        raise ValueError("qualification JSONL contains a blank record")
    records: list[QualificationRecord] = []
    for line_number, line in enumerate(record_lines, start=1):
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("record must be an object")
            record = _artifact_record_from_dict(payload)
            validate_qualification_record(record, manifest)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid qualification JSONL record at line {line_number}: {exc}") from exc
        records.append(record)
    rebuilt_summary = summarize_qualification(records, manifest)
    marker_summary = {key: value for key, value in summary.items() if key != "artifact_commit"}
    if canonical_json(marker_summary) != canonical_json(rebuilt_summary):
        raise ValueError("artifact commit marker summary does not match validated records and manifest")
    return summary


def cleanup_uncommitted_qualification_artifacts(
    *,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    t1_output: str | Path,
) -> None:
    output_path = Path(output_jsonl)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    if summary_path.exists():
        raise ValueError("refusing to clean a committed qualification artifact bundle")
    for path in (output_path, Path(manifest_json), Path(t1_output)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
