from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig, normalize_float_dtype_name
from peakaware.contracts import (
    AnalysisBundle,
    CacheStats,
    CapturedJointGraph,
    DryRunResult,
    EvaluatedPlan,
    HardwareSpec,
    JointTrainingIR,
    LoweredPartition,
    MeasuredExecutable,
    OptimizedTrainingResult,
    TrainingRequest,
)
from peakaware.cache import (
    build_compiled_artifact_key,
    build_plan_evaluation_key,
    load_analysis_cache,
    load_capture_cache,
    load_executable_measurement_cache,
    store_analysis_cache,
    store_capture_cache,
    store_executable_measurement_cache,
)
from peakaware.cost.composite import build_composite_provider
from peakaware.errors import InfeasibleBudgetError
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import (
    analyze_coarse_feasibility,
    analyze_refined_feasibility,
    build_optimizer_spec,
)
from peakaware.partition.aot import lower_partition_graphs, partition_default_graph
from peakaware.partition.verifier import run_aot_eager_dry_run
from peakaware.plugins import ServiceKind, build_default_registry
from peakaware.runtime.executor import build_training_step_executor, make_measured_executable
from peakaware.runtime.isolation import run_in_worker_process
from peakaware.search.engine import apply_early_stop_policy, search_plans_with_diagnostics


CAPTURE_SCHEMA_VERSION = "capture-v2-guarded-graph-key"
ANALYSIS_SCHEMA_VERSION = "analysis-v3-search-diagnostics"


def _hardware_spec(args: tuple[Any, ...], kwargs: dict[str, Any]) -> HardwareSpec:
    devices = [value.device for value in list(args) + list(kwargs.values()) if isinstance(value, Tensor)]
    device = str(devices[0]) if devices else ("cuda" if torch.cuda.is_available() else "cpu")
    total = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        total = int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    return HardwareSpec(device=device, cuda_available=torch.cuda.is_available(), total_memory_bytes=total)


def _request_key(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: int,
    config: PeakAwareConfig,
) -> str:
    h = hashlib.sha256()
    h.update(model.__class__.__qualname__.encode("utf-8"))
    h.update(str(budget).encode("utf-8"))
    h.update(repr(config.precision_fingerprint()).encode("utf-8"))
    for value in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(value, Tensor):
            h.update(str(tuple(value.shape)).encode("utf-8"))
            h.update(str(value.dtype).encode("utf-8"))
            h.update(str(value.device).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_root(config: PeakAwareConfig) -> Path | None:
    return None if config.cache_root is None else Path(config.cache_root)


def _capture_cache_provenance(request: TrainingRequest) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "request_key": request.request_key,
        "capture_backend": request.config.capture_backend,
        "precision": dict(request.config.precision_fingerprint()),
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
    }


def _can_cache_capture(config: PeakAwareConfig) -> bool:
    # PyTorch 2.13 AOT GraphModules can become invalid after pickle round-trips.
    return config.capture_backend == "fx"


def _try_store_capture_cache(
    root: Path,
    key: str,
    capture: CapturedJointGraph,
    provenance: dict[str, Any],
) -> bool:
    try:
        store_capture_cache(root, key, capture, provenance)
    except Exception:
        return False
    return True


def _analysis_cache_key(
    graph_key: str,
    optimizer_name: str,
    memory_budget_bytes: int,
    config: PeakAwareConfig,
) -> str:
    return build_plan_evaluation_key(
        analysis_key=graph_key,
        optimizer_mode=optimizer_name,
        cost_database_version=str(config.profile_db_path or "none"),
        search_policy_version=f"peakaware-topk{config.top_k}-dh{int(config.enable_diagnostic_hints)}",
        budget_bucket=memory_budget_bytes,
    )


def _analysis_cache_provenance(
    request: TrainingRequest,
    graph_key: str,
    optimizer_name: str,
    memory_budget_bytes: int,
    config: PeakAwareConfig,
) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "request_key": request.request_key,
        "graph_key": graph_key,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "optimizer": optimizer_name,
        "budget_bytes": memory_budget_bytes,
        "top_k": config.top_k,
        "enable_diagnostic_hints": config.enable_diagnostic_hints,
        "safety_margin_bytes": config.safety_margin_bytes,
        "safety_margin_ratio": config.safety_margin_ratio,
        "manual_saved_value_ids": [list(sorted(item)) for item in config.manual_saved_value_ids],
        "profile_db_path": str(config.profile_db_path or "none"),
        "precision": dict(config.precision_fingerprint()),
    }


def _rebind_evaluated_plans(evaluated: tuple[EvaluatedPlan, ...], graph_key: str) -> tuple[EvaluatedPlan, ...]:
    return tuple(replace(item, plan=replace(item.plan, graph_key=graph_key)) for item in evaluated)


def _executable_cache_key(
    request: TrainingRequest,
    capture: CapturedJointGraph,
    candidate: EvaluatedPlan,
) -> str:
    if request.config.enable_inductor:
        compiler_version = "inductor"
    elif request.config.enable_compile:
        compiler_version = "aot_eager"
    else:
        compiler_version = "eager"
    return build_compiled_artifact_key(
        lowered_plan_fingerprint=(
            f"{candidate.plan.graph_key}:{candidate.plan.plan_id}:"
            f"{tuple(sorted(candidate.plan.saved_value_ids))}"
        ),
        state_signature=request.request_key,
        input_guards=tuple((guard.name, guard.value) for guard in capture.guards),
        device_capability=request.hardware.device,
        torch_version=torch.__version__,
        compiler_version=compiler_version,
        partition_plugin_version="core",
    )


def _executable_cache_provenance(
    request: TrainingRequest,
    candidate: EvaluatedPlan,
) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "request_key": request.request_key,
        "plan_id": candidate.plan.plan_id,
        "graph_key": candidate.plan.graph_key,
        "correctness_required": True,
    }


def _iter_tensors(value: Any) -> tuple[Tensor, ...]:
    found: list[Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, Tensor):
            found.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)

    collect(value)
    return tuple(found)


def _tensor_devices(tensors: tuple[Tensor, ...]) -> frozenset[str]:
    return frozenset(str(tensor.device) for tensor in tensors)


def _floating_dtype_names(tensors: tuple[Tensor, ...]) -> frozenset[str]:
    return frozenset(
        normalize_float_dtype_name(str(tensor.dtype))
        for tensor in tensors
        if tensor.is_floating_point()
    )


def _validate_request(
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    memory_budget_bytes: int,
    config: PeakAwareConfig,
) -> None:
    del loss_fn
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple")
    config.validate()
    model_params = {id(p) for p in model.parameters()}
    optim_params = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
    }
    if not optim_params:
        raise ValueError("optimizer has no parameters")
    if not optim_params.issubset(model_params):
        raise ValueError("optimizer contains parameters that are not owned by model")
    example_tensors = _iter_tensors(example_args) + _iter_tensors(example_kwargs)
    if any(value.requires_grad for value in example_tensors):
        raise ValueError("M0 expects training inputs without requires_grad")
    model_tensors = tuple(model.parameters()) + tuple(model.buffers())
    optimizer_state_tensors = tuple(
        tensor
        for state in optimizer.state.values()
        for tensor in _iter_tensors(state)
    )
    device_sets = [
        devices
        for devices in (
            _tensor_devices(example_tensors),
            _tensor_devices(model_tensors),
            _tensor_devices(optimizer_state_tensors),
        )
        if devices
    ]
    all_devices = frozenset(device for devices in device_sets for device in devices)
    if len(all_devices) > 1:
        raise ValueError(f"M0 expects a single device, got: {sorted(all_devices)}")
    expected_dtype = normalize_float_dtype_name(config.precision_dtype)
    dtype_sets = [
        dtypes
        for dtypes in (
            _floating_dtype_names(example_tensors),
            _floating_dtype_names(model_tensors),
            _floating_dtype_names(optimizer_state_tensors),
        )
        if dtypes
    ]
    all_dtypes = frozenset(dtype for dtypes in dtype_sets for dtype in dtypes)
    if any(dtype != expected_dtype for dtype in all_dtypes):
        raise ValueError(
            f"M0 expects floating tensors to use precision_dtype={expected_dtype}, "
            f"got: {sorted(all_dtypes)}"
        )


def _lower_candidate(capture: CapturedJointGraph, candidate: EvaluatedPlan, ir: JointTrainingIR) -> LoweredPartition:
    if capture.backend == "aot":
        from peakaware.partition.aot import partition_joint_graph

        return partition_joint_graph(
            capture.joint_module,
            candidate.plan,
            ir,
            num_fwd_outputs=capture.num_fwd_outputs,
            static_lifetime_input_indices=capture.static_lifetime_input_indices,
        )
    if capture.fw_module is None and capture.bw_module is None:
        return partition_default_graph(capture.joint_module, candidate.plan, ir)
    return lower_partition_graphs(capture.joint_module, capture.fw_module, capture.bw_module, candidate.plan, ir)


def _dry_run_candidate(
    lowered: LoweredPartition,
    ir: JointTrainingIR,
    *,
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[..., Tensor],
    config: PeakAwareConfig,
) -> DryRunResult:
    return run_aot_eager_dry_run(
        lowered,
        model=model,
        args=example_args,
        kwargs=example_kwargs,
        loss_fn=loss_fn,
        atol=config.atol,
        rtol=config.rtol,
        ir=ir,
    )


def _select_measured_candidate(
    measured: tuple[MeasuredExecutable, ...],
    *,
    memory_budget_bytes: int,
    selection_objective: str,
) -> MeasuredExecutable | None:
    feasible = [
        item
        for item in measured
        if item.correctness_passed and item.measured_peak_bytes <= memory_budget_bytes
    ]
    if not feasible:
        return None
    if selection_objective == "min_time_then_peak":
        feasible.sort(key=lambda item: (item.measured_step_us, item.measured_peak_bytes, item.plan_id))
    else:
        feasible.sort(key=lambda item: (item.measured_peak_bytes, item.measured_step_us, item.plan_id))
    return feasible[0]


@dataclass(frozen=True)
class _CandidateMeasurement:
    plan_id: str
    measured_peak_bytes: int
    measured_step_us: float
    phase_metrics: dict[str, int | float]


@dataclass(frozen=True)
class _CandidateValidation:
    dry_run: DryRunResult
    measurement: _CandidateMeasurement | None
    cache_hit: bool = False


def _validate_and_measure_candidate(payload: dict[str, Any]) -> _CandidateValidation:
    capture = payload["capture"]
    candidate = payload["candidate"]
    ir = payload["ir"]
    request = payload["request"]
    model = payload["model"]
    optimizer = payload["optimizer"]
    loss_fn = payload["loss_fn"]
    config = payload["config"]
    example_args = payload["example_args"]
    example_kwargs = payload["example_kwargs"]

    if capture is None or ir is None:
        capture = capture_joint_graph(request)
        ir, ir_report = build_joint_ir(capture)
        if not ir_report.valid:
            raise ValueError(f"invalid worker IR: {ir_report.errors}")
    lowered = _lower_candidate(capture, candidate, ir)
    dry_run = _dry_run_candidate(
        lowered,
        ir,
        model=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        loss_fn=loss_fn,
        config=config,
    )
    if not (dry_run.abi_valid and dry_run.outputs_match and dry_run.gradients_match):
        return _CandidateValidation(dry_run=dry_run, measurement=None)
    executor = build_training_step_executor(
        model,
        optimizer,
        loss_fn,
        config,
        capture.guards,
        selection_objective=config.selection_objective,
    )
    cache_root = _cache_root(config)
    executable_key = _executable_cache_key(request, capture, candidate)
    executable_provenance = _executable_cache_provenance(request, candidate)
    if cache_root is not None:
        cached = load_executable_measurement_cache(
            cache_root,
            executable_key,
            executor.executable,
            executable_provenance,
        )
        if cached is not None and cached.correctness_passed:
            return _CandidateValidation(
                dry_run=dry_run,
                measurement=_CandidateMeasurement(
                    plan_id=cached.plan_id,
                    measured_peak_bytes=cached.measured_peak_bytes,
                    measured_step_us=cached.measured_step_us,
                    phase_metrics=cached.phase_metrics,
                ),
                cache_hit=True,
            )
    measured = make_measured_executable(
        candidate.plan.plan_id,
        executor,
        example_args,
        example_kwargs,
        candidate.simulation.estimated_peak_bytes,
    )
    if cache_root is not None:
        store_executable_measurement_cache(cache_root, executable_key, measured, executable_provenance)
    return _CandidateValidation(
        dry_run=dry_run,
        measurement=_CandidateMeasurement(
            plan_id=measured.plan_id,
            measured_peak_bytes=measured.measured_peak_bytes,
            measured_step_us=measured.measured_step_us,
            phase_metrics=measured.phase_metrics,
        ),
    )


def _candidate_payload(
    request: TrainingRequest,
    capture: CapturedJointGraph,
    candidate: EvaluatedPlan,
    ir: JointTrainingIR,
    *,
    isolate: bool,
) -> dict[str, Any]:
    return {
        "capture": None if isolate else capture,
        "candidate": candidate,
        "ir": None if isolate else ir,
        "request": request,
        "model": request.model,
        "optimizer": request.optimizer,
        "loss_fn": request.loss_fn,
        "config": request.config,
        "example_args": request.example_args,
        "example_kwargs": request.example_kwargs,
    }


def _measure_candidate_for_parent(
    executor: Any,
    validation: _CandidateValidation,
) -> MeasuredExecutable:
    measurement = validation.measurement
    if measurement is None:
        raise ValueError("candidate measurement is unavailable")
    return MeasuredExecutable(
        plan_id=measurement.plan_id,
        forward_backward=executor.executable,
        measured_peak_bytes=measurement.measured_peak_bytes,
        measured_step_us=measurement.measured_step_us,
        correctness_passed=True,
        phase_metrics=measurement.phase_metrics,
    )


def optimize_training(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    example_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    memory_budget_bytes: int,
    config: PeakAwareConfig | None = None,
) -> OptimizedTrainingResult:
    config = config or PeakAwareConfig()
    example_kwargs = dict(example_kwargs or {})
    _validate_request(model, example_args, example_kwargs, loss_fn, optimizer, memory_budget_bytes, config)
    optimization_start = time.perf_counter()
    optimization_metrics: dict[str, int | float | None] = {}

    def mark_elapsed(name: str, start: float) -> None:
        optimization_metrics[name] = (time.perf_counter() - start) * 1_000_000.0

    model.train()
    setup_start = time.perf_counter()
    registry = build_default_registry(profile_db_path=config.profile_db_path)
    cache_root = _cache_root(config)
    cache_hits: dict[str, int] = {}
    cache_misses: dict[str, int] = {}

    def record_cache(layer: str, hit: bool) -> None:
        target = cache_hits if hit else cache_misses
        target[layer] = target.get(layer, 0) + 1

    optimizer_spec = build_optimizer_spec(optimizer, model)
    request = TrainingRequest(
        model=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=memory_budget_bytes,
        config=config,
        optimizer_spec=optimizer_spec,
        hardware=_hardware_spec(example_args, example_kwargs),
        request_key=_request_key(model, example_args, example_kwargs, memory_budget_bytes, config),
    )
    mark_elapsed("request_setup_us", setup_start)

    coarse_start = time.perf_counter()
    fixed_timeline, coarse = analyze_coarse_feasibility(model, optimizer, memory_budget_bytes)
    mark_elapsed("coarse_feasibility_us", coarse_start)
    if coarse.status == "INFEASIBLE_BY_ACTIVATION_ONLY":
        raise InfeasibleBudgetError(coarse.explanations[0])

    capture_start = time.perf_counter()
    capture_provenance = _capture_cache_provenance(request)
    use_capture_cache = cache_root is not None and _can_cache_capture(config)
    actual_joint_capture_count = 0
    capture = None if not use_capture_cache else load_capture_cache(cache_root, request.request_key, capture_provenance)
    if use_capture_cache:
        record_cache("capture", capture is not None)
    if capture is None:
        actual_joint_capture_count += 1
        capture = capture_joint_graph(request)
        if use_capture_cache:
            _try_store_capture_cache(cache_root, request.request_key, capture, capture_provenance)
    mark_elapsed("capture_us", capture_start)
    ir_start = time.perf_counter()
    ir, ir_report = build_joint_ir(capture)
    mark_elapsed("ir_build_us", ir_start)
    if not ir_report.valid:
        raise ValueError(f"invalid IR: {ir_report.errors}")

    refined_start = time.perf_counter()
    feasibility = analyze_refined_feasibility(ir, fixed_timeline, memory_budget_bytes)
    mark_elapsed("refined_feasibility_us", refined_start)
    if feasibility.status == "INFEASIBLE_BY_ACTIVATION_ONLY":
        raise InfeasibleBudgetError(feasibility.explanations[0])

    analysis_start = time.perf_counter()
    analysis_key = _analysis_cache_key(ir.graph_key, optimizer_spec.name, memory_budget_bytes, config)
    analysis_provenance = _analysis_cache_provenance(
        request,
        ir.graph_key,
        optimizer_spec.name,
        memory_budget_bytes,
        config,
    )
    cached_analysis = None if cache_root is None else load_analysis_cache(cache_root, analysis_key, analysis_provenance)
    if cache_root is not None:
        record_cache("analysis", cached_analysis is not None)
    if cached_analysis is not None:
        evaluated = _rebind_evaluated_plans(cached_analysis.baseline_results, ir.graph_key)
        early_stop = getattr(cached_analysis, "early_stop", None)
        capture_failures = getattr(cached_analysis, "capture_failures", capture.failures)
        search_diagnostics = getattr(cached_analysis, "search_diagnostics", None)
    else:
        capture_failures = capture.failures
        safety_margin = max(config.safety_margin_bytes, int(memory_budget_bytes * config.safety_margin_ratio))
        evaluated, search_diagnostics = search_plans_with_diagnostics(
            ir,
            fixed_timeline,
            budget_bytes=memory_budget_bytes,
            safety_margin_bytes=safety_margin,
            manual_saved_value_ids=config.manual_saved_value_ids,
            cost_provider=build_composite_provider(
                tuple(record.service for record in registry.services_for(ServiceKind.COST_PROVIDER))
            ),
            enable_diagnostic_hints=config.enable_diagnostic_hints,
            top_k=config.top_k,
        )
        early_stop = apply_early_stop_policy(evaluated, fixed_timeline=fixed_timeline)
        if cache_root is not None:
            store_analysis_cache(
                cache_root,
                analysis_key,
                AnalysisBundle(
                    ir=ir,
                    fixed_timeline=fixed_timeline,
                    baseline_results=evaluated,
                    analysis_key=analysis_key,
                    early_stop=early_stop,
                    capture_failures=capture_failures,
                    search_diagnostics=search_diagnostics,
                ),
                analysis_provenance,
            )
    mark_elapsed("analysis_us", analysis_start)
    if not evaluated:
        raise InfeasibleBudgetError("no plans were generated")
    executor_start = time.perf_counter()
    executor = build_training_step_executor(
        model,
        optimizer,
        loss_fn,
        config,
        capture.guards,
        selection_objective=config.selection_objective,
    )
    mark_elapsed("executor_build_us", executor_start)
    measured_candidates: list[MeasuredExecutable] = []
    dry_runs: dict[str, DryRunResult] = {}
    rejected: dict[str, str] = {}
    validation_start = time.perf_counter()
    for candidate in evaluated:
        payload = _candidate_payload(request, capture, candidate, ir, isolate=config.isolate_candidate_measurement)
        if config.isolate_candidate_measurement:
            worker = run_in_worker_process(
                _validate_and_measure_candidate,
                payload,
                timeout_s=config.candidate_worker_timeout_s,
            )
            if not worker.ok:
                rejected[candidate.plan.plan_id] = f"{worker.error_type}: {worker.message}"
                continue
            validation = worker.value
        else:
            try:
                validation = _validate_and_measure_candidate(payload)
            except Exception as exc:
                rejected[candidate.plan.plan_id] = f"{type(exc).__name__}: {exc}"
                continue
        dry_runs[candidate.plan.plan_id] = validation.dry_run
        if not (
            validation.dry_run.abi_valid
            and validation.dry_run.outputs_match
            and validation.dry_run.gradients_match
        ):
            rejected[candidate.plan.plan_id] = validation.dry_run.failure_reason or "candidate failed dry-run"
            continue
        if cache_root is not None:
            record_cache("executable", validation.cache_hit)
        if validation.measurement is None:
            rejected[candidate.plan.plan_id] = "candidate measurement is unavailable"
            continue
        measured_candidates.append(_measure_candidate_for_parent(executor, validation))
    mark_elapsed("candidate_validation_measurement_us", validation_start)
    measured_tuple = tuple(measured_candidates)
    selected_measured = _select_measured_candidate(
        measured_tuple,
        memory_budget_bytes=memory_budget_bytes,
        selection_objective=config.selection_objective,
    )
    if selected_measured is None:
        details = "; ".join(f"{plan_id}: {reason}" for plan_id, reason in sorted(rejected.items()))
        raise InfeasibleBudgetError(f"no Top-K candidate passed dry-run and measurement: {details}")
    selected = next(plan for plan in evaluated if plan.plan.plan_id == selected_measured.plan_id)
    dry_run = dry_runs[selected.plan.plan_id]
    measured = selected_measured
    fallback_ids = tuple(item.plan_id for item in measured_tuple if item.plan_id != selected.plan.plan_id)
    executor.current_plan_id = selected.plan.plan_id
    executor.selection_objective = config.selection_objective
    executor.fallback_executables = tuple(
        (item.plan_id, item.forward_backward)
        for item in measured_tuple
        if item.plan_id != selected.plan.plan_id and item.correctness_passed
    )
    executor.runtime_peak_threshold_bytes = min(
        memory_budget_bytes,
        measured.measured_peak_bytes + config.runtime_peak_safety_margin_bytes,
    )
    analysis = AnalysisBundle(
        ir=ir,
        fixed_timeline=fixed_timeline,
        baseline_results=evaluated,
        analysis_key=analysis_key,
        early_stop=early_stop,
        capture_failures=capture_failures,
        search_diagnostics=search_diagnostics,
    )
    optimization_metrics["candidate_count"] = len(evaluated)
    optimization_metrics["measured_candidate_count"] = len(measured_tuple)
    optimization_metrics["rejected_candidate_count"] = len(rejected)
    optimization_metrics["actual_joint_capture_count"] = actual_joint_capture_count
    optimization_metrics["total_optimization_us"] = (time.perf_counter() - optimization_start) * 1_000_000.0
    return OptimizedTrainingResult(
        selected_plan=selected.plan,
        executable=measured,
        executor=executor,
        feasibility=feasibility,
        fallback_plan_ids=fallback_ids,
        analysis=analysis,
        dry_run=dry_run,
        measured_candidates=measured_tuple,
        cache_stats=CacheStats(layer_hits=cache_hits, layer_misses=cache_misses),
        optimization_metrics=optimization_metrics,
    )
