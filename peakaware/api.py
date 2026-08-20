from __future__ import annotations

import hashlib
import gc
import random
import time
from contextlib import nullcontext
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
    SearchDiagnostics,
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
from peakaware.cost.base import provider_cache_safe
from peakaware.errors import InfeasibleBudgetError
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import (
    analyze_coarse_feasibility,
    analyze_refined_feasibility,
    build_optimizer_spec,
)
from peakaware.memory.fx_timeline import (
    lowered_fx_l2_structure_summary,
    simulate_lowered_fx_l2_event_trace,
    summarize_lowered_fx_l2_event_trace,
)
from peakaware.memory.simulator import SimulationCostCache, build_simulation_cost_cache
from peakaware.partition.aot import lower_partition_graphs, partition_default_graph
from peakaware.partition.outer_aot import capture_outer_aot_partition
from peakaware.partition.verifier import run_aot_eager_dry_run
from peakaware.plugins import ServiceKind, build_default_registry
from peakaware.runtime.executor import (
    build_aot_partition_executable,
    build_training_step_executor,
    make_measured_executable,
)
from peakaware.runtime.isolation import run_in_worker_process
from peakaware.search.engine import apply_early_stop_policy, evaluate_plan, search_plans_with_diagnostics
from peakaware.search.beam import solve_peak_aware_beam


CAPTURE_SCHEMA_VERSION = "capture-v2-guarded-graph-key"
ANALYSIS_SCHEMA_VERSION = "analysis-v8-observer-free-state"


def _hardware_spec(args: tuple[Any, ...], kwargs: dict[str, Any]) -> HardwareSpec:
    devices = [value.device for value in _iter_tensors(args) + _iter_tensors(kwargs)]
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
    for value in _iter_tensors(args) + _iter_tensors({key: kwargs[key] for key in sorted(kwargs)}):
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
        search_policy_version=(
            f"peakaware-{config.search_algorithm}-topk{config.top_k}"
            f"-greedy{config.max_greedy_candidates}-beam{config.beam_width}"
            f"-beammax{config.max_beam_candidates}"
            f"-overflow{config.beam_candidate_overflow_policy}"
            f"-fxl2top{config.compiler_refinement_top_k}"
            f"-{config.search_algorithm}-v1"
            f"-dh{int(config.enable_diagnostic_hints)}"
        ),
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
        "max_greedy_candidates": config.max_greedy_candidates,
        "search_algorithm": config.search_algorithm,
        "beam_width": config.beam_width,
        "max_beam_candidates": config.max_beam_candidates,
        "beam_candidate_overflow_policy": config.beam_candidate_overflow_policy,
        "compiler_refinement_top_k": config.compiler_refinement_top_k,
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
    *,
    runtime_adapter: str = "eager_model",
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
        compiler_version=f"{compiler_version}:{runtime_adapter}",
        partition_plugin_version="core",
    )


def _executable_cache_provenance(
    request: TrainingRequest,
    candidate: EvaluatedPlan,
    *,
    runtime_adapter: str = "eager_model",
) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "request_key": request.request_key,
        "plan_id": candidate.plan.plan_id,
        "graph_key": candidate.plan.graph_key,
        "correctness_required": True,
        "runtime_adapter": runtime_adapter,
    }


def _iter_tensors(value: Any) -> tuple[Tensor, ...]:
    found: list[Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, Tensor):
            found.append(item)
        elif isinstance(item, dict):
            for key in sorted(item):
                collect(item[key])
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
    num_fwd_outputs: int = 1,
    kwarg_names: tuple[str, ...] | None = None,
    output_tree_spec: Any | None = None,
    output_tangent_mask: tuple[bool, ...] = (),
    arg_tree_specs: tuple[Any, ...] = (),
    kwarg_tree_specs: tuple[tuple[str, Any], ...] = (),
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
        num_fwd_outputs=num_fwd_outputs,
        kwarg_names=kwarg_names,
        output_tree_spec=output_tree_spec,
        output_tangent_mask=output_tangent_mask,
        arg_tree_specs=arg_tree_specs,
        kwarg_tree_specs=kwarg_tree_specs,
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


def _measured_candidate_rejection_reason(
    measured: MeasuredExecutable,
    *,
    memory_budget_bytes: int,
) -> str | None:
    if measured.correctness_passed is not True:
        return f"correctness_passed={measured.correctness_passed}"
    if measured.measured_peak_bytes > memory_budget_bytes:
        return (
            f"measured peak {measured.measured_peak_bytes} bytes exceeds "
            f"budget {memory_budget_bytes} bytes by "
            f"{measured.measured_peak_bytes - memory_budget_bytes} bytes"
        )
    return None


def _select_validation_candidates(
    evaluated: tuple[EvaluatedPlan, ...],
    *,
    validation_top_k: int | None,
    selection_objective: str,
    selection_policy: str = "ranked",
) -> tuple[EvaluatedPlan, ...]:
    if validation_top_k is None or validation_top_k >= len(evaluated):
        return evaluated

    def key(candidate: EvaluatedPlan) -> tuple[Any, ...]:
        if selection_objective == "min_time_then_peak":
            objective = (
                candidate.simulation.estimated_step_us,
                candidate.simulation.estimated_peak_bytes,
            )
        else:
            objective = (
                candidate.simulation.estimated_peak_bytes,
                candidate.simulation.estimated_step_us,
            )
        return (
            not candidate.feasible,
            *objective,
            candidate.simulation.risk_score,
            -candidate.simulation.confidence,
            candidate.plan.plan_id,
        )

    ranked = tuple(sorted(evaluated, key=key))
    if selection_policy == "ranked":
        return ranked[:validation_top_k]
    if selection_policy != "structural_diverse":
        raise ValueError(f"unsupported validation selection policy: {selection_policy}")

    feasible = tuple(candidate for candidate in ranked if candidate.feasible)
    eligible = feasible if len(feasible) >= validation_top_k else ranked
    rank_index = {id(candidate): index for index, candidate in enumerate(ranked)}
    selected: list[EvaluatedPlan] = [eligible[0]]

    for preferred_plan_id in ("torch_min_cut", "block_checkpoint"):
        preferred = next(
            (
                candidate
                for candidate in eligible
                if candidate.plan.plan_id == preferred_plan_id and candidate not in selected
            ),
            None,
        )
        if preferred is not None and len(selected) < validation_top_k:
            selected.append(preferred)

    def saved_set_distance(left: EvaluatedPlan, right: EvaluatedPlan) -> float:
        left_ids = left.plan.saved_value_ids
        right_ids = right.plan.saved_value_ids
        union = left_ids | right_ids
        if not union:
            return 0.0
        return 1.0 - len(left_ids & right_ids) / len(union)

    while len(selected) < validation_top_k:
        remaining = [candidate for candidate in eligible if candidate not in selected]
        if not remaining:
            remaining = [candidate for candidate in ranked if candidate not in selected]
        if not remaining:
            break
        chosen = max(
            remaining,
            key=lambda candidate: (
                min(saved_set_distance(candidate, item) for item in selected),
                -rank_index[id(candidate)],
            ),
        )
        selected.append(chosen)

    selected_ids = {id(candidate) for candidate in selected}
    return tuple(candidate for candidate in ranked if id(candidate) in selected_ids)


def _order_validation_candidates(
    candidates: tuple[EvaluatedPlan, ...],
    *,
    seed: int | None,
) -> tuple[EvaluatedPlan, ...]:
    if seed is None or len(candidates) < 2:
        return candidates
    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _reset_candidate_measurement_state(
    config: PeakAwareConfig,
    model: nn.Module,
) -> None:
    if not config.reset_compiler_before_candidate_measurement:
        return
    if config.enable_compile:
        torch.compiler.reset()
    gc.collect()
    parameter = next(model.parameters(), None)
    device = None if parameter is None else parameter.device
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)


def _release_dry_run_temporaries(model: nn.Module) -> None:
    """Release correctness-replay tensors before measuring the candidate.

    AOT dry-run verification materializes baseline and lowered gradients.  The
    returned ``DryRunResult`` contains no tensors, so retaining those temporary
    autograd objects into the measurement window would measure verifier state
    rather than the executable's steady training footprint.
    """

    gc.collect()
    parameter = next(model.parameters(), None)
    device = None if parameter is None else parameter.device
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _candidate_attempt_row(
    candidate: EvaluatedPlan,
    *,
    elapsed_us: float,
    status: str,
    measurement: MeasuredExecutable | None = None,
    error_type: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    error_text = " ".join(value for value in (error_type, failure_reason) if value).lower()
    actual_oom = (
        "outofmemory" in error_text
        or "out of memory" in error_text
        or "cuda oom" in error_text
        or "cuda_oom" in error_text
    )
    phase_metrics = {} if measurement is None else measurement.phase_metrics
    return {
        "plan_id": candidate.plan.plan_id,
        "estimated_peak_bytes": candidate.simulation.estimated_peak_bytes,
        "estimated_step_us": candidate.simulation.estimated_step_us,
        "estimated_feasible": candidate.feasible,
        "risk_score": candidate.simulation.risk_score,
        "confidence": candidate.simulation.confidence,
        "validation_elapsed_us": elapsed_us,
        "status": status,
        "error_type": error_type,
        "failure_reason": failure_reason,
        "actual_oom": actual_oom,
        "measured_peak_bytes": None if measurement is None else measurement.measured_peak_bytes,
        "measured_step_us": None if measurement is None else measurement.measured_step_us,
        "correctness_passed": None if measurement is None else measurement.correctness_passed,
        "measured_fw_peak_bytes": phase_metrics.get("fw_peak_bytes"),
        "measured_bw_peak_bytes": phase_metrics.get("bw_peak_bytes"),
        "measured_optimizer_peak_bytes": phase_metrics.get("optimizer_peak_bytes"),
        "measured_peak_phase": phase_metrics.get("peak_phase"),
    }


@dataclass(frozen=True)
class _CandidateMeasurement:
    plan_id: str
    measured_peak_bytes: int
    measured_step_us: float
    phase_metrics: dict[str, int | float]
    activation_checkpoint: bool = False
    aot_partition_runtime: bool = False


@dataclass(frozen=True)
class _CandidateValidation:
    dry_run: DryRunResult
    measurement: _CandidateMeasurement | None
    cache_hit: bool = False
    lowered: LoweredPartition | None = None
    kwarg_names: tuple[str, ...] = ()
    num_fwd_outputs: int = 1
    output_tree_spec: Any | None = None
    output_tangent_mask: tuple[bool, ...] = ()
    arg_tree_specs: tuple[Any, ...] = ()
    kwarg_tree_specs: tuple[tuple[str, Any], ...] = ()


def _lowered_fx_l2_phase_metrics(
    lowered: LoweredPartition | None,
    fixed_timeline: Any,
    config: PeakAwareConfig,
) -> dict[str, Any]:
    if lowered is None:
        return {}
    try:
        registry = build_default_registry(profile_db_path=config.profile_db_path)
        cost_provider = build_composite_provider(
            tuple(record.service for record in registry.services_for(ServiceKind.COST_PROVIDER))
        )
        trace = simulate_lowered_fx_l2_event_trace(
            lowered,
            fixed_timeline,
            cost_provider=cost_provider,
        )
    except Exception as exc:
        return {
            "lowered_fx_l2_simulated_memory_event_trace_error_type": type(exc).__name__,
            "lowered_fx_l2_simulated_memory_event_trace_error_message": str(exc),
        }
    return {
        "lowered_fx_l2_simulated_memory_event_trace_kind": "lowered_fx_l2_liveness_costmodel_time",
        "lowered_fx_l2_simulated_memory_event_trace": trace,
    }


def _refine_candidate_with_lowered_fx_l2(
    candidate: EvaluatedPlan,
    capture: CapturedJointGraph,
    request: TrainingRequest,
    ir: JointTrainingIR,
    fixed_timeline: Any,
    cost_provider: Any,
) -> tuple[EvaluatedPlan, LoweredPartition]:
    """Refine one coarse candidate without executing it on the target GPU."""

    lowered = _lower_candidate(capture, candidate, ir)
    refined_lowered = lowered
    refinement_source = "lowered_fx_l2_liveness"
    outer_capture_failure: dict[str, str] | None = None
    if request.config.enable_compile and capture.backend == "aot":
        try:
            base_executable = build_aot_partition_executable(
                lowered,
                request.model,
                num_fwd_outputs=capture.num_fwd_outputs,
                kwarg_names=tuple(request.example_kwargs),
                output_tree_spec=capture.output_tree_spec,
                output_tangent_mask=capture.output_tangent_mask,
                arg_tree_specs=capture.arg_tree_specs,
                kwarg_tree_specs=capture.kwarg_tree_specs,
            )
            refined_lowered = capture_outer_aot_partition(
                base_executable,
                request.example_args,
                request.example_kwargs,
                plan_id=candidate.plan.plan_id,
            )
            refinement_source = "outer_aot_fx_l3_liveness"
        except Exception as exc:
            outer_capture_failure = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    trace = simulate_lowered_fx_l2_event_trace(
        refined_lowered,
        fixed_timeline,
        cost_provider=cost_provider,
        loss_fn=request.loss_fn,
        num_fwd_outputs=capture.num_fwd_outputs,
    )
    summary = summarize_lowered_fx_l2_event_trace(trace)
    peak_row = summary["peak_row"]
    phase_peaks = summary["phase_peak_bytes"]
    peak_phase = str(summary["peak_phase"])
    snapshot_phase = "fw" if peak_phase == "after_fw" else peak_phase
    refined_peak = int(summary["estimated_peak_bytes"])
    optimizer_step_us = float(
        candidate.simulation.cost_breakdown.get("optimizer_step_us", 0.0)
    )
    refined_step_us = max(
        float(summary["trace_end_time_us"]) + optimizer_step_us,
        0.0,
    )
    after_fw_payload = int(summary["after_fw_payload_bytes"])
    peak_payload = int(peak_row.get("payload_bytes", 0))
    simulation = replace(
        candidate.simulation,
        estimated_peak_bytes=refined_peak,
        estimated_step_us=refined_step_us,
        peak_snapshot=replace(
            candidate.simulation.peak_snapshot,
            phase=snapshot_phase,
            live_storage_ids=frozenset(),
            live_bytes=peak_payload,
            parameter_bytes=int(peak_row.get("parameter_bytes", fixed_timeline.parameter_bytes)),
            gradient_bytes=int(peak_row.get("gradient_bytes", 0)),
            optimizer_bytes=int(peak_row.get("optimizer_bytes", 0)),
            saved_activation_bytes=min(after_fw_payload, peak_payload),
            recomputed_bytes=0,
            workspace_bytes=int(peak_row.get("workspace_bytes", 0)),
            runtime_replica_bytes=int(peak_row.get("runtime_replica_bytes", 0)),
        ),
        after_fw_retained_bytes=int(summary["after_fw_retained_bytes"]),
        fw_peak_bytes=int(phase_peaks["fw"]),
        bw_peak_bytes=int(phase_peaks["bw"]),
        optimizer_peak_bytes=int(phase_peaks["optimizer"]),
        cost_breakdown={
            **dict(candidate.simulation.cost_breakdown),
            "compiler_refinement": {
                "source": refinement_source,
                "candidate_gpu_measurements_used": 0,
                "coarse_estimated_peak_bytes": candidate.simulation.estimated_peak_bytes,
                "coarse_estimated_step_us": candidate.simulation.estimated_step_us,
                "refined_estimated_peak_bytes": refined_peak,
                "refined_estimated_step_us": refined_step_us,
                "phase_peak_bytes": dict(phase_peaks),
                "trace_end_time_us": float(summary["trace_end_time_us"]),
                "lowered_structure": lowered_fx_l2_structure_summary(
                    refined_lowered
                ),
                "outer_aot_capture_failure": outer_capture_failure,
            },
        },
        simulated_memory_event_trace=trace,
    )
    closure_valid = candidate.rejection_reason in {
        None,
        "estimated peak exceeds search budget",
    }
    feasible = closure_valid and refined_peak <= (
        candidate.plan.budget_bytes - candidate.plan.safety_margin_bytes
    )
    rejection_reason = candidate.rejection_reason
    if closure_valid:
        rejection_reason = None if feasible else "compiler-refined peak exceeds search budget"
    plan = replace(
        candidate.plan,
        estimated_peak_bytes=refined_peak,
        estimated_step_us=refined_step_us,
        cost_sources=tuple(
            dict.fromkeys((*candidate.plan.cost_sources, refinement_source))
        ),
    )
    return (
        EvaluatedPlan(
            plan=plan,
            simulation=simulation,
            feasible=feasible,
            rejection_reason=rejection_reason,
        ),
        lowered,
    )


def _compiler_refine_candidates(
    evaluated: tuple[EvaluatedPlan, ...],
    capture: CapturedJointGraph,
    request: TrainingRequest,
    ir: JointTrainingIR,
    fixed_timeline: Any,
    cost_provider: Any,
    config: PeakAwareConfig,
) -> tuple[
    tuple[EvaluatedPlan, ...],
    tuple[EvaluatedPlan, ...],
    dict[str, LoweredPartition],
    tuple[dict[str, str], ...],
]:
    top_k = min(config.compiler_refinement_top_k, len(evaluated))
    if top_k <= 0:
        return evaluated, evaluated, {}, ()
    coarse_pool = _select_validation_candidates(
        evaluated,
        validation_top_k=top_k,
        selection_objective=config.selection_objective,
        selection_policy="ranked",
    )
    refined_by_plan_id: dict[str, EvaluatedPlan] = {}
    lowered_by_plan_id: dict[str, LoweredPartition] = {}
    failures: list[dict[str, str]] = []
    for candidate in coarse_pool:
        try:
            refined, lowered = _refine_candidate_with_lowered_fx_l2(
                candidate,
                capture,
                request,
                ir,
                fixed_timeline,
                cost_provider,
            )
        except Exception as exc:
            failures.append(
                {
                    "plan_id": candidate.plan.plan_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        refined_by_plan_id[candidate.plan.plan_id] = refined
        lowered_by_plan_id[candidate.plan.plan_id] = lowered
    if not refined_by_plan_id:
        return evaluated, evaluated, {}, tuple(failures)
    updated = tuple(
        refined_by_plan_id.get(item.plan.plan_id, item) for item in evaluated
    )
    refined_pool = tuple(
        refined_by_plan_id.get(item.plan.plan_id, item) for item in coarse_pool
    )
    return updated, refined_pool, lowered_by_plan_id, tuple(failures)


def _candidate_uses_activation_checkpoint(candidate: EvaluatedPlan, ir: JointTrainingIR) -> bool:
    residual_value_ids = frozenset(
        value.id for value in ir.values if value.phase == "fw" and value.crosses_fw_bw
    )
    effective_saved = candidate.plan.saved_value_ids | candidate.plan.mandatory_value_ids
    return not residual_value_ids.issubset(effective_saved)


def _aot_candidate_cudnn_context(capture: CapturedJointGraph, model: nn.Module) -> Any:
    if capture.backend != "aot":
        return nullcontext()
    has_training_batchnorm = any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        for module in model.modules()
    )
    if not has_training_batchnorm:
        return nullcontext()
    return torch.backends.cudnn.flags(enabled=False)


def _validate_and_measure_candidate(payload: dict[str, Any]) -> _CandidateValidation:
    capture = payload["capture"]
    candidate = payload["candidate"]
    ir = payload["ir"]
    request = payload["request"]
    model = payload["model"]
    optimizer = payload["optimizer"]
    loss_fn = payload["loss_fn"]
    config = payload["config"]
    fixed_timeline = payload["fixed_timeline"]
    example_args = payload["example_args"]
    example_kwargs = payload["example_kwargs"]
    kwarg_names = tuple(example_kwargs)

    if capture is None or ir is None:
        capture = capture_joint_graph(request)
        ir, ir_report = build_joint_ir(capture)
        if not ir_report.valid:
            raise ValueError(f"invalid worker IR: {ir_report.errors}")
    can_return_lowered = capture.backend == "aot"
    with _aot_candidate_cudnn_context(capture, model):
        lowered = _lower_candidate(capture, candidate, ir)
        dry_run = _dry_run_candidate(
            lowered,
            ir,
            model=model,
            example_args=example_args,
            example_kwargs=example_kwargs,
            loss_fn=loss_fn,
            config=config,
            num_fwd_outputs=capture.num_fwd_outputs,
            kwarg_names=kwarg_names,
            output_tree_spec=capture.output_tree_spec,
            output_tangent_mask=capture.output_tangent_mask,
            arg_tree_specs=capture.arg_tree_specs,
            kwarg_tree_specs=capture.kwarg_tree_specs,
        )
    if not (dry_run.abi_valid and dry_run.outputs_match and dry_run.gradients_match):
        return _CandidateValidation(dry_run=dry_run, measurement=None)
    activation_checkpoint = _candidate_uses_activation_checkpoint(candidate, ir)
    executable_override = None
    aot_partition_runtime = False
    if can_return_lowered and dry_run.replay_mode == "lowered_aot":
        try:
            executable_override = build_aot_partition_executable(
                lowered,
                model,
                num_fwd_outputs=capture.num_fwd_outputs,
                kwarg_names=kwarg_names,
                output_tree_spec=capture.output_tree_spec,
                output_tangent_mask=capture.output_tangent_mask,
                arg_tree_specs=capture.arg_tree_specs,
                kwarg_tree_specs=capture.kwarg_tree_specs,
            )
            activation_checkpoint = False
            aot_partition_runtime = True
        except Exception:
            executable_override = None
            aot_partition_runtime = False
    executor = build_training_step_executor(
        model,
        optimizer,
        loss_fn,
        config,
        capture.guards,
        selection_objective=config.selection_objective,
        activation_checkpoint=activation_checkpoint,
        executable_override=executable_override,
        aot_partition_runtime=aot_partition_runtime,
    )
    cache_root = _cache_root(config)
    if aot_partition_runtime:
        runtime_adapter = "lowered_aot_partition"
    elif activation_checkpoint:
        runtime_adapter = "activation_checkpoint"
    else:
        runtime_adapter = "eager_model"
    _release_dry_run_temporaries(model)
    executable_key = _executable_cache_key(request, capture, candidate, runtime_adapter=runtime_adapter)
    executable_provenance = _executable_cache_provenance(request, candidate, runtime_adapter=runtime_adapter)
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
                    phase_metrics={
                        **dict(cached.phase_metrics),
                        **_lowered_fx_l2_phase_metrics(
                            lowered if aot_partition_runtime else None,
                            fixed_timeline,
                            config,
                        ),
                    },
                    activation_checkpoint=activation_checkpoint,
                    aot_partition_runtime=bool(cached.phase_metrics.get("aot_partition_runtime", 0)),
                ),
                cache_hit=True,
                lowered=lowered if aot_partition_runtime else None,
                kwarg_names=kwarg_names if aot_partition_runtime else (),
                num_fwd_outputs=capture.num_fwd_outputs,
                output_tree_spec=capture.output_tree_spec if aot_partition_runtime else None,
                output_tangent_mask=capture.output_tangent_mask if aot_partition_runtime else (),
                arg_tree_specs=capture.arg_tree_specs if aot_partition_runtime else (),
                kwarg_tree_specs=capture.kwarg_tree_specs if aot_partition_runtime else (),
            )
    with _aot_candidate_cudnn_context(capture, model):
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
            phase_metrics={
                **dict(measured.phase_metrics),
                **_lowered_fx_l2_phase_metrics(
                    lowered if aot_partition_runtime else None,
                    fixed_timeline,
                    config,
                ),
            },
            activation_checkpoint=activation_checkpoint,
            aot_partition_runtime=aot_partition_runtime,
        ),
        lowered=lowered if aot_partition_runtime else None,
        kwarg_names=kwarg_names if aot_partition_runtime else (),
        num_fwd_outputs=capture.num_fwd_outputs,
        output_tree_spec=capture.output_tree_spec if aot_partition_runtime else None,
        output_tangent_mask=capture.output_tangent_mask if aot_partition_runtime else (),
        arg_tree_specs=capture.arg_tree_specs if aot_partition_runtime else (),
        kwarg_tree_specs=capture.kwarg_tree_specs if aot_partition_runtime else (),
    )


def _candidate_payload(
    request: TrainingRequest,
    capture: CapturedJointGraph,
    candidate: EvaluatedPlan,
    ir: JointTrainingIR,
    fixed_timeline: Any,
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
        "fixed_timeline": fixed_timeline,
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
    executable_override = None
    if measurement.aot_partition_runtime:
        if validation.lowered is None:
            raise ValueError("AOT partition runtime measurement is missing lowered graphs")
        executable_override = build_aot_partition_executable(
            validation.lowered,
            executor.model,
            num_fwd_outputs=validation.num_fwd_outputs,
            kwarg_names=validation.kwarg_names,
            output_tree_spec=validation.output_tree_spec,
            output_tangent_mask=validation.output_tangent_mask,
            arg_tree_specs=validation.arg_tree_specs,
            kwarg_tree_specs=validation.kwarg_tree_specs,
        )
    candidate_executor = build_training_step_executor(
        executor.model,
        executor.optimizer,
        executor.loss_fn,
        executor.config,
        executor.guards,
        selection_objective=executor.selection_objective,
        activation_checkpoint=measurement.activation_checkpoint,
        executable_override=executable_override,
        aot_partition_runtime=measurement.aot_partition_runtime,
    )
    return MeasuredExecutable(
        plan_id=measurement.plan_id,
        forward_backward=candidate_executor.executable,
        measured_peak_bytes=measurement.measured_peak_bytes,
        measured_step_us=measurement.measured_step_us,
        correctness_passed=True,
        phase_metrics=measurement.phase_metrics,
    )


def _realize_candidate_from_simulation(
    executor: Any,
    capture: CapturedJointGraph,
    candidate: EvaluatedPlan,
    ir: JointTrainingIR,
    kwarg_names: tuple[str, ...],
    fixed_timeline: Any,
    config: PeakAwareConfig,
    lowered: LoweredPartition | None = None,
) -> MeasuredExecutable:
    """Build the simulator-selected executable without executing or benchmarking it."""

    activation_checkpoint = _candidate_uses_activation_checkpoint(candidate, ir)
    executable_override = None
    aot_partition_runtime = False
    if capture.backend == "aot":
        lowered = lowered or _lower_candidate(capture, candidate, ir)
        executable_override = build_aot_partition_executable(
            lowered,
            executor.model,
            num_fwd_outputs=capture.num_fwd_outputs,
            kwarg_names=kwarg_names,
            output_tree_spec=capture.output_tree_spec,
            output_tangent_mask=capture.output_tangent_mask,
            arg_tree_specs=capture.arg_tree_specs,
            kwarg_tree_specs=capture.kwarg_tree_specs,
        )
        activation_checkpoint = False
        aot_partition_runtime = True
    candidate_executor = build_training_step_executor(
        executor.model,
        executor.optimizer,
        executor.loss_fn,
        executor.config,
        executor.guards,
        selection_objective=executor.selection_objective,
        activation_checkpoint=activation_checkpoint,
        executable_override=executable_override,
        aot_partition_runtime=aot_partition_runtime,
    )
    phase_metrics = {
        "measurement_source": "simulation_only",
        "measurement_warmup_steps": 0,
        "measurement_repeats": 0,
        "overall_peak_bytes": candidate.simulation.estimated_peak_bytes,
        "step_us": candidate.simulation.estimated_step_us,
        "activation_checkpoint": int(activation_checkpoint),
        "aot_partition_runtime": int(aot_partition_runtime),
        "candidate_gpu_measurements_used": 0,
        "compiler_refinement_source": (
            str(
                candidate.simulation.cost_breakdown["compiler_refinement"][
                    "source"
                ]
            )
            if "compiler_refinement" in candidate.simulation.cost_breakdown
            else "none"
        ),
    }
    if lowered is not None:
        phase_metrics.update(
            _lowered_fx_l2_phase_metrics(lowered, fixed_timeline, config)
        )
    return MeasuredExecutable(
        plan_id=candidate.plan.plan_id,
        forward_backward=candidate_executor.executable,
        measured_peak_bytes=candidate.simulation.estimated_peak_bytes,
        measured_step_us=candidate.simulation.estimated_step_us,
        correctness_passed=None,
        phase_metrics=phase_metrics,
        evidence_source="simulated",
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
    optimization_metrics: dict[str, Any] = {}

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
    cost_provider = build_composite_provider(
        tuple(record.service for record in registry.services_for(ServiceKind.COST_PROVIDER))
    )
    simulation_cost_cache: SimulationCostCache | None = None
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
        if config.search_algorithm in {
            "pareto_beam",
            "lagrangian_beam",
            "lagrangian_sweep_beam",
        }:
            if provider_cache_safe(cost_provider):
                simulation_cost_cache = build_simulation_cost_cache(
                    ir,
                    fixed_timeline,
                    cost_provider,
                )
            beam_result = solve_peak_aware_beam(
                ir,
                fixed_timeline,
                budget_bytes=memory_budget_bytes,
                safety_margin_bytes=safety_margin,
                beam_width=config.beam_width,
                max_candidate_count=config.max_beam_candidates,
                candidate_overflow_policy=config.beam_candidate_overflow_policy,
                cost_provider=cost_provider,
                selection_objective=config.selection_objective,
                pruning_strategy={
                    "pareto_beam": "pareto_lexicographic",
                    "lagrangian_beam": "lagrangian",
                    "lagrangian_sweep_beam": "lagrangian_sweep",
                }[config.search_algorithm],
                event_trace_materialization="none",
                simulation_cost_cache=simulation_cost_cache,
            )
            evaluated = beam_result.evaluated
            feasible_count = sum(item.feasible for item in evaluated)
            search_diagnostics = SearchDiagnostics(
                diagnostic_hints_enabled=False,
                manual_hint_count=0,
                diagnostic_hint_count=0,
                diagnostic_hint_kinds=(),
                diagnostic_hint_candidate_match_count=0,
                diagnostic_hint_order_changed=False,
                diagnostic_hint_order_delta_count=0,
                greedy_plan_count=beam_result.evaluated_plan_count,
                feasible_before_repair_count=feasible_count,
                repaired_candidate_count=0,
                repair_success_count=0,
                feasible_after_repair_count=feasible_count,
                repaired_plan_ids=(),
            )
        else:
            evaluated, search_diagnostics = search_plans_with_diagnostics(
                ir,
                fixed_timeline,
                budget_bytes=memory_budget_bytes,
                safety_margin_bytes=safety_margin,
                manual_saved_value_ids=config.manual_saved_value_ids,
                cost_provider=cost_provider,
                enable_diagnostic_hints=config.enable_diagnostic_hints,
                top_k=config.top_k,
                max_greedy_candidates=config.max_greedy_candidates,
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
    refinement_start = time.perf_counter()
    (
        evaluated,
        candidate_selection_pool,
        refined_lowered_by_plan_id,
        compiler_refinement_failures,
    ) = _compiler_refine_candidates(
        evaluated,
        capture,
        request,
        ir,
        fixed_timeline,
        cost_provider,
        config,
    )
    mark_elapsed("compiler_refinement_us", refinement_start)
    refinement_sources = tuple(
        dict.fromkeys(
            str(item.simulation.cost_breakdown["compiler_refinement"]["source"])
            for item in candidate_selection_pool
            if "compiler_refinement" in item.simulation.cost_breakdown
        )
    )
    optimization_metrics["compiler_refinement_source"] = (
        "none"
        if not refinement_sources
        else refinement_sources[0]
        if len(refinement_sources) == 1
        else "mixed:" + ",".join(refinement_sources)
    )
    optimization_metrics["compiler_refinement_requested_count"] = min(
        config.compiler_refinement_top_k,
        len(evaluated),
    )
    optimization_metrics["compiler_refinement_success_count"] = len(
        refined_lowered_by_plan_id
    )
    optimization_metrics["compiler_refinement_failure_count"] = len(
        compiler_refinement_failures
    )
    optimization_metrics["compiler_refinement_failures"] = compiler_refinement_failures
    optimization_metrics["compiler_refinement_candidate_gpu_measurements_used"] = 0
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
    simulation_only = config.validation_top_k == 0
    candidate_attempts: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    if simulation_only:
        selected = _select_validation_candidates(
            candidate_selection_pool,
            validation_top_k=1,
            selection_objective=config.selection_objective,
            selection_policy="ranked",
        )[0]
        realization_start = time.perf_counter()
        measured = _realize_candidate_from_simulation(
            executor,
            capture,
            selected,
            ir,
            tuple(example_kwargs),
            fixed_timeline,
            config,
            refined_lowered_by_plan_id.get(selected.plan.plan_id),
        )
        mark_elapsed("candidate_realization_us", realization_start)
        optimization_metrics["candidate_validation_measurement_us"] = 0.0
        validation_candidates: tuple[EvaluatedPlan, ...] = ()
        measured_tuple: tuple[MeasuredExecutable, ...] = ()
        dry_run = None
    else:
        validation_candidates = _select_validation_candidates(
            candidate_selection_pool,
            validation_top_k=config.validation_top_k,
            selection_objective=config.selection_objective,
            selection_policy=config.validation_selection_policy,
        )
        validation_candidates = _order_validation_candidates(
            validation_candidates,
            seed=config.candidate_measurement_order_seed,
        )
        measured_candidates: list[MeasuredExecutable] = []
        dry_runs: dict[str, DryRunResult] = {}
        validation_start = time.perf_counter()
        for candidate in validation_candidates:
            _reset_candidate_measurement_state(config, model)
            candidate_start = time.perf_counter()
            payload = _candidate_payload(
                request,
                capture,
                candidate,
                ir,
                fixed_timeline,
                isolate=config.isolate_candidate_measurement,
            )
            if config.isolate_candidate_measurement:
                worker = run_in_worker_process(
                    _validate_and_measure_candidate,
                    payload,
                    timeout_s=config.candidate_worker_timeout_s,
                )
                if not worker.ok:
                    reason = f"{worker.error_type}: {worker.message}"
                    rejected[candidate.plan.plan_id] = reason
                    candidate_attempts.append(
                        _candidate_attempt_row(
                            candidate,
                            elapsed_us=(time.perf_counter() - candidate_start) * 1_000_000.0,
                            status="failed",
                            error_type=worker.error_type,
                            failure_reason=worker.message,
                        )
                    )
                    continue
                validation = worker.value
            else:
                try:
                    validation = _validate_and_measure_candidate(payload)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    rejected[candidate.plan.plan_id] = reason
                    candidate_attempts.append(
                        _candidate_attempt_row(
                            candidate,
                            elapsed_us=(time.perf_counter() - candidate_start) * 1_000_000.0,
                            status="failed",
                            error_type=type(exc).__name__,
                            failure_reason=str(exc),
                        )
                    )
                    continue
            dry_runs[candidate.plan.plan_id] = validation.dry_run
            if not (
                validation.dry_run.abi_valid
                and validation.dry_run.outputs_match
                and validation.dry_run.gradients_match
            ):
                reason = validation.dry_run.failure_reason or "candidate failed dry-run"
                rejected[candidate.plan.plan_id] = reason
                candidate_attempts.append(
                    _candidate_attempt_row(
                        candidate,
                        elapsed_us=(time.perf_counter() - candidate_start) * 1_000_000.0,
                        status="dry_run_failed",
                        error_type="DryRunFailure",
                        failure_reason=reason,
                    )
                )
                continue
            if cache_root is not None:
                record_cache("executable", validation.cache_hit)
            if validation.measurement is None:
                reason = "candidate measurement is unavailable"
                rejected[candidate.plan.plan_id] = reason
                candidate_attempts.append(
                    _candidate_attempt_row(
                        candidate,
                        elapsed_us=(time.perf_counter() - candidate_start) * 1_000_000.0,
                        status="measurement_unavailable",
                        error_type="MeasurementUnavailable",
                        failure_reason=reason,
                    )
                )
                continue
            measured_candidate = _measure_candidate_for_parent(executor, validation)
            elapsed_us = (time.perf_counter() - candidate_start) * 1_000_000.0
            measured_candidate = replace(
                measured_candidate,
                phase_metrics={
                    **dict(measured_candidate.phase_metrics),
                    "candidate_validation_elapsed_us": elapsed_us,
                    "candidate_validation_cache_hit": int(validation.cache_hit),
                },
            )
            measured_candidates.append(measured_candidate)
            candidate_attempts.append(
                _candidate_attempt_row(
                    candidate,
                    elapsed_us=elapsed_us,
                    status="measured",
                    measurement=measured_candidate,
                )
            )
        mark_elapsed("candidate_validation_measurement_us", validation_start)
        measured_tuple = tuple(measured_candidates)
        selected_measured = _select_measured_candidate(
            measured_tuple,
            memory_budget_bytes=memory_budget_bytes,
            selection_objective=config.selection_objective,
        )
        if selected_measured is None:
            for item in measured_tuple:
                reason = _measured_candidate_rejection_reason(
                    item,
                    memory_budget_bytes=memory_budget_bytes,
                )
                if reason is not None:
                    rejected.setdefault(item.plan_id, reason)
            details = "; ".join(
                f"{plan_id}: {reason}" for plan_id, reason in sorted(rejected.items())
            )
            raise InfeasibleBudgetError(
                f"no Top-K candidate passed dry-run and measurement: {details}"
            )
        selected = next(
            plan for plan in evaluated if plan.plan.plan_id == selected_measured.plan_id
        )
        dry_run = dry_runs[selected.plan.plan_id]
        measured = selected_measured
    if not selected.simulation.simulated_memory_event_trace:
        if simulation_cost_cache is None and provider_cache_safe(cost_provider):
            simulation_cost_cache = build_simulation_cost_cache(
                ir,
                fixed_timeline,
                cost_provider,
            )
        selected = evaluate_plan(
            ir,
            selected.plan,
            fixed_timeline,
            cost_provider=cost_provider,
            simulation_cost_cache=simulation_cost_cache,
            materialize_event_trace=True,
        )
        evaluated = tuple(
            selected if item.plan.plan_id == selected.plan.plan_id else item
            for item in evaluated
        )
    fallback_ids = tuple(item.plan_id for item in measured_tuple if item.plan_id != selected.plan.plan_id)
    executor.current_plan_id = selected.plan.plan_id
    executor.executable = measured.forward_backward
    executor.activation_checkpoint = bool(measured.phase_metrics.get("activation_checkpoint", 0))
    executor.aot_partition_runtime = bool(measured.phase_metrics.get("aot_partition_runtime", 0))
    executor.selection_objective = config.selection_objective
    executor.fallback_executables = tuple(
        (item.plan_id, item.forward_backward)
        for item in measured_tuple
        if item.plan_id != selected.plan.plan_id and item.correctness_passed
    )
    executor.fallback_activation_checkpoints = {
        item.plan_id: bool(item.phase_metrics.get("activation_checkpoint", 0))
        for item in measured_tuple
        if item.plan_id != selected.plan.plan_id and item.correctness_passed
    }
    executor.fallback_aot_partition_runtimes = {
        item.plan_id: bool(item.phase_metrics.get("aot_partition_runtime", 0))
        for item in measured_tuple
        if item.plan_id != selected.plan.plan_id and item.correctness_passed
    }
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
    optimization_metrics["candidate_validation_count"] = len(validation_candidates)
    optimization_metrics["candidate_realization_count"] = int(simulation_only)
    optimization_metrics["simulation_only_selection"] = int(simulation_only)
    optimization_metrics["candidate_validation_limit"] = config.validation_top_k
    optimization_metrics["candidate_measurement_order_seed"] = (
        config.candidate_measurement_order_seed
    )
    optimization_metrics["candidate_compiler_reset_enabled"] = int(
        config.reset_compiler_before_candidate_measurement
    )
    optimization_metrics["candidate_validation_policy"] = (
        "simulation_only_ranked"
        if simulation_only
        else config.validation_selection_policy
    )
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
        candidate_attempts=tuple(candidate_attempts),
        cache_stats=CacheStats(layer_hits=cache_hits, layer_misses=cache_misses),
        optimization_metrics=optimization_metrics,
    )
