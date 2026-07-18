from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import Tensor, fx, nn


@dataclass(frozen=True)
class GuardSpec:
    name: str
    value: str


@dataclass(frozen=True)
class ParameterBinding:
    name: str
    index: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    nbytes: int


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    param_group_count: int
    parameter_count: int
    state_bytes: int
    temporary_bytes: int


@dataclass(frozen=True)
class HardwareSpec:
    device: str
    cuda_available: bool
    total_memory_bytes: int | None


@dataclass(frozen=True)
class TrainingRequest:
    model: nn.Module
    example_args: tuple[Any, ...]
    example_kwargs: dict[str, Any]
    loss_fn: Callable[..., Tensor]
    optimizer: torch.optim.Optimizer
    memory_budget_bytes: int
    config: Any
    optimizer_spec: OptimizerSpec
    hardware: HardwareSpec
    request_key: str


@dataclass(frozen=True)
class CapturedJointGraph:
    joint_module: fx.GraphModule
    guards: tuple[GuardSpec, ...]
    parameter_mapping: tuple[ParameterBinding, ...]
    capture_key: str
    fw_module: fx.GraphModule | None = None
    bw_module: fx.GraphModule | None = None
    backend: str = "fx"
    failures: tuple[FailureRecord, ...] = ()
    num_fwd_outputs: int = 1
    static_lifetime_input_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class OpInfo:
    id: int
    name: str
    target: str
    phase: str
    input_value_ids: tuple[int, ...]
    output_value_ids: tuple[int, ...]
    recomputable: bool
    mandatory_save_reason: str | None


@dataclass(frozen=True)
class ValueInfo:
    id: int
    producer_id: int | None
    consumer_ids: tuple[int, ...]
    storage_id: int | None
    logical_nbytes: int
    phase: str
    crosses_fw_bw: bool
    recomputable: bool
    mandatory_save_reason: str | None
    name: str = ""


@dataclass(frozen=True)
class StorageInfo:
    id: int
    value_ids: tuple[int, ...]
    physical_nbytes: int
    is_external: bool


@dataclass(frozen=True)
class RegionInfo:
    id: int
    name: str
    op_ids: tuple[int, ...]
    kind: str = "generic"


@dataclass(frozen=True)
class JointTrainingIR:
    ops: tuple[OpInfo, ...]
    values: tuple[ValueInfo, ...]
    storages: tuple[StorageInfo, ...]
    regions: tuple[RegionInfo, ...]
    graph_key: str


@dataclass(frozen=True)
class IRValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedTimeline:
    parameter_bytes: int
    buffer_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    optimizer_temporary_bytes: int
    mandatory_workspace_bytes: int = 0

    @property
    def steady_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.buffer_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
        )

    @property
    def peak_lower_bound_bytes(self) -> int:
        return self.steady_bytes + self.optimizer_temporary_bytes + self.mandatory_workspace_bytes


@dataclass(frozen=True)
class CoarseFeasibilityReport:
    user_budget_bytes: int
    fixed_peak_lower_bound: int
    activation_budget_bytes: int
    status: str
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class FeasibilityReport:
    user_budget_bytes: int
    fixed_peak_lower_bound: int
    activation_budget_bytes: int
    dominant_phase: str
    status: str
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class StorageEffect:
    storage_id: int
    decision: str
    decision_value_ids: tuple[int, ...]
    alias_value_ids: tuple[int, ...]
    released_at_peak_bytes: int
    retained_after_fw_bytes: int
    pinning_value_ids: tuple[int, ...]
    confidence: float


@dataclass(frozen=True)
class PlanMetrics:
    estimated_peak_bytes: int
    estimated_step_us: float
    max_recompute_live_bytes: int
    recompute_span_ops: int
    recompute_before_first_bw_op_bytes: int
    risk_score: float
    confidence: float


@dataclass(frozen=True)
class RecomputePlan:
    graph_key: str
    budget_bytes: int
    storage_effects: tuple[StorageEffect, ...]
    saved_value_ids: frozenset[int]
    mandatory_value_ids: frozenset[int]
    estimated_peak_bytes: int
    estimated_step_us: float
    max_recompute_live_bytes: int
    recompute_span_ops: int
    recompute_before_first_bw_op_bytes: int
    risk_score: float
    confidence: float
    safety_margin_bytes: int
    cost_sources: tuple[str, ...]
    plan_id: str = ""


@dataclass(frozen=True)
class PeakSnapshot:
    phase: str
    op_id: int | None
    live_storage_ids: frozenset[int]
    live_bytes: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    saved_activation_bytes: int
    recomputed_bytes: int
    workspace_bytes: int


@dataclass(frozen=True)
class SimulationResult:
    plan_id: str
    estimated_peak_bytes: int
    estimated_step_us: float
    peak_snapshot: PeakSnapshot
    after_fw_retained_bytes: int
    fw_peak_bytes: int
    bw_peak_bytes: int
    optimizer_peak_bytes: int
    max_recompute_live_bytes: int
    recompute_span_ops: int
    recompute_before_first_bw_op_bytes: int
    risk_score: float
    confidence: float


@dataclass(frozen=True)
class AnalysisBundle:
    ir: JointTrainingIR
    fixed_timeline: FixedTimeline
    baseline_results: tuple["EvaluatedPlan", ...]
    analysis_key: str


@dataclass(frozen=True)
class EvaluatedPlan:
    plan: RecomputePlan
    simulation: SimulationResult
    feasible: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class PartitionABI:
    fw_output_value_ids: tuple[int, ...]
    bw_placeholder_value_ids: tuple[int, ...]
    tangent_value_ids: tuple[int, ...]
    rng_state_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class LoweredPartition:
    plan_id: str
    fw_graph: fx.GraphModule
    bw_graph: fx.GraphModule
    partition_abi: PartitionABI


@dataclass(frozen=True)
class DryRunResult:
    plan_id: str
    abi_valid: bool
    outputs_match: bool
    gradients_match: bool
    rng_match: bool
    failure_reason: str | None


@dataclass(frozen=True)
class MeasuredExecutable:
    plan_id: str
    forward_backward: Callable[..., Tensor]
    measured_peak_bytes: int
    measured_step_us: float
    correctness_passed: bool
    phase_metrics: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    loss: Tensor
    optimizer_step_performed: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingStepExecutor(Protocol):
    def step(self, *args: Any, **kwargs: Any) -> StepResult:
        """Run zero_grad, forward/backward and eager optimizer.step."""


@dataclass(frozen=True)
class RepairHint:
    kind: str
    target_ids: tuple[int, ...]
    priority: float
    reason: str


@dataclass(frozen=True)
class FailureRecord:
    stage: str
    error_type: str
    message: str
    recovered: bool
    next_fallback: str | None = None


@dataclass(frozen=True)
class OptimizedTrainingResult:
    selected_plan: RecomputePlan
    executable: MeasuredExecutable
    executor: TrainingStepExecutor
    feasibility: FeasibilityReport
    fallback_plan_ids: tuple[str, ...]
    analysis: AnalysisBundle | None = None
    dry_run: DryRunResult | None = None


@dataclass(frozen=True)
class TrainingTaskSpec:
    name: str
    build_model: Callable[[], nn.Module]
    build_batch: Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]
    loss_fn: Callable[..., Tensor]
    build_optimizer: Callable[[nn.Module], torch.optim.Optimizer]
    dynamic_shapes: dict[str, Any] | None = None


@dataclass(frozen=True)
class MicrobatchCandidateResult:
    microbatch_size: int
    result: OptimizedTrainingResult
    useful_samples_per_second: float


@dataclass(frozen=True)
class MicrobatchSearchResult:
    candidates: tuple[MicrobatchCandidateResult, ...]
    selected: MicrobatchCandidateResult
