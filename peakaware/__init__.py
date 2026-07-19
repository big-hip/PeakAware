"""PeakAware public package."""

from .api import optimize_training
from .config import PeakAwareConfig
from .contracts import CacheStats, OptimizedTrainingResult, StepResult
from .experiments import (
    ExperimentCase,
    ExperimentRecord,
    ExperimentSummary,
    run_experiment_matrix,
    summarize_baseline_comparisons,
    summarize_hint_ablation,
    summarize_layered_simulation_accuracy,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
)
from .microbatch import optimize_microbatches
from .external_baselines import summarize_external_baseline_capabilities
from .reporting import (
    export_plan_artifact_json,
    export_result_json,
    load_plan_artifact_json,
    render_text_report,
    summarize_plan_artifact,
    summarize_result,
    validate_plan_artifact_identity,
)
from .root_cause_benchmark import run_synthetic_root_cause_benchmark

__all__ = [
    "ExperimentCase",
    "ExperimentRecord",
    "ExperimentSummary",
    "CacheStats",
    "PeakAwareConfig",
    "export_plan_artifact_json",
    "export_result_json",
    "load_plan_artifact_json",
    "OptimizedTrainingResult",
    "render_text_report",
    "StepResult",
    "summarize_plan_artifact",
    "summarize_baseline_comparisons",
    "summarize_external_baseline_capabilities",
    "summarize_hint_ablation",
    "summarize_layered_simulation_accuracy",
    "summarize_experiment_records",
    "summarize_experiment_records_by_variant",
    "summarize_result",
    "validate_plan_artifact_identity",
    "optimize_microbatches",
    "optimize_training",
    "run_experiment_matrix",
    "run_synthetic_root_cause_benchmark",
]
