"""PeakAware public package."""

from .api import optimize_training
from .config import PeakAwareConfig
from .contracts import CacheStats, OptimizedTrainingResult, StepResult
from .experiments import (
    ExperimentCase,
    ExperimentRecord,
    ExperimentSummary,
    run_experiment_matrix,
    summarize_hint_ablation,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
)
from .microbatch import optimize_microbatches
from .reporting import (
    export_plan_artifact_json,
    export_result_json,
    render_text_report,
    summarize_plan_artifact,
    summarize_result,
)

__all__ = [
    "ExperimentCase",
    "ExperimentRecord",
    "ExperimentSummary",
    "CacheStats",
    "PeakAwareConfig",
    "export_plan_artifact_json",
    "export_result_json",
    "OptimizedTrainingResult",
    "render_text_report",
    "StepResult",
    "summarize_plan_artifact",
    "summarize_hint_ablation",
    "summarize_experiment_records",
    "summarize_experiment_records_by_variant",
    "summarize_result",
    "optimize_microbatches",
    "optimize_training",
    "run_experiment_matrix",
]
