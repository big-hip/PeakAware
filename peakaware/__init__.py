"""PeakAware public package."""

from .api import optimize_training
from .config import PeakAwareConfig
from .contracts import CacheStats, OptimizedTrainingResult, StepResult
from .experiments import ExperimentCase, ExperimentRecord, run_experiment_matrix
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
    "CacheStats",
    "PeakAwareConfig",
    "export_plan_artifact_json",
    "export_result_json",
    "OptimizedTrainingResult",
    "render_text_report",
    "StepResult",
    "summarize_plan_artifact",
    "summarize_result",
    "optimize_microbatches",
    "optimize_training",
    "run_experiment_matrix",
]
