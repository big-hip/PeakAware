"""PeakAware public package."""

from .api import optimize_training
from .config import PeakAwareConfig
from .contracts import OptimizedTrainingResult, StepResult
from .microbatch import optimize_microbatches
from .reporting import export_result_json, render_text_report, summarize_result

__all__ = [
    "PeakAwareConfig",
    "export_result_json",
    "OptimizedTrainingResult",
    "render_text_report",
    "StepResult",
    "summarize_result",
    "optimize_microbatches",
    "optimize_training",
]
