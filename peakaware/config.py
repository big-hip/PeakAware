from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FLOAT_DTYPE_ALIASES = {
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "torch.float32": "float32",
    "double": "float64",
    "fp64": "float64",
    "float64": "float64",
    "torch.float64": "float64",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "torch.float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
}


def normalize_float_dtype_name(dtype: str) -> str:
    try:
        return _FLOAT_DTYPE_ALIASES[dtype.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted({value for value in _FLOAT_DTYPE_ALIASES.values()}))
        raise ValueError(f"dtype must be one of: {allowed}") from exc


@dataclass(frozen=True)
class PeakAwareConfig:
    """Configuration switches for the M0/M1 PeakAware pipeline."""

    enable_compile: bool = False
    enable_inductor: bool = False
    enable_min_cut_baseline: bool = True
    manual_saved_value_ids: tuple[frozenset[int], ...] = ()
    safety_margin_ratio: float = 0.02
    safety_margin_bytes: int = 1 << 20
    top_k: int = 3
    max_greedy_candidates: int = 4
    search_algorithm: str = "greedy"
    beam_width: int = 16
    max_beam_candidates: int = 128
    beam_candidate_overflow_policy: str = "error"
    compiler_refinement_top_k: int = 0
    validation_top_k: int | None = None
    validation_selection_policy: str = "ranked"
    candidate_measurement_order_seed: int | None = None
    reset_compiler_before_candidate_measurement: bool = False
    zero_grad_set_to_none: bool = True
    require_cuda_measurement: bool = False
    allow_real_input_capture: bool = False
    capture_backend: str = "auto"
    isolate_candidate_measurement: bool = False
    candidate_worker_timeout_s: float = 60.0
    profile_db_path: str | Path | None = None
    cache_root: str | Path | None = None
    measurement_warmup_steps: int = 0
    measurement_repeats: int = 1
    candidate_measurement_protocol: str = "legacy_phase"
    runtime_peak_safety_margin_bytes: int = 1 << 20
    enable_diagnostic_hints: bool = True
    dynamic_shapes: dict[str, Any] | None = None
    gradient_accumulation_steps: int = 1
    fsdp_enabled: bool = False
    offload_enabled: bool = False
    selection_objective: str = "min_peak_then_time"
    rng_seed: int | None = 1337
    atol: float = 1e-5
    rtol: float = 1e-4
    precision_dtype: str = "float32"
    autocast_enabled: bool = False
    autocast_dtype: str | None = None
    grad_scaler_enabled: bool = False

    def precision_fingerprint(self) -> tuple[tuple[str, str | bool | None], ...]:
        autocast_dtype = None if self.autocast_dtype is None else normalize_float_dtype_name(self.autocast_dtype)
        return (
            ("precision_dtype", normalize_float_dtype_name(self.precision_dtype)),
            ("autocast_enabled", self.autocast_enabled),
            ("autocast_dtype", autocast_dtype),
            ("grad_scaler_enabled", self.grad_scaler_enabled),
        )

    def validate(self) -> None:
        if self.safety_margin_ratio < 0:
            raise ValueError("safety_margin_ratio must be non-negative")
        if self.safety_margin_bytes < 0:
            raise ValueError("safety_margin_bytes must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_greedy_candidates <= 0:
            raise ValueError("max_greedy_candidates must be positive")
        if self.search_algorithm not in {
            "greedy",
            "pareto_beam",
            "lagrangian_beam",
            "lagrangian_sweep_beam",
        }:
            raise ValueError(
                "search_algorithm must be one of: greedy, pareto_beam, "
                "lagrangian_beam, lagrangian_sweep_beam"
            )
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_beam_candidates <= 0:
            raise ValueError("max_beam_candidates must be positive")
        if self.beam_candidate_overflow_policy not in {"error", "coarsen_tail"}:
            raise ValueError(
                "beam_candidate_overflow_policy must be one of: error, coarsen_tail"
            )
        if self.compiler_refinement_top_k < 0:
            raise ValueError("compiler_refinement_top_k must be non-negative")
        if self.compiler_refinement_top_k > 0 and self.capture_backend != "aot":
            raise ValueError(
                "compiler_refinement_top_k requires capture_backend='aot'"
            )
        if self.validation_top_k is not None and self.validation_top_k < 0:
            raise ValueError("validation_top_k must be non-negative when set")
        if self.validation_selection_policy not in {"ranked", "structural_diverse"}:
            raise ValueError(
                "validation_selection_policy must be one of: ranked, structural_diverse"
            )
        if self.enable_inductor and not self.enable_compile:
            raise ValueError("enable_inductor requires enable_compile")
        if self.capture_backend not in {"auto", "aot", "fx"}:
            raise ValueError("capture_backend must be one of: auto, aot, fx")
        if self.candidate_worker_timeout_s <= 0:
            raise ValueError("candidate_worker_timeout_s must be positive")
        if self.measurement_warmup_steps < 0:
            raise ValueError("measurement_warmup_steps must be non-negative")
        if self.measurement_repeats <= 0:
            raise ValueError("measurement_repeats must be positive")
        if self.candidate_measurement_protocol not in {
            "legacy_phase",
            "publication_overall",
        }:
            raise ValueError(
                "candidate_measurement_protocol must be one of: "
                "legacy_phase, publication_overall"
            )
        if self.candidate_measurement_protocol == "publication_overall":
            if not self.enable_compile:
                raise ValueError(
                    "publication_overall candidate measurement requires enable_compile=True"
                )
            required_warmup = 10 if self.enable_inductor else 5
            if self.measurement_warmup_steps != required_warmup:
                raise ValueError(
                    "publication_overall candidate measurement requires "
                    f"measurement_warmup_steps={required_warmup}"
                )
            if self.measurement_repeats < 20:
                raise ValueError(
                    "publication_overall candidate measurement requires "
                    "measurement_repeats>=20"
                )
        if self.runtime_peak_safety_margin_bytes < 0:
            raise ValueError("runtime_peak_safety_margin_bytes must be non-negative")
        if self.dynamic_shapes is not None:
            raise ValueError("M0 does not support dynamic shape requests; set dynamic_shapes=None")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("M0 does not support gradient accumulation; set gradient_accumulation_steps=1")
        if self.fsdp_enabled:
            raise ValueError("M0 does not support FSDP; set fsdp_enabled=False")
        if self.offload_enabled:
            raise ValueError("M0 does not support offload; set offload_enabled=False")
        if self.selection_objective not in {"min_peak_then_time", "min_time_then_peak"}:
            raise ValueError("selection_objective must be one of: min_peak_then_time, min_time_then_peak")
        normalize_float_dtype_name(self.precision_dtype)
        if self.autocast_dtype is not None:
            normalize_float_dtype_name(self.autocast_dtype)
        if self.autocast_enabled:
            raise ValueError("M0 does not support autocast; set autocast_enabled=False")
        if self.autocast_dtype is not None:
            raise ValueError("autocast_dtype requires autocast_enabled=True")
        if self.grad_scaler_enabled:
            raise ValueError("M0 does not support GradScaler; set grad_scaler_enabled=False")
