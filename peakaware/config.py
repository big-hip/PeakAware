from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    runtime_peak_safety_margin_bytes: int = 1 << 20
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
        if self.runtime_peak_safety_margin_bytes < 0:
            raise ValueError("runtime_peak_safety_margin_bytes must be non-negative")
        normalize_float_dtype_name(self.precision_dtype)
        if self.autocast_dtype is not None:
            normalize_float_dtype_name(self.autocast_dtype)
        if self.autocast_enabled:
            raise ValueError("M0 does not support autocast; set autocast_enabled=False")
        if self.autocast_dtype is not None:
            raise ValueError("autocast_dtype requires autocast_enabled=True")
        if self.grad_scaler_enabled:
            raise ValueError("M0 does not support GradScaler; set grad_scaler_enabled=False")
