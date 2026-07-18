from __future__ import annotations

from dataclasses import dataclass


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
    rng_seed: int | None = 1337
    atol: float = 1e-5
    rtol: float = 1e-4

    def validate(self) -> None:
        if self.safety_margin_ratio < 0:
            raise ValueError("safety_margin_ratio must be non-negative")
        if self.safety_margin_bytes < 0:
            raise ValueError("safety_margin_bytes must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.enable_inductor and not self.enable_compile:
            raise ValueError("enable_inductor requires enable_compile")
