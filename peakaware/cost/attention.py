from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from peakaware.cost.base import OpCost, OpSignature, current_software_version


@dataclass(frozen=True)
class AttentionHardwareSpec:
    hardware_id: str
    compute_tflops: dict[str, float]
    vector_tflops: dict[str, float]
    hbm_tbps: float
    fused_launch_us: float
    compute_utilization: float = 0.10
    vector_utilization: float = 0.10
    hbm_utilization: float = 0.80


def _hardware_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "Costmodel/zhanlu/backend/analytical_model/hardware/chip_configs"
    )


def _current_chip_name() -> str | None:
    override = os.environ.get("PEAKAWARE_COSTMODEL_HARDWARE")
    if override:
        return override.partition(",")[0]
    if not torch.cuda.is_available():
        return None
    name = str(torch.cuda.get_device_properties(torch.cuda.current_device()).name).lower()
    if "rtx a6000" in name:
        return "RTX_A6000"
    return None


def load_attention_hardware(chip_name: str | None = None) -> AttentionHardwareSpec | None:
    selected = chip_name or _current_chip_name()
    if not selected:
        return None
    path = _hardware_root() / f"{selected}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AttentionHardwareSpec(
        hardware_id=selected,
        compute_tflops={str(key): float(value) for key, value in payload["compute"]["cube"].items()},
        vector_tflops={str(key): float(value) for key, value in payload["compute"]["vector"].items()},
        hbm_tbps=float(payload["memory"]["hbm"]["bandwidth"]),
        fused_launch_us=float(payload.get("head_tail_time_for_fused_op", 1.0)),
    )


def _dtype_name(signature: OpSignature, index: int) -> str:
    raw = signature.input_dtypes[index] if index < len(signature.input_dtypes) else signature.dtype
    normalized = str(raw).replace("torch.", "").lower()
    if normalized in {"half", "float16"}:
        return "float16"
    if normalized in {"bfloat16"}:
        return "bfloat16"
    return "float32"


def _dtype_bytes(dtype: str) -> int:
    return 2 if dtype in {"float16", "bfloat16"} else 4


def _shape_bytes(shape: tuple[int, ...], dtype: str) -> int:
    elements = 1
    for dimension in shape:
        elements *= max(int(dimension), 0)
    return elements * _dtype_bytes(dtype)


class ScaledDotProductAttentionCostProvider:
    """Fusion-aware roofline model for CUDA fused scaled-dot-product attention.

    The model treats SDPA as one fused kernel instead of summing materialized
    QK, softmax and AV intermediates. Register/shared-memory tiles are not
    reported as allocator workspace because they do not contribute to
    ``torch.cuda.max_memory_allocated``. Exact A6000 profiles can override this
    provider through the higher-priority ProfileDB providers.
    """

    source = "analytical:sdpa_fused"
    cache_safe = True

    def __init__(self, hardware: AttentionHardwareSpec | None = None) -> None:
        self.hardware = hardware if hardware is not None else load_attention_hardware()

    @staticmethod
    def _is_supported_target(target: str) -> bool:
        normalized = str(target).lower()
        return "scaled_dot_product" in normalized and any(
            token in normalized for token in ("efficient_attention", "flash_attention", "cudnn_attention")
        )

    def supports(self, signature: OpSignature) -> bool:
        return self.hardware is not None and self._is_supported_target(signature.target)

    @staticmethod
    def _qkv(signature: OpSignature) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool] | None:
        shapes = tuple(shape for shape in signature.input_shapes if len(shape) == 4)
        backward = "backward" in str(signature.target).lower()
        if backward:
            if len(shapes) < 4:
                return None
            query, key, value = shapes[1:4]
        else:
            if len(shapes) < 3:
                return None
            query, key, value = shapes[:3]
        if any(int(dimension) <= 0 for shape in (query, key, value) for dimension in shape):
            return None
        if query[0] != key[0] or query[0] != value[0] or query[1] != key[1] or query[1] != value[1]:
            return None
        return query, key, value, backward

    def estimate(self, signature: OpSignature) -> OpCost | None:
        if not self.supports(signature) or self.hardware is None:
            return None
        qkv = self._qkv(signature)
        if qkv is None:
            return None
        query, key, value, backward = qkv
        batch, heads, query_length, head_dim = (int(dimension) for dimension in query)
        key_length = int(key[-2])
        value_dim = int(value[-1])
        score_elements = batch * heads * query_length * key_length
        if backward:
            # QK recomputation, dP/dV, and dQ/dK matrix products.
            matrix_flops = score_elements * (6 * head_dim + 4 * value_dim)
            vector_flops = 10 * score_elements
            output_bytes = sum(
                _shape_bytes(shape, _dtype_name(signature, index))
                for index, shape in enumerate((query, key, value), start=1)
            )
        else:
            # QK^T and P@V plus fused scale/mask/softmax work.
            matrix_flops = 2 * score_elements * (head_dim + value_dim)
            vector_flops = 5 * score_elements
            output_bytes = _shape_bytes(query, _dtype_name(signature, 0)) + 4 * batch * heads * query_length
        dtype = _dtype_name(signature, 1 if backward else 0)
        cube_tflops = self.hardware.compute_tflops.get(dtype) or self.hardware.compute_tflops["float32"]
        vector_tflops = self.hardware.vector_tflops.get(dtype) or self.hardware.vector_tflops["float32"]
        cube_us = matrix_flops / (
            cube_tflops * 1_000_000.0 * max(self.hardware.compute_utilization, 1e-6)
        )
        vector_us = vector_flops / (
            vector_tflops * 1_000_000.0 * max(self.hardware.vector_utilization, 1e-6)
        )
        memory_bytes = max(int(signature.input_bytes), 0) + max(int(signature.output_bytes), output_bytes)
        memory_us = memory_bytes / (
            self.hardware.hbm_tbps * 1_000_000.0 * max(self.hardware.hbm_utilization, 1e-6)
        )
        estimated_us = max(cube_us + 0.5 * vector_us, memory_us) + self.hardware.fused_launch_us
        return OpCost(
            estimated_us=max(float(estimated_us), self.hardware.fused_launch_us),
            memory_bytes=0,
            source=self.source,
            confidence=0.60,
            hardware_version=f"zhanlu:{self.hardware.hardware_id}",
            software_version=current_software_version(),
        )
