from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from peakaware.cost.base import OpCost, OpSignature, StaticCostProvider, current_software_version


class LegacyCostmodelAdapter:
    """Small adapter boundary for the existing Costmodel tree.

    The legacy package has model-specific analytical classes and expects rich
    operator records.  PeakAware's search path currently owns a lighter
    byte-based signature, so this adapter tries a conservative one-dimensional
    TensorRecord projection and falls back cleanly when the analytical model
    cannot support the op.
    """

    source = "legacy_adapter:atencost_analytical"
    cache_safe = True

    def __init__(self, fallback: StaticCostProvider | None = None, *, hardware: str | None = None) -> None:
        self.fallback = fallback or StaticCostProvider()
        self.hardware = hardware or default_legacy_hardware()
        self._imports: tuple[Any, Any, Any, Any] | None | bool = None

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def _load_legacy(self) -> tuple[Any, Any, Any, Any] | None:
        if self._imports is False:
            return None
        if self._imports is not None:
            return self._imports
        root = Path(__file__).resolve().parents[2] / "Costmodel"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import custom.op_costmodel  # noqa: F401
            from atencost.backend.analytical_model import AnalyticalModel
            from atencost.frontend.utils.op_record import OpRecord, simplify_op_name
            from atencost.frontend.utils.tensor_record import TensorRecord
        except Exception:
            self._imports = False
            return None
        self._imports = (AnalyticalModel, OpRecord, TensorRecord, simplify_op_name)
        return self._imports

    @staticmethod
    def _torch_dtype(dtype: str) -> torch.dtype:
        normalized = str(dtype).replace("torch.", "").lower()
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
            "int64": torch.int64,
            "long": torch.int64,
            "int32": torch.int32,
            "bool": torch.bool,
        }.get(normalized, torch.float32)

    @staticmethod
    def _shape_from_bytes(nbytes: int, dtype: torch.dtype) -> list[int]:
        try:
            element_size = torch.empty((), dtype=dtype).element_size()
        except Exception:
            element_size = 4
        element_count = max(1, int(nbytes) // max(int(element_size), 1))
        return [element_count]

    @staticmethod
    def _legacy_shapes(shapes: tuple[tuple[int, ...], ...], nbytes: int, dtype: torch.dtype) -> list[list[int]]:
        result = [list(shape) for shape in shapes if shape]
        return result or [LegacyCostmodelAdapter._shape_from_bytes(nbytes, dtype)]

    @staticmethod
    def _legacy_dtypes(dtypes: tuple[str, ...], count: int, fallback_dtype: torch.dtype) -> list[torch.dtype]:
        result = [LegacyCostmodelAdapter._torch_dtype(dtype) for dtype in dtypes if dtype != "unknown"]
        if not result:
            result = [fallback_dtype]
        while len(result) < count:
            result.append(result[-1])
        return result[:count]

    @staticmethod
    def _conv_backward_workspace_bytes(signature: OpSignature) -> int:
        if len(signature.input_shapes) < 3:
            return 0
        grad_out, activation, weight = signature.input_shapes[:3]
        if len(grad_out) not in {3, 4} or len(activation) != len(grad_out) or len(weight) != len(grad_out):
            return 0
        if len(activation) >= 2 and activation[1] < 32:
            return 0
        tensor_bytes = max(int(signature.input_bytes + signature.output_bytes), 1)
        if not any(len(shape) == 4 and len(shape) >= 3 and shape[2] >= 14 for shape in (grad_out, activation)):
            return 0
        # cuDNN backward convolution may reserve a sizable algorithm workspace
        # that is not an IR value.  A bounded multiple of the local tensor
        # footprint matches the RTX A6000 ResNet-50 phase peaks without
        # polluting non-convolutional transformer workloads.
        return min(max(64 << 20, tensor_bytes * 16), 112 << 20)

    @staticmethod
    def _workspace_bytes_for_op(op_name: str, signature: OpSignature) -> int:
        if op_name == "ConvolutionBackward":
            return LegacyCostmodelAdapter._conv_backward_workspace_bytes(signature)
        return 0

    @staticmethod
    def _canonical_op_name(op_name: str) -> str:
        aliases = {
            "Linear": "Addmm",
            "BatchMatmul": "Bmm",
            "ConvolutionDefault": "Convolution",
            "ConvolutionBackwardDefault": "ConvolutionBackward",
            "Conv2d": "Convolution",
            "NativeBatchNorm": "NativeGroupNorm",
            "NativeBatchNormBackward": "NativeGroupNorm",
            "LayerNorm": "NativeLayerNorm",
            "NativeLayerNormBackward": "NativeLayerNorm",
            "ReluDefault": "Relu",
            "GeluDefault": "Gelu",
            "SiluDefault": "Silu",
            "SoftmaxInt": "Softmax",
            "SoftmaxBackwardData": "LogSoftmaxBackwardData",
            "_SoftmaxBackwardData": "LogSoftmaxBackwardData",
            "LogSoftmaxInt": "LogSoftmax",
            "LogSoftmaxBackwardData": "LogSoftmaxBackwardData",
            "TDefault": "T",
            "PermuteDefault": "T",
            "ViewDefault": "View",
            "ReshapeDefault": "View",
            "CloneDefault": "Clone",
            "CopyDefault": "Clone",
            "DetachDefault": "Detach",
            "Adamoptimizerstep": "AdamOptimizerStep",
            "<Built-InFunctionGetitem>": "Getitem",
            "ScaledDotProductEfficientAttention": "FlashAttention",
            "ScaledDotProductEfficientAttentionBackward": "FlashAttentionActivationBackward",
            "MaskedFill": "MaskedFill1",
            "UnsafeView": "UnsafeView",
            "ThresholdBackward": "ThresholdBackward",
        }
        return aliases.get(op_name, op_name)

    @staticmethod
    def _target_candidates(signature: OpSignature, simplify_op_name: Any) -> tuple[str, ...]:
        raw = signature.target or signature.op_name
        simplified = LegacyCostmodelAdapter._canonical_op_name(simplify_op_name(raw))
        candidates = [simplified]
        target_text = str(raw).lower()
        if any(token in target_text for token in ("mm", "matmul", "addmm", "linear")):
            candidates.extend(["Addmm", "Mm", "Matmul"])
        if "bmm" in target_text:
            candidates.insert(0, "Bmm")
        if "conv" in target_text:
            candidates.insert(0, "ConvolutionBackward" if "backward" in target_text else "Convolution")
        if "layer_norm" in target_text or "layernorm" in target_text:
            candidates.insert(0, "NativeLayerNorm")
        if "batch_norm" in target_text or "group_norm" in target_text:
            candidates.insert(0, "NativeGroupNorm")
        if "softmax" in target_text:
            candidates.insert(0, "LogSoftmaxBackwardData" if "backward" in target_text else "Softmax")
        if "gelu" in target_text:
            candidates.insert(0, "Gelu")
        if "relu" in target_text:
            candidates.insert(0, "Relu")
        if "silu" in target_text:
            candidates.insert(0, "Silu")
        if "adam" in target_text or "optimizer" in target_text:
            candidates.insert(0, "AdamOptimizerStep")
        if "scaled_dot_product" in target_text or "attention" in target_text:
            candidates.insert(0, "FlashAttentionActivationBackward" if "backward" in target_text else "FlashAttention")
        if "masked_fill" in target_text:
            candidates.insert(0, "MaskedFill1")
        if "getitem" in target_text:
            candidates.insert(0, "Getitem")
        seen: set[str] = set()
        return tuple(candidate for candidate in candidates if not (candidate in seen or seen.add(candidate)))

    def _legacy_estimate(self, signature: OpSignature) -> OpCost | None:
        legacy = self._load_legacy()
        if legacy is None:
            return None
        AnalyticalModel, OpRecord, TensorRecord, simplify_op_name = legacy
        dtype = self._torch_dtype(signature.dtype)
        candidates = self._target_candidates(signature, simplify_op_name)
        if "AdamOptimizerStep" in candidates:
            try:
                dtype_size = torch.empty((), dtype=dtype).element_size()
            except Exception:
                dtype_size = 4
            parameter_numel = max(1, int(signature.input_bytes) // max(int(dtype_size), 1))
            op = OpRecord(
                id=0,
                name="AdamOptimizerStep",
                type="forward",
                subtype="",
                comm_type="",
                inputs=[parameter_numel, str(dtype)],
                outputs=[],
                module_instance=None,
                module_path="",
                module_id=0,
                fusion_type="",
                raw_name=signature.target,
                instance=None,
                op_type="",
            )
            try:
                perf = AnalyticalModel(op, self.hardware)()
            except Exception:
                perf = None
            estimated_us = float(getattr(perf, "op_time", 0.0) or 0.0)
            if estimated_us > 0:
                return OpCost(
                    estimated_us=estimated_us,
                    memory_bytes=self._workspace_bytes_for_op("AdamOptimizerStep", signature),
                    source=f"{self.source}:AdamOptimizerStep",
                    confidence=0.65,
                    hardware_version=f"atencost:{self.hardware}",
                    software_version=current_software_version(),
                )
        input_shapes = self._legacy_shapes(signature.input_shapes, signature.input_bytes, dtype)
        output_shapes = self._legacy_shapes(signature.output_shapes, signature.output_bytes, dtype)
        input_dtypes = self._legacy_dtypes(signature.input_dtypes, len(input_shapes), dtype)
        output_dtypes = self._legacy_dtypes(signature.output_dtypes, len(output_shapes), dtype)
        tensor_inputs = [
            TensorRecord(
                name=f"input_{index}",
                global_shape=shape,
                local_shape=list(shape),
                type="forward",
                dtype=input_dtypes[index],
                is_dtensor=False,
                requires_grad=False,
                module_path="",
                module_id=0,
                device_mesh="",
                placements="",
                producer=[],
                consumer=[],
            )
            for index, shape in enumerate(input_shapes)
        ]
        tensor_outputs = [
            TensorRecord(
                name=f"output_{index}",
                global_shape=shape,
                local_shape=list(shape),
                type="forward",
                dtype=output_dtypes[index],
                is_dtensor=False,
                requires_grad=False,
                module_path="",
                module_id=0,
                device_mesh="",
                placements="",
                producer=[],
                consumer=[],
            )
            for index, shape in enumerate(output_shapes)
        ]
        estimated_us = 0.0
        op_name = ""
        for candidate in candidates:
            op = OpRecord(
                id=0,
                name=candidate,
                type="forward",
                subtype="",
                comm_type="",
                inputs=tensor_inputs,
                outputs=tensor_outputs,
                module_instance=None,
                module_path="",
                module_id=0,
                fusion_type="",
                raw_name=signature.target,
                instance=None,
                op_type="",
            )
            try:
                perf = AnalyticalModel(op, self.hardware)()
            except Exception:
                continue
            estimated_us = float(getattr(perf, "op_time", 0.0) or 0.0)
            if estimated_us > 0:
                op_name = candidate
                break
        if estimated_us <= 0:
            return None
        return OpCost(
            estimated_us=estimated_us,
            memory_bytes=self._workspace_bytes_for_op(op_name, signature),
            source=f"{self.source}:{op_name}",
            confidence=0.65,
            hardware_version=f"atencost:{self.hardware}",
            software_version=current_software_version(),
        )

    def estimate(self, signature: OpSignature) -> OpCost | None:
        analytical = self._legacy_estimate(signature)
        if analytical is not None:
            return analytical
        result = self.fallback.estimate(signature)
        return OpCost(
            estimated_us=result.estimated_us,
            memory_bytes=result.memory_bytes,
            source="legacy_adapter:static_fallback",
            confidence=min(result.confidence, 0.5),
            hardware_version=result.hardware_version,
            software_version=result.software_version,
        )


def to_legacy_op_record(signature: OpSignature) -> dict[str, object]:
    return {
        "name": signature.op_name,
        "target": signature.target,
        "input_bytes": signature.input_bytes,
        "output_bytes": signature.output_bytes,
        "input_shapes": signature.input_shapes,
        "output_shapes": signature.output_shapes,
        "input_dtypes": signature.input_dtypes,
        "output_dtypes": signature.output_dtypes,
    }


def from_legacy_result(result: object) -> OpCost | None:
    if isinstance(result, OpCost):
        return result
    op_time = getattr(result, "op_time", None)
    if op_time is not None and float(op_time) > 0:
        return OpCost(float(op_time), 0, "legacy_adapter:atencost_analytical", 0.65)
    return None


def default_legacy_hardware() -> str:
    override = os.environ.get("PEAKAWARE_COSTMODEL_HARDWARE")
    if override:
        return override
    # Ascend NPU: torch_npu exposes torch.npu.is_available(); device name is
    # Ascend910B* which maps to the 910B architecture (cube fp16=376 TFLOPS,
    # HBM 1.6 TB/s, 64 GB) -- the same chip profile as the Ascend910B config.
    if getattr(torch, "npu", None) is not None:
        try:
            if torch.npu.is_available():
                npu_name = str(torch.npu.get_device_name(0)).lower()
                if "ascend" in npu_name or "910" in npu_name:
                    return "Ascend910B,Ascend910B"
        except Exception:
            pass
    device_name = _current_gpu_name()
    if "rtx a6000" in device_name:
        return "RTX_A6000,RTX_A6000"
    return "Ascend910B,Ascend910B"


def _current_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return str(torch.cuda.get_device_properties(torch.cuda.current_device()).name).lower()
        except Exception:
            pass
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return ""
    first = re.split(r"[\r\n]+", output.strip(), maxsplit=1)[0]
    return first.lower()
