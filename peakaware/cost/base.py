from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from peakaware.contracts import JointTrainingIR, OpInfo


@dataclass(frozen=True)
class OpSignature:
    op_name: str
    target: str
    input_bytes: int
    output_bytes: int
    dtype: str = "unknown"


@dataclass(frozen=True)
class OpCost:
    estimated_us: float
    memory_bytes: int
    source: str
    confidence: float
    hardware_version: str = "unknown"
    software_version: str = "unknown"


def current_hardware_version() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = ".".join(str(item) for item in torch.cuda.get_device_capability(device))
    return f"cuda:{properties.name}:sm{capability}"


def current_software_version() -> str:
    cuda = "cpu" if torch.version.cuda is None else f"cuda:{torch.version.cuda}"
    return f"torch:{torch.__version__};{cuda}"


class CostProvider(Protocol):
    def supports(self, signature: OpSignature) -> bool:
        ...

    def estimate(self, signature: OpSignature) -> OpCost | None:
        ...


class StaticCostProvider:
    """Shape-derived fallback cost model used by M1 search."""

    source = "static_fallback"

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def estimate(self, signature: OpSignature) -> OpCost:
        tensor_bytes = max(signature.input_bytes + signature.output_bytes, 1)
        estimated_us = max(1.0, tensor_bytes / (32 << 20))
        return OpCost(
            estimated_us=estimated_us,
            memory_bytes=signature.output_bytes,
            source=self.source,
            confidence=0.55,
            hardware_version="generic",
            software_version=current_software_version(),
        )


class RooflineFallbackProvider:
    source = "roofline_fallback"

    def __init__(self, *, bandwidth_bytes_per_us: float = 32 << 20, launch_overhead_us: float = 1.0) -> None:
        self.bandwidth_bytes_per_us = bandwidth_bytes_per_us
        self.launch_overhead_us = launch_overhead_us

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def estimate(self, signature: OpSignature) -> OpCost:
        tensor_bytes = max(signature.input_bytes + signature.output_bytes, 1)
        latency = self.launch_overhead_us + tensor_bytes / self.bandwidth_bytes_per_us
        workspace = max(0, signature.output_bytes // 8)
        return OpCost(
            latency,
            workspace,
            self.source,
            0.2,
            hardware_version=current_hardware_version(),
            software_version=current_software_version(),
        )


def signature_for_op(ir: JointTrainingIR, op: OpInfo) -> OpSignature:
    value_by_id = {value.id: value for value in ir.values}
    input_bytes = sum(value_by_id[v].logical_nbytes for v in op.input_value_ids if v in value_by_id)
    output_bytes = sum(value_by_id[v].logical_nbytes for v in op.output_value_ids if v in value_by_id)
    return OpSignature(op_name=op.name, target=op.target, input_bytes=input_bytes, output_bytes=output_bytes)
