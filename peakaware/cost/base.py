from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
        )


def signature_for_op(ir: JointTrainingIR, op: OpInfo) -> OpSignature:
    value_by_id = {value.id: value for value in ir.values}
    input_bytes = sum(value_by_id[v].logical_nbytes for v in op.input_value_ids if v in value_by_id)
    output_bytes = sum(value_by_id[v].logical_nbytes for v in op.output_value_ids if v in value_by_id)
    return OpSignature(op_name=op.name, target=op.target, input_bytes=input_bytes, output_bytes=output_bytes)
