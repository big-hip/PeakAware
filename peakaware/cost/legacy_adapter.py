from __future__ import annotations

from peakaware.cost.base import OpCost, OpSignature, StaticCostProvider


class LegacyCostmodelAdapter:
    """Small adapter boundary for the existing Costmodel tree.

    The legacy package has model-specific analytical classes but no stable
    universal operator API yet, so M1 falls back to the static provider while
    preserving provenance.
    """

    def __init__(self, fallback: StaticCostProvider | None = None) -> None:
        self.fallback = fallback or StaticCostProvider()

    def supports(self, signature: OpSignature) -> bool:
        return self.fallback.supports(signature)

    def estimate(self, signature: OpSignature) -> OpCost | None:
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
    }


def from_legacy_result(result: object) -> OpCost | None:
    if isinstance(result, OpCost):
        return result
    return None
