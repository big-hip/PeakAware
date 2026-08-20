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
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    input_dtypes: tuple[str, ...] = ()
    output_dtypes: tuple[str, ...] = ()


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
    """Operator-cost interface.

    Search-level caching is opt-in. A concrete provider class must define its
    own ``cache_safe`` attribute (or property) and return ``True``; inheriting
    that marker from a deterministic base class is intentionally insufficient
    because a subclass may override ``estimate`` with stateful behavior.
    """

    def supports(self, signature: OpSignature) -> bool:
        ...

    def estimate(self, signature: OpSignature) -> OpCost | None:
        ...


def provider_cache_safe(provider: CostProvider | None) -> bool:
    """Return whether this concrete provider explicitly opts into caching."""

    if provider is None:
        return True
    if "cache_safe" not in type(provider).__dict__:
        return False
    return bool(getattr(provider, "cache_safe", False))


class StaticCostProvider:
    """Shape-derived fallback cost model used by M1 search.

    ``memory_bytes`` is interpreted by the memory simulator as transient
    workspace, not as normal operator output storage.  Unknown ops therefore
    report zero workspace instead of double-counting outputs that are already
    represented in the IR liveness model.
    """

    source = "static_fallback"
    cache_safe = True

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def estimate(self, signature: OpSignature) -> OpCost:
        tensor_bytes = max(signature.input_bytes + signature.output_bytes, 1)
        estimated_us = max(1.0, tensor_bytes / (32 << 20))
        return OpCost(
            estimated_us=estimated_us,
            memory_bytes=0,
            source=self.source,
            confidence=0.55,
            hardware_version="generic",
            software_version=current_software_version(),
        )


class StructuralZeroCostProvider:
    """Explicit zero-cost model for FX/AOT graph interface nodes.

    Placeholders, tangents and the graph output describe dataflow boundaries;
    they do not launch kernels or allocate transient workspaces. Treating them
    as unknown operators adds artificial launch latency to every simulation.
    """

    source = "structural_zero"
    cache_safe = True

    @staticmethod
    def _is_structural_name(value: str) -> bool:
        normalized = str(value).strip().lower()
        if normalized == "output":
            return True
        prefixes = (
            "primals_",
            "tangents_",
            "placeholder_",
            "lifted_tensor_",
        )
        if normalized.startswith("aten."):
            return False
        if normalized.startswith(prefixes):
            return True
        # FX-generated positional placeholders use names such as ``arg0`` or
        # ``arg_1``.  Do not use a broad ``arg*`` match: real kernels such as
        # argmax/argmin may also appear as node names.
        return (
            normalized.startswith("arg")
            and len(normalized) > 3
            and (normalized[3].isdigit() or normalized[3] == "_")
        )

    def supports(self, signature: OpSignature) -> bool:
        return self._is_structural_name(signature.target) or self._is_structural_name(
            signature.op_name
        )

    def estimate(self, signature: OpSignature) -> OpCost | None:
        if not self.supports(signature):
            return None
        return OpCost(
            estimated_us=0.0,
            memory_bytes=0,
            source=self.source,
            confidence=1.0,
            hardware_version=current_hardware_version(),
            software_version=current_software_version(),
        )


class MetadataViewCostProvider:
    """Zero CUDA-kernel cost for alias-preserving ATen metadata operations.

    These operations may create a new Tensor object on the host, but do not
    launch a CUDA kernel or allocate device storage. Device storage lifetime
    remains represented by IR storage aliases; this provider only prevents a
    legacy per-kernel launch penalty from being charged to metadata nodes.
    """

    source = "metadata_view_zero"
    cache_safe = True
    _TARGET_PREFIXES = (
        "aten._unsafe_view.",
        "aten.alias.",
        "aten.as_strided.",
        "aten.detach.",
        "aten.expand.",
        "aten.permute.",
        "aten.select.",
        "aten.slice.",
        "aten.squeeze.",
        "aten.t.",
        "aten.transpose.",
        "aten.unsqueeze.",
        "aten.view.",
    )

    @classmethod
    def _is_metadata_target(cls, target: str) -> bool:
        normalized = str(target).strip().lower()
        return normalized in {
            "<built-in function getitem>",
            "operator.getitem",
        } or normalized.startswith(cls._TARGET_PREFIXES)

    def supports(self, signature: OpSignature) -> bool:
        return self._is_metadata_target(signature.target)

    def estimate(self, signature: OpSignature) -> OpCost | None:
        if not self.supports(signature):
            return None
        return OpCost(
            estimated_us=0.0,
            memory_bytes=0,
            source=self.source,
            confidence=1.0,
            hardware_version=current_hardware_version(),
            software_version=current_software_version(),
        )


class RooflineFallbackProvider:
    source = "roofline_fallback"
    cache_safe = True

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
    input_values = tuple(value_by_id[v] for v in op.input_value_ids if v in value_by_id)
    output_values = tuple(value_by_id[v] for v in op.output_value_ids if v in value_by_id)
    input_dtypes = tuple(value.dtype for value in input_values if value.dtype != "unknown")
    output_dtypes = tuple(value.dtype for value in output_values if value.dtype != "unknown")
    dtype = next(iter(output_dtypes or input_dtypes), "unknown")
    return OpSignature(
        op_name=op.name,
        target=op.target,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        dtype=dtype,
        input_shapes=op.input_shapes or tuple(value.shape for value in input_values),
        output_shapes=op.output_shapes or tuple(value.shape for value in output_values),
        input_dtypes=op.input_dtypes or tuple(value.dtype for value in input_values),
        output_dtypes=op.output_dtypes or tuple(value.dtype for value in output_values),
    )
