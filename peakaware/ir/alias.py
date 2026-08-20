from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


_ALIAS_PRESERVING_TARGET_PREFIXES = (
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


def is_alias_preserving_target(target: object) -> bool:
    normalized = str(target).strip().lower()
    return normalized.startswith(_ALIAS_PRESERVING_TARGET_PREFIXES)


def storage_nbytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.numel() * value.element_size())
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            elem = torch.empty((), dtype=dtype).element_size()
            numel = 1
            for dim in tuple(shape):
                numel *= int(dim)
            return int(numel * elem)
        except Exception:
            return 0
    if isinstance(value, (tuple, list)):
        return sum(storage_nbytes(v) for v in value)
    return 0


def resolve_view_base(value: Any) -> Any:
    if isinstance(value, Tensor):
        base = getattr(value, "_base", None)
        return base if base is not None else value
    return value


def _storage_key(value: Tensor) -> tuple[str, int, int]:
    base = resolve_view_base(value)
    if isinstance(base, Tensor):
        if base.__class__.__name__ == "FakeTensor":
            return (f"fake:{base.device}", id(base), int(base.storage_offset()))
        try:
            ptr = int(base.untyped_storage().data_ptr())
        except Exception:
            ptr = id(base)
        return (str(base.device), ptr, int(base.storage_offset()))
    return ("object", id(value), 0)


def build_storage_groups(values: dict[int, Any], external_value_ids: frozenset[int]) -> dict[int, tuple[tuple[int, ...], int, bool]]:
    storage_key_to_id: dict[tuple[str, int, int], int] = {}
    storage_members: dict[int, list[int]] = {}
    storage_bytes: dict[int, int] = {}
    storage_external: dict[int, bool] = {}

    for value_id, value in values.items():
        if isinstance(value, Tensor):
            key = _storage_key(value)
        elif hasattr(value, "shape"):
            key = ("meta", value_id, 0)
        else:
            continue
        storage_id = storage_key_to_id.setdefault(key, len(storage_key_to_id))
        storage_members.setdefault(storage_id, []).append(value_id)
        storage_bytes[storage_id] = max(storage_bytes.get(storage_id, 0), storage_nbytes(resolve_view_base(value)))
        storage_external[storage_id] = storage_external.get(storage_id, False) or value_id in external_value_ids

    return {
        storage_id: (tuple(member_ids), storage_bytes[storage_id], storage_external[storage_id])
        for storage_id, member_ids in storage_members.items()
    }


def merge_alias_storage_groups(
    storage_groups: dict[int, tuple[tuple[int, ...], int, bool]],
    alias_edges: tuple[tuple[int, int], ...],
) -> dict[int, tuple[tuple[int, ...], int, bool]]:
    """Merge FakeTensor view outputs into their input physical storage.

    FakeTensor metadata does not always populate ``Tensor._base``. Pointer-based
    grouping can therefore assign view/transpose/detach outputs distinct storage
    ids even though the ATen operation is alias preserving. The input storage
    size is retained: expand does not allocate its logical output numel, while a
    slice keeps the full base storage alive.
    """

    groups: dict[int, list[Any]] = {
        storage_id: [list(value_ids), int(nbytes), bool(is_external)]
        for storage_id, (value_ids, nbytes, is_external) in storage_groups.items()
    }
    value_to_storage = {
        value_id: storage_id
        for storage_id, (value_ids, _nbytes, _is_external) in storage_groups.items()
        for value_id in value_ids
    }
    for output_value_id, input_value_id in alias_edges:
        output_storage_id = value_to_storage.get(output_value_id)
        input_storage_id = value_to_storage.get(input_value_id)
        if (
            output_storage_id is None
            or input_storage_id is None
            or output_storage_id == input_storage_id
            or output_storage_id not in groups
            or input_storage_id not in groups
        ):
            continue
        output_members, output_nbytes, output_external = groups.pop(output_storage_id)
        input_members, input_nbytes, input_external = groups[input_storage_id]
        input_members.extend(output_members)
        groups[input_storage_id] = [
            input_members,
            input_nbytes if input_nbytes > 0 else output_nbytes,
            input_external or output_external,
        ]
        for value_id in output_members:
            value_to_storage[value_id] = input_storage_id
    return {
        storage_id: (tuple(value_ids), int(nbytes), bool(is_external))
        for storage_id, (value_ids, nbytes, is_external) in groups.items()
    }


def tuple_output_storage_map(value: Any) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        return (storage_nbytes(value),)
    if isinstance(value, (tuple, list)):
        return tuple(storage_nbytes(v) for v in value)
    return ()


def validate_alias_invariants(storage_value_ids: dict[int, tuple[int, ...]]) -> tuple[str, ...]:
    seen: set[int] = set()
    errors: list[str] = []
    for storage_id, value_ids in storage_value_ids.items():
        for value_id in value_ids:
            if value_id in seen:
                errors.append(f"value {value_id} appears in more than one storage group")
            seen.add(value_id)
        if not value_ids:
            errors.append(f"storage {storage_id} has no values")
    return tuple(errors)
