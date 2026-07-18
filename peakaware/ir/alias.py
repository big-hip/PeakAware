from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


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
