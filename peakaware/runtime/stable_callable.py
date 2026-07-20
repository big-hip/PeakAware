from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable


class _RegistryToken:
    __slots__ = ()


@dataclass(frozen=True)
class StableCallableSnapshot:
    object_id: int
    token: _RegistryToken
    reference: weakref.ReferenceType[Any]


@dataclass(frozen=True)
class _RegistryEntry:
    reference: weakref.ReferenceType[Any]
    token: _RegistryToken


_LOCK = threading.Lock()
_ENTRIES: dict[int, _RegistryEntry] = {}


def register_stable_callable(value: Callable[..., Any]) -> object:
    if not callable(value):
        raise TypeError("stable callable registry accepts callable objects only")
    object_id = id(value)
    token = _RegistryToken()

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        with _LOCK:
            entry = _ENTRIES.get(object_id)
            if entry is not None and entry.reference is reference:
                del _ENTRIES[object_id]

    reference = weakref.ref(value, remove)
    with _LOCK:
        entry = _ENTRIES.get(object_id)
        if entry is not None and entry.reference() is value:
            raise RuntimeError("callable already has an active stable registration")
        _ENTRIES[object_id] = _RegistryEntry(reference, token)
    return token


def unregister_stable_callable(value: Callable[..., Any], token: object) -> None:
    with _LOCK:
        entry = _ENTRIES.get(id(value))
        if entry is None or entry.reference() is not value or entry.token is not token:
            raise RuntimeError("stable callable registration ownership mismatch")
        del _ENTRIES[id(value)]


def snapshot_registered_callable(value: Any) -> StableCallableSnapshot | None:
    if not callable(value):
        return None
    with _LOCK:
        entry = _ENTRIES.get(id(value))
        if entry is None or entry.reference() is not value:
            return None
        return StableCallableSnapshot(id(value), entry.token, entry.reference)


def validate_registered_callable(value: Any, snapshot: StableCallableSnapshot) -> None:
    with _LOCK:
        entry = _ENTRIES.get(snapshot.object_id)
        if (
            id(value) != snapshot.object_id
            or snapshot.reference() is not value
            or entry is None
            or entry.reference() is not value
            or entry.token is not snapshot.token
        ):
            raise ValueError("registered stable callable identity or ownership token changed")
