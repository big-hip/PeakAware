from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from peakaware.errors import PluginConflictError


@dataclass(frozen=True)
class ServiceRecord:
    kind: str
    name: str
    service: Any
    priority: int
    order: int


@dataclass(frozen=True)
class HookRecord:
    event: str
    callback: Callable[..., Any]
    priority: int
    order: int


@dataclass(frozen=True)
class RegistrySnapshot:
    services: tuple[ServiceRecord, ...]
    hooks: tuple[HookRecord, ...]
    patch_specs: tuple[Any, ...]

    def resolve(self, kind: str, name: str | None = None) -> Any:
        matches = [record for record in self.services if record.kind == kind]
        if name is not None:
            matches = [record for record in matches if record.name == name]
        if not matches:
            return None
        matches.sort(key=lambda record: (-record.priority, record.order))
        return matches[0].service

    def services_for(self, kind: str) -> tuple[ServiceRecord, ...]:
        records = [record for record in self.services if record.kind == kind]
        records.sort(key=lambda record: (-record.priority, record.order))
        return tuple(records)

    def hooks_for(self, event: str) -> tuple[HookRecord, ...]:
        records = [record for record in self.hooks if record.event == event]
        records.sort(key=lambda record: (-record.priority, record.order))
        return tuple(records)


class PluginRegistry:
    def __init__(self) -> None:
        self._services: list[ServiceRecord] = []
        self._hooks: list[HookRecord] = []
        self._patch_specs: list[Any] = []
        self._frozen = False
        self._order = 0

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise PluginConflictError("registry is frozen")

    def register_service(self, kind: str, name: str, service: Any, *, priority: int = 0) -> None:
        self._ensure_mutable()
        if any(record.kind == kind and record.name == name for record in self._services):
            raise PluginConflictError(f"duplicate service: {kind}/{name}")
        self._services.append(ServiceRecord(kind, name, service, priority, self._order))
        self._order += 1

    def register_hook(self, event: str, callback: Callable[..., Any], *, priority: int = 0) -> None:
        self._ensure_mutable()
        self._hooks.append(HookRecord(event, callback, priority, self._order))
        self._order += 1

    def register_patch(self, spec: Any) -> None:
        self._ensure_mutable()
        target = (
            getattr(spec, "owner", None),
            getattr(spec, "target_module", None),
            getattr(spec, "target_attribute", None),
        )
        if any(
            (
                getattr(old, "owner", None),
                getattr(old, "target_module", None),
                getattr(old, "target_attribute", None),
            )
            == target
            for old in self._patch_specs
        ):
            raise PluginConflictError(f"duplicate patch target: {target}")
        self._patch_specs.append(spec)

    def validate_conflicts(self) -> None:
        seen: set[tuple[str, str]] = set()
        for record in self._services:
            key = (record.kind, record.name)
            if key in seen:
                raise PluginConflictError(f"duplicate service: {record.kind}/{record.name}")
            seen.add(key)

    def resolve(self, kind: str, name: str | None = None) -> Any:
        return self.freeze().resolve(kind, name)

    def freeze(self) -> RegistrySnapshot:
        self.validate_conflicts()
        self._frozen = True
        return RegistrySnapshot(tuple(self._services), tuple(self._hooks), tuple(self._patch_specs))
