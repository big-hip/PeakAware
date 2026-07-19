from __future__ import annotations

import importlib
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable

from peakaware.errors import PatchRestoreError, UnsupportedTorchVersionError


_PATCH_LOCK = threading.Lock()


@dataclass(frozen=True)
class PatchSpec:
    owner: str
    target_module: str
    target_attribute: str
    wrapper: Callable[..., Any]
    torch_version_range: str = "*"
    expected_signature: str | None = None
    priority: int = 0
    failure_policy: str = "fail_closed"


def patch_method(
    *,
    owner: str,
    target_module: str,
    target_attribute: str,
    torch_version_range: str = "*",
    expected_signature: str | None = None,
    priority: int = 0,
    failure_policy: str = "fail_closed",
) -> Callable[[Callable[..., Any]], PatchSpec]:
    def decorate(wrapper: Callable[..., Any]) -> PatchSpec:
        return PatchSpec(
            owner=owner,
            target_module=target_module,
            target_attribute=target_attribute,
            wrapper=wrapper,
            torch_version_range=torch_version_range,
            expected_signature=expected_signature,
            priority=priority,
            failure_policy=failure_policy,
        )

    return decorate


def _validate_torch_version(spec: PatchSpec, torch_version: str) -> None:
    if spec.torch_version_range in {"*", torch_version}:
        return
    if torch_version.startswith(spec.torch_version_range.rstrip(".*")):
        return
    raise UnsupportedTorchVersionError(
        f"patch {spec.owner} expects torch {spec.torch_version_range}, got {torch_version}"
    )


def compose_wrapper_chain(original: Callable[..., Any], specs: tuple[PatchSpec, ...]) -> Callable[..., Any]:
    wrapped = original
    for spec in sorted(specs, key=lambda item: (item.priority, item.owner)):
        next_fn = wrapped

        def make_wrapper(current_spec: PatchSpec, current_next: Callable[..., Any]) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any) -> Any:
                return current_spec.wrapper(current_next, *args, **kwargs)

            return call

        wrapped = make_wrapper(spec, next_fn)
    return wrapped


def validate_target_signature(target: Callable[..., Any], expected_signature: str | None) -> None:
    if expected_signature is None:
        return
    actual = str(inspect.signature(target))
    if actual != expected_signature:
        raise PatchRestoreError(f"signature mismatch: expected {expected_signature}, got {actual}")


class PatchSession:
    def __init__(self, patch_specs: tuple[PatchSpec, ...], environment: dict[str, str] | None = None) -> None:
        self.patch_specs = patch_specs
        self.environment = environment or {}
        self._originals: list[tuple[Any, str, Any]] = []
        self._lock_acquired = False

    def __enter__(self) -> "PatchSession":
        import torch

        _PATCH_LOCK.acquire()
        self._lock_acquired = True
        try:
            grouped: dict[tuple[str, str], list[PatchSpec]] = {}
            for spec in self.patch_specs:
                _validate_torch_version(spec, torch.__version__)
                grouped.setdefault((spec.target_module, spec.target_attribute), []).append(spec)
            for (module_name, attr_name), specs in grouped.items():
                module = importlib.import_module(module_name)
                original = getattr(module, attr_name)
                for spec in specs:
                    validate_target_signature(original, spec.expected_signature)
                self._originals.append((module, attr_name, original))
                setattr(module, attr_name, compose_wrapper_chain(original, tuple(specs)))
            return self
        except BaseException:
            self._restore_originals()
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._restore_originals()

    def _restore_originals(self) -> None:
        try:
            for module, attr_name, original in reversed(self._originals):
                setattr(module, attr_name, original)
                if getattr(module, attr_name) is not original:
                    raise PatchRestoreError(f"failed to restore {module.__name__}.{attr_name}")
        finally:
            self._originals.clear()
            if self._lock_acquired:
                self._lock_acquired = False
                _PATCH_LOCK.release()
