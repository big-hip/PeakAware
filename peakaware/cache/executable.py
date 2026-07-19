from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from peakaware.cache.store import CacheEntry, load_cache_entry, store_cache_entry
from peakaware.contracts import MeasuredExecutable


def _cached_forward_backward_unavailable(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("cached executable measurement must be rebound to a live executor before use")


def store_executable_cache(root: str | Path, key: str, executable: MeasuredExecutable, provenance: dict[str, Any]) -> None:
    store_cache_entry(root, "executable", CacheEntry(key, executable, provenance))


def load_executable_cache(root: str | Path, key: str, expected_provenance: dict[str, Any] | None = None) -> MeasuredExecutable | None:
    entry = load_cache_entry(root, "executable", key, expected_provenance)
    if entry is None:
        return None
    if not isinstance(entry.artifact, MeasuredExecutable):
        return None
    return entry.artifact


def store_executable_measurement_cache(
    root: str | Path,
    key: str,
    executable: MeasuredExecutable,
    provenance: dict[str, Any],
) -> None:
    serializable = replace(executable, forward_backward=_cached_forward_backward_unavailable)
    store_executable_cache(root, key, serializable, provenance)


def load_executable_measurement_cache(
    root: str | Path,
    key: str,
    forward_backward: Any,
    expected_provenance: dict[str, Any] | None = None,
) -> MeasuredExecutable | None:
    cached = load_executable_cache(root, key, expected_provenance)
    if cached is None:
        return None
    return replace(cached, forward_backward=forward_backward)


def select_cached_executable(
    executables: tuple[MeasuredExecutable, ...],
    *,
    memory_budget_bytes: int,
) -> MeasuredExecutable | None:
    feasible = [
        executable
        for executable in executables
        if executable.correctness_passed and executable.measured_peak_bytes <= memory_budget_bytes
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda executable: (executable.measured_step_us, executable.measured_peak_bytes, executable.plan_id))
    return feasible[0]
