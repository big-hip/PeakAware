from __future__ import annotations

from pathlib import Path
from typing import Any

from peakaware.cache.store import CacheEntry, load_cache_entry, store_cache_entry
from peakaware.contracts import CapturedJointGraph


def store_capture_cache(root: str | Path, key: str, capture: CapturedJointGraph, provenance: dict[str, Any]) -> None:
    store_cache_entry(root, "capture", CacheEntry(key, capture, provenance))


def load_capture_cache(root: str | Path, key: str, expected_provenance: dict[str, Any] | None = None) -> CapturedJointGraph | None:
    entry = load_cache_entry(root, "capture", key, expected_provenance)
    if entry is None:
        return None
    if not isinstance(entry.artifact, CapturedJointGraph):
        return None
    return entry.artifact
