from __future__ import annotations

from pathlib import Path
from typing import Any

from peakaware.cache.store import CacheEntry, load_cache_entry, store_cache_entry
from peakaware.contracts import AnalysisBundle


def store_analysis_cache(root: str | Path, key: str, analysis: AnalysisBundle, provenance: dict[str, Any]) -> None:
    store_cache_entry(root, "analysis", CacheEntry(key, analysis, provenance))


def load_analysis_cache(root: str | Path, key: str, expected_provenance: dict[str, Any] | None = None) -> AnalysisBundle | None:
    entry = load_cache_entry(root, "analysis", key, expected_provenance)
    if entry is None:
        return None
    if not isinstance(entry.artifact, AnalysisBundle):
        return None
    return entry.artifact
