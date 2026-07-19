from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    key: str
    artifact: Any
    provenance: dict[str, Any]


def _layer_dir(root: str | Path, layer: str) -> Path:
    path = Path(root) / layer
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_path(root: str | Path, layer: str, key: str) -> Path:
    return _layer_dir(root, layer) / f"{key}.pkl"


def _provenance_path(root: str | Path, layer: str, key: str) -> Path:
    return _layer_dir(root, layer) / f"{key}.json"


def store_cache_entry(root: str | Path, layer: str, entry: CacheEntry) -> None:
    artifact_path = _artifact_path(root, layer, entry.key)
    provenance_path = _provenance_path(root, layer, entry.key)
    artifact_tmp = artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")
    provenance_tmp = provenance_path.with_suffix(f"{provenance_path.suffix}.tmp")
    try:
        with artifact_tmp.open("wb") as handle:
            pickle.dump(entry.artifact, handle)
        provenance_tmp.write_text(
            json.dumps(entry.provenance, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        artifact_tmp.replace(artifact_path)
        provenance_tmp.replace(provenance_path)
    except Exception:
        artifact_tmp.unlink(missing_ok=True)
        provenance_tmp.unlink(missing_ok=True)
        raise


def load_cache_entry(root: str | Path, layer: str, key: str, expected_provenance: dict[str, Any] | None = None) -> CacheEntry | None:
    artifact_path = _artifact_path(root, layer, key)
    provenance_path = _provenance_path(root, layer, key)
    if not artifact_path.exists() or not provenance_path.exists():
        return None
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if expected_provenance is not None and not validate_cache_provenance(provenance, expected_provenance):
        return None
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    return CacheEntry(key=key, artifact=artifact, provenance=provenance)


def validate_cache_provenance(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            return False
    return True


def invalidate_downstream(root: str | Path, changed_layer: str) -> tuple[str, ...]:
    order = ("capture", "analysis", "executable")
    if changed_layer not in order:
        raise ValueError(f"unknown cache layer: {changed_layer}")
    changed_index = order.index(changed_layer)
    removed: list[str] = []
    for layer in order[changed_index + 1 :]:
        layer_path = Path(root) / layer
        if not layer_path.exists():
            continue
        for path in layer_path.glob("*"):
            if path.is_file():
                removed.append(str(path))
                path.unlink()
    return tuple(removed)
