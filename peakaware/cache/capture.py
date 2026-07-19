from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import fx

from peakaware.cache.store import CacheEntry, load_cache_entry, store_cache_entry
from peakaware.contracts import CapturedJointGraph


@dataclass(frozen=True)
class _CachedCaptureArtifact:
    capture: CapturedJointGraph
    joint_tensor_meta: dict[str, Any]
    fw_tensor_meta: dict[str, Any]
    bw_tensor_meta: dict[str, Any]


def _collect_tensor_meta(gm: fx.GraphModule | None) -> dict[str, Any]:
    if gm is None:
        return {}
    return {
        node.name: node.meta["tensor_meta"]
        for node in gm.graph.nodes
        if "tensor_meta" in node.meta
    }


def _restore_tensor_meta(gm: fx.GraphModule | None, tensor_meta_by_name: dict[str, Any]) -> None:
    if gm is None:
        return
    for node in gm.graph.nodes:
        if node.name in tensor_meta_by_name:
            node.meta["tensor_meta"] = tensor_meta_by_name[node.name]


def _wrap_capture(capture: CapturedJointGraph) -> _CachedCaptureArtifact:
    return _CachedCaptureArtifact(
        capture=capture,
        joint_tensor_meta=_collect_tensor_meta(capture.joint_module),
        fw_tensor_meta=_collect_tensor_meta(capture.fw_module),
        bw_tensor_meta=_collect_tensor_meta(capture.bw_module),
    )


def _unwrap_capture(artifact: Any) -> CapturedJointGraph | None:
    if isinstance(artifact, CapturedJointGraph):
        return artifact
    if not isinstance(artifact, _CachedCaptureArtifact):
        return None
    capture = artifact.capture
    _restore_tensor_meta(capture.joint_module, artifact.joint_tensor_meta)
    _restore_tensor_meta(capture.fw_module, artifact.fw_tensor_meta)
    _restore_tensor_meta(capture.bw_module, artifact.bw_tensor_meta)
    return capture


def store_capture_cache(root: str | Path, key: str, capture: CapturedJointGraph, provenance: dict[str, Any]) -> None:
    store_cache_entry(root, "capture", CacheEntry(key, _wrap_capture(capture), provenance))


def load_capture_cache(root: str | Path, key: str, expected_provenance: dict[str, Any] | None = None) -> CapturedJointGraph | None:
    entry = load_cache_entry(root, "capture", key, expected_provenance)
    if entry is None:
        return None
    return _unwrap_capture(entry.artifact)
