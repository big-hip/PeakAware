from __future__ import annotations

import hashlib

import torch
from torch import fx, nn

from peakaware.contracts import GuardSpec


def hash_graph_module(gm: fx.GraphModule) -> str:
    return hashlib.sha256(gm.code.encode("utf-8")).hexdigest()[:16]


def hash_state_signature(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tensor.device).encode("utf-8"))
        h.update(str(tensor.requires_grad).encode("utf-8"))
    return h.hexdigest()[:16]


def hash_guard_signature(guards: tuple[GuardSpec, ...]) -> str:
    h = hashlib.sha256()
    for guard in guards:
        h.update(guard.name.encode("utf-8"))
        h.update(b"\0")
        h.update(guard.value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_graph_key(gm: fx.GraphModule, model: nn.Module, guards: tuple[GuardSpec, ...] = ()) -> str:
    return (
        f"{hash_graph_module(gm)}-{hash_state_signature(model)}-"
        f"{hash_guard_signature(guards)}-torch{torch.__version__}"
    )
