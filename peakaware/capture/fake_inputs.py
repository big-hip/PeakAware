from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def fakeify_module_state(model: nn.Module) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
    return tuple(
        (name, tuple(t.shape), str(t.dtype), str(t.device))
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
    )


def fakeify_tree(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return {"shape": tuple(obj.shape), "dtype": str(obj.dtype), "device": str(obj.device)}
    if isinstance(obj, tuple):
        return tuple(fakeify_tree(x) for x in obj)
    if isinstance(obj, list):
        return [fakeify_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {k: fakeify_tree(v) for k, v in obj.items()}
    return obj


def assert_no_large_real_storage(args: tuple[Any, ...], kwargs: dict[str, Any], max_bytes: int = 512 << 20) -> None:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, Tensor) and value.numel() * value.element_size() > max_bytes:
            raise ValueError("real input tensor exceeds M0 capture safety limit")


def infer_logical_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.device:
    devices = {
        value.device
        for value in list(args) + list(kwargs.values())
        if isinstance(value, Tensor)
    }
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise ValueError(f"mixed input devices are not supported in M0: {sorted(map(str, devices))}")
    return next(iter(devices))
