from __future__ import annotations

from typing import Any

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

from peakaware.contracts import CapturedJointGraph, GuardSpec, ParameterBinding, TrainingRequest
from peakaware.errors import CaptureError

from .fake_inputs import assert_no_large_real_storage
from .graph_key import build_graph_key


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _collect_parameter_mapping(request: TrainingRequest) -> tuple[ParameterBinding, ...]:
    bindings: list[ParameterBinding] = []
    for index, (name, param) in enumerate(request.model.named_parameters()):
        bindings.append(
            ParameterBinding(
                name=name,
                index=index,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                device=str(param.device),
                requires_grad=param.requires_grad,
                nbytes=_tensor_nbytes(param),
            )
        )
    return tuple(bindings)


def _collect_guards(request: TrainingRequest) -> tuple[GuardSpec, ...]:
    guards: list[GuardSpec] = []
    for index, arg in enumerate(request.example_args):
        if isinstance(arg, torch.Tensor):
            guards.append(GuardSpec(f"arg{index}.shape", str(tuple(arg.shape))))
            guards.append(GuardSpec(f"arg{index}.dtype", str(arg.dtype)))
            guards.append(GuardSpec(f"arg{index}.device", str(arg.device)))
    for name, arg in sorted(request.example_kwargs.items()):
        if isinstance(arg, torch.Tensor):
            guards.append(GuardSpec(f"kw.{name}.shape", str(tuple(arg.shape))))
            guards.append(GuardSpec(f"kw.{name}.dtype", str(arg.dtype)))
            guards.append(GuardSpec(f"kw.{name}.device", str(arg.device)))
    guards.append(GuardSpec("torch_version", torch.__version__))
    return tuple(guards)


def _capture_with_fake_tensor(request: TrainingRequest) -> fx.GraphModule:
    del request
    raise CaptureError("M0 public implementation uses symbolic FX capture; fake AOT capture is an M1/M2 adapter")


def _detect_graph_breaks(gm: fx.GraphModule) -> tuple[str, ...]:
    breaks: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and "break" in str(node.target).lower():
            breaks.append(node.name)
    return tuple(breaks)


def capture_joint_graph(request: TrainingRequest, adapter: Any | None = None) -> CapturedJointGraph:
    if adapter is not None:
        return adapter.capture_joint_graph(request)

    assert_no_large_real_storage(request.example_args, request.example_kwargs)
    try:
        gm = fx.symbolic_trace(request.model)
        if request.example_kwargs:
            ShapeProp(gm).propagate(*request.example_args, **request.example_kwargs)
        else:
            ShapeProp(gm).propagate(*request.example_args)
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        raise CaptureError(f"failed to capture model with torch.fx: {exc}") from exc

    breaks = _detect_graph_breaks(gm)
    if breaks:
        raise CaptureError(f"graph breaks are unsupported in M0: {breaks}")

    capture_key = build_graph_key(gm, request.model)
    return CapturedJointGraph(
        joint_module=gm,
        guards=_collect_guards(request),
        parameter_mapping=_collect_parameter_mapping(request),
        capture_key=capture_key,
    )
