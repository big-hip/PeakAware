from __future__ import annotations

import copy
from typing import Any

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

from peakaware.contracts import CapturedJointGraph, FailureRecord, GuardSpec, ParameterBinding, TrainingRequest
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


def _clone_example(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, tuple):
        return tuple(_clone_example(item) for item in value)
    if isinstance(value, list):
        return [_clone_example(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_example(item) for key, item in value.items()}
    return value


def _capture_with_fx(request: TrainingRequest, failures: tuple[FailureRecord, ...] = ()) -> CapturedJointGraph:
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

    guards = _collect_guards(request)
    capture_key = build_graph_key(gm, request.model, guards)
    return CapturedJointGraph(
        joint_module=gm,
        guards=guards,
        parameter_mapping=_collect_parameter_mapping(request),
        capture_key=capture_key,
        backend="fx",
        failures=failures,
    )


def _capture_with_aot_autograd(request: TrainingRequest) -> CapturedJointGraph:
    from torch._functorch.aot_autograd import aot_module
    from torch._functorch.partitioners import default_partition
    from functorch.compile import make_boxed_func

    captured: dict[str, fx.GraphModule] = {}
    partition_meta: dict[str, Any] = {}
    model_copy = copy.deepcopy(request.model)
    model_copy.train(request.model.training)
    args = tuple(_clone_example(arg) for arg in request.example_args)
    kwargs = {key: _clone_example(value) for key, value in request.example_kwargs.items()}

    def fw_compiler(gm: fx.GraphModule, _example_inputs: Any) -> Any:
        captured["fw"] = gm
        return make_boxed_func(gm.forward)

    def bw_compiler(gm: fx.GraphModule, _example_inputs: Any) -> Any:
        captured["bw"] = gm
        return make_boxed_func(gm.forward)

    def partition_fn(joint_module: fx.GraphModule, joint_inputs: Any, **kwargs_for_partition: Any) -> Any:
        captured["joint"] = joint_module
        partition_meta.update(kwargs_for_partition)
        return default_partition(joint_module, joint_inputs, **kwargs_for_partition)

    rng_state = torch.get_rng_state()
    try:
        compiled = aot_module(
            model_copy,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        output = compiled(*args, **kwargs)
        loss = request.loss_fn(output)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise CaptureError("loss_fn must return a scalar tensor for AOT capture")
        loss.backward()
    finally:
        torch.set_rng_state(rng_state)

    joint = captured.get("joint")
    if joint is None:
        raise CaptureError("AOTAutograd did not invoke partition_fn")
    guards = _collect_guards(request)
    capture_key = build_graph_key(joint, request.model, guards)
    return CapturedJointGraph(
        joint_module=joint,
        guards=guards,
        parameter_mapping=_collect_parameter_mapping(request),
        capture_key=capture_key,
        fw_module=captured.get("fw"),
        bw_module=captured.get("bw"),
        backend="aot",
        num_fwd_outputs=int(partition_meta.get("num_fwd_outputs", 1)),
        static_lifetime_input_indices=tuple(partition_meta.get("static_lifetime_input_indices") or ()),
    )


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
    backend = getattr(request.config, "capture_backend", "auto")
    if backend in {"auto", "aot"}:
        try:
            return _capture_with_aot_autograd(request)
        except Exception as exc:
            if backend == "aot":
                raise CaptureError(f"failed to capture AOTAutograd joint graph: {exc}") from exc
            failure = FailureRecord(
                stage="capture_aot",
                error_type=type(exc).__name__,
                message=str(exc),
                recovered=True,
                next_fallback="fx",
                applied_adapters=("aot_autograd", "default_partition"),
                applied_plugins=(),
            )
            return _capture_with_fx(request, failures=(failure,))
    return _capture_with_fx(request)
