from __future__ import annotations

import copy
import enum
import functools
import hashlib
import json
import re
import threading
import types
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
from torch import fx, nn
from torch.utils import _pytree
from torch.utils import checkpoint

from peakaware.contracts import CapturedJointGraph, LoweredPartition, PartitionABI
from peakaware.runtime.executor import build_aot_partition_executable


class UnsupportedMethodError(RuntimeError):
    """Raised when a publication method cannot preserve its declared semantics."""


class StableSerializationError(ValueError):
    """Raised when behavior-affecting state cannot be serialized without process identity."""


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    is_real: bool
    api: str
    policy: str | None
    region_family: tuple[str, ...]
    compiler_protocol: str


@dataclass(frozen=True)
class RuntimeIdentity:
    method_id: str
    status: str
    is_real: bool
    api: str
    policy: str | None
    region_paths: tuple[str, ...]
    compiler_protocol: str
    fallback_reason: str | None = None
    fw_graph_sha256: str | None = None
    bw_graph_sha256: str | None = None
    model_sha256: str | None = None
    executable_sha256: str | None = None
    fw_residual_names: tuple[str, ...] = ()
    bw_placeholder_names: tuple[str, ...] = ()
    provenance: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = dict(self.provenance)
        return json.loads(json.dumps(payload, sort_keys=True, default=str))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class PreparedMethod:
    spec: MethodSpec
    identity: RuntimeIdentity
    executable: Callable[..., Any] | nn.Module | None = field(default=None, repr=False, compare=False)
    fw_graph: fx.GraphModule | None = field(default=None, repr=False, compare=False)
    bw_graph: fx.GraphModule | None = field(default=None, repr=False, compare=False)

    @property
    def supported(self) -> bool:
        return self.identity.status == "ready"

    def require_supported(self) -> PreparedMethod:
        if not self.supported or self.executable is None:
            reason = self.identity.fallback_reason or "method was not prepared"
            raise UnsupportedMethodError(f"{self.spec.method_id} is unsupported: {reason}")
        return self


def _unsupported(
    spec: MethodSpec,
    reason: str,
    *,
    regions: tuple[str, ...] = (),
    fw_graph: fx.GraphModule | None = None,
    bw_graph: fx.GraphModule | None = None,
    model_sha256: str | None = None,
    executable_sha256: str | None = None,
) -> PreparedMethod:
    return PreparedMethod(
        spec=spec,
        identity=RuntimeIdentity(
            method_id=spec.method_id,
            status="unsupported",
            is_real=spec.is_real,
            api=spec.api,
            policy=spec.policy,
            region_paths=regions,
            compiler_protocol=spec.compiler_protocol,
            fallback_reason=reason,
            fw_graph_sha256=None if fw_graph is None else _graph_sha256(fw_graph),
            bw_graph_sha256=None if bw_graph is None else _graph_sha256(bw_graph),
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        ),
        fw_graph=fw_graph,
        bw_graph=bw_graph,
    )


def _node_ref(value: Any) -> Any:
    if isinstance(value, fx.Node):
        return {"node": value.name}
    if isinstance(value, tuple):
        return [_node_ref(item) for item in value]
    if isinstance(value, list):
        return [_node_ref(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _node_ref(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, torch.dtype):
        return str(value)
    text = str(value)
    if re.search(r"0x[0-9a-fA-F]+", text):
        raise StableSerializationError("FX graph value contains an object address")
    return {
        "value_class": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": text,
    }


def _graph_sha256(graph_module: fx.GraphModule) -> str:
    nodes = [
        {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": _node_ref(node.args),
            "kwargs": _node_ref(node.kwargs),
        }
        for node in graph_module.graph.nodes
    ]
    encoded = json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_code_constant(value: Any) -> Any:
    if value is None or value is Ellipsis or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_stable_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_stable_code_constant(item) for item in value]
        return {"frozenset": sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))}
    if isinstance(value, types.CodeType):
        return {
            "code": value.co_code.hex(),
            "consts": [_stable_code_constant(item) for item in value.co_consts],
            "names": value.co_names,
        }
    raise StableSerializationError(
        "callable code contains unsupported constant "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _callable_payload(value: Any, stack: set[int]) -> dict[str, Any]:
    if isinstance(value, functools.partial):
        return {
            "kind": "partial",
            "function": _callable_payload(value.func, stack),
            "args": _stable_serialize(value.args, stack),
            "keywords": _stable_serialize(value.keywords or {}, stack),
        }
    callable_object = getattr(value, "__func__", value)
    code = getattr(callable_object, "__code__", None)
    module_name = getattr(callable_object, "__module__", None)
    qualified_name = getattr(callable_object, "__qualname__", None)
    if code is None:
        if module_name is None or qualified_name is None:
            raise StableSerializationError(
                "callable has no stable code or qualified name: "
                f"{type(callable_object).__module__}.{type(callable_object).__qualname__}"
            )
        return {"kind": "qualified_callable", "module": module_name, "qualname": qualified_name}
    closure = getattr(callable_object, "__closure__", None) or ()
    try:
        closure_values = tuple(cell.cell_contents for cell in closure)
    except ValueError as exc:
        raise StableSerializationError("callable contains an empty closure cell") from exc
    return {
        "kind": "python_callable",
        "module": module_name,
        "qualname": qualified_name,
        "code": {
            "bytes": code.co_code.hex(),
            "consts": [_stable_code_constant(item) for item in code.co_consts],
            "names": code.co_names,
        },
        "defaults": _stable_serialize(getattr(callable_object, "__defaults__", None), stack),
        "kwdefaults": _stable_serialize(getattr(callable_object, "__kwdefaults__", None), stack),
        "closure": _stable_serialize(closure_values, stack),
    }


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    if tensor.device.type == "meta":
        raise StableSerializationError("meta tensor has no stable content")
    value = tensor.detach()
    if value.is_sparse:
        value = value.to_dense()
    if value.is_quantized:
        value = value.int_repr()
    value = value.cpu().contiguous()
    try:
        raw = value.view(torch.uint8).numpy().tobytes()
    except Exception as exc:
        raise StableSerializationError(f"tensor content cannot be serialized: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _stable_serialize(value: Any, stack: set[int] | None = None) -> Any:
    stack = set() if stack is None else stack
    if isinstance(value, enum.Enum):
        return {
            "enum_class": f"{type(value).__module__}.{type(value).__qualname__}",
            "enum_name": value.name,
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex": (value.real.hex(), value.imag.hex())}
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": tuple(value.shape),
            "tensor_dtype": str(value.dtype),
            "tensor_device": str(value.device),
            "tensor_content_sha256": _tensor_content_sha256(value),
        }
    tracked = isinstance(value, (tuple, list, dict)) or callable(value)
    value_id = id(value)
    if tracked:
        if value_id in stack:
            raise StableSerializationError("cyclic behavior attribute is unsupported")
        stack.add(value_id)
    try:
        if isinstance(value, tuple):
            return {"tuple": [_stable_serialize(item, stack) for item in value]}
        if isinstance(value, list):
            return {"list": [_stable_serialize(item, stack) for item in value]}
        if isinstance(value, dict):
            entries = [
                (_stable_serialize(key, stack), _stable_serialize(item, stack))
                for key, item in value.items()
            ]
            entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True, default=str))
            return {"dict": entries}
        if callable(value):
            return {"callable": _callable_payload(value, stack)}
    finally:
        if tracked:
            stack.remove(value_id)
    raise StableSerializationError(
        "unsupported behavior attribute type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _forward_callable_sha256(module: nn.Module) -> str:
    return _canonical_sha256(_callable_payload(module.forward, set()))


_PYTORCH_MODULE_INTERNAL_ATTRS = {
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_compiled_call_impl",
}


def _module_behavior_attributes(module: nn.Module) -> dict[str, Any]:
    return {
        name: _stable_serialize(value)
        for name, value in sorted(module.__dict__.items())
        if name not in _PYTORCH_MODULE_INTERNAL_ATTRS
    }


def _stable_extra_repr(module: nn.Module) -> str:
    value = module.extra_repr()
    if not isinstance(value, str):
        raise StableSerializationError("module extra_repr() did not return a string")
    if re.search(r"0x[0-9a-fA-F]+", value):
        raise StableSerializationError("module extra_repr() contains an object address")
    return value


def _model_sha256(model: nn.Module, region_paths: tuple[str, ...]) -> str:
    try:
        payload = {
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "modules": [
                {
                    "name": name,
                    "class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "extra_repr": _stable_extra_repr(module),
                    "training": module.training,
                    "forward_callable_sha256": _forward_callable_sha256(module),
                    "behavior_attributes": _module_behavior_attributes(module),
                }
                for name, module in model.named_modules()
            ],
            "parameters": [
                (name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad)
                for name, parameter in model.named_parameters()
            ],
            "buffers": [
                (name, tuple(buffer.shape), str(buffer.dtype))
                for name, buffer in model.named_buffers()
            ],
            "region_paths": region_paths,
        }
        return _canonical_sha256(payload)
    except StableSerializationError:
        raise
    except Exception as exc:
        raise StableSerializationError(
            "model behavior metadata serialization failed: "
            f"{type(exc).__module__}.{type(exc).__qualname__}"
        ) from exc


def _executable_sha256(
    spec: MethodSpec,
    model_sha256: str,
    *,
    region_paths: tuple[str, ...],
    fw_graph_sha256: str | None = None,
    bw_graph_sha256: str | None = None,
) -> str:
    return _canonical_sha256(
        {
            "method": asdict(spec),
            "model_sha256": model_sha256,
            "region_paths": region_paths,
            "fw_graph_sha256": fw_graph_sha256,
            "bw_graph_sha256": bw_graph_sha256,
        }
    )


def _flatten_nodes(value: Any) -> tuple[fx.Node, ...]:
    if isinstance(value, fx.Node):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(node for item in value for node in _flatten_nodes(item))
    if isinstance(value, dict):
        return tuple(node for item in value.values() for node in _flatten_nodes(item))
    return ()


def _forward_output_names(graph_module: fx.GraphModule) -> tuple[str, ...]:
    output = next((node for node in graph_module.graph.nodes if node.op == "output"), None)
    if output is None or not output.args:
        return ()
    return tuple(node.name for node in _flatten_nodes(output.args[0]))


def _placeholder_names(graph_module: fx.GraphModule) -> tuple[str, ...]:
    return tuple(node.name for node in graph_module.graph.nodes if node.op == "placeholder")


def _clone_grads(model: nn.Module) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if param.grad is None else param.grad.detach().clone() for param in model.parameters())


def _restore_grads(model: nn.Module, grads: tuple[torch.Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


def _clone_output(value: Any) -> Any:
    return _pytree.tree_map(
        lambda leaf: leaf.detach().clone() if isinstance(leaf, torch.Tensor) else copy.deepcopy(leaf),
        value,
    )


def _outputs_match(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    actual_flat, actual_spec = _pytree.tree_flatten(actual)
    expected_flat, expected_spec = _pytree.tree_flatten(expected)
    if actual_spec != expected_spec or len(actual_flat) != len(expected_flat):
        return False
    for actual_leaf, expected_leaf in zip(actual_flat, expected_flat):
        if isinstance(actual_leaf, torch.Tensor) and isinstance(expected_leaf, torch.Tensor):
            if actual_leaf.shape != expected_leaf.shape or actual_leaf.dtype != expected_leaf.dtype:
                return False
            if not torch.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol, equal_nan=True):
                return False
        elif type(actual_leaf) is not type(expected_leaf) or actual_leaf != expected_leaf:
            return False
    return True


def _cuda_rng_state() -> list[torch.Tensor] | None:
    if not torch.cuda.is_available():
        return None
    return [state.clone() for state in torch.cuda.get_rng_state_all()]


def _restore_rng(cpu_rng: torch.Tensor, cuda_rng: list[torch.Tensor] | None) -> None:
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


def _rng_equal(actual: list[torch.Tensor] | None, expected: list[torch.Tensor] | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return len(actual) == len(expected) and all(
        torch.equal(left, right) for left, right in zip(actual, expected)
    )


def _clone_buffers(model: nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple((name, buffer.detach().clone()) for name, buffer in model.named_buffers())


def _buffers_equal(
    actual: tuple[tuple[str, torch.Tensor], ...],
    expected: tuple[tuple[str, torch.Tensor], ...],
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        actual_name == expected_name
        and actual_buffer.shape == expected_buffer.shape
        and actual_buffer.dtype == expected_buffer.dtype
        and torch.equal(actual_buffer, expected_buffer)
        for (actual_name, actual_buffer), (expected_name, expected_buffer) in zip(actual, expected)
    )


def _training_flags(model: nn.Module) -> tuple[tuple[str, bool], ...]:
    return tuple((name, module.training) for name, module in model.named_modules())


def _validate_capture_model(capture: CapturedJointGraph, model: nn.Module) -> str | None:
    parameters = tuple(model.named_parameters())
    if len(parameters) != len(capture.parameter_mapping):
        return "model parameter count does not match captured parameter mapping"
    for (name, parameter), binding in zip(parameters, capture.parameter_mapping):
        if name != binding.name:
            return f"model parameter {name!r} does not match captured parameter {binding.name!r}"
        if tuple(parameter.shape) != binding.shape or str(parameter.dtype) != binding.dtype:
            return f"model parameter {name!r} shape or dtype differs from capture"
    return None


@dataclass(frozen=True)
class _QualificationResult:
    passed: bool
    failure_reason: str | None
    buffers_match: bool
    cpu_rng_match: bool
    cuda_rng_match: bool
    training_flags_match: bool = False


def _qualify_training_executable(
    executable: Callable[..., Any],
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> _QualificationResult:
    original_state = copy.deepcopy(model.state_dict())
    original_grads = _clone_grads(model)
    original_training_flags = tuple((module, module.training) for module in model.modules())
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_state()

    def restore() -> None:
        model.load_state_dict(original_state)
        _restore_grads(model, original_grads)
        for module, training in original_training_flags:
            module.training = training
        _restore_rng(cpu_rng, cuda_rng)

    try:
        restore()
        model.zero_grad(set_to_none=True)
        eager_output = model(*example_args, **example_kwargs)
        eager_loss = loss_fn(eager_output)
        if not isinstance(eager_loss, torch.Tensor) or eager_loss.ndim != 0:
            return _QualificationResult(False, "loss_fn must return a scalar tensor", False, False, False)
        eager_loss.backward()
        expected_output = _clone_output(eager_output)
        expected_loss = eager_loss.detach().clone()
        expected_grads = _clone_grads(model)
        expected_buffers = _clone_buffers(model)
        expected_cpu_rng = torch.get_rng_state().clone()
        expected_cuda_rng = _cuda_rng_state()
        expected_training_flags = _training_flags(model)

        restore()
        model.zero_grad(set_to_none=True)
        actual_output = executable(*example_args, **example_kwargs)
        actual_loss = loss_fn(actual_output)
        if not isinstance(actual_loss, torch.Tensor) or actual_loss.ndim != 0:
            return _QualificationResult(
                False,
                "prepared executable loss_fn result is not a scalar tensor",
                False,
                False,
                False,
            )
        actual_loss.backward()
        actual_grads = _clone_grads(model)
        actual_buffers = _clone_buffers(model)
        actual_cpu_rng = torch.get_rng_state().clone()
        actual_cuda_rng = _cuda_rng_state()
        actual_training_flags = _training_flags(model)

        if not _outputs_match(actual_output, expected_output, atol=atol, rtol=rtol):
            return _QualificationResult(False, "prepared executable outputs do not match eager", False, False, False)
        if not torch.allclose(actual_loss.detach(), expected_loss, atol=atol, rtol=rtol, equal_nan=True):
            return _QualificationResult(False, "prepared executable loss does not match eager", False, False, False)
        if len(actual_grads) != len(expected_grads):
            return _QualificationResult(
                False,
                "prepared executable gradient count does not match eager",
                False,
                False,
                False,
            )
        for (name, _), actual, expected in zip(model.named_parameters(), actual_grads, expected_grads):
            if actual is None and expected is None:
                continue
            if actual is None or expected is None:
                return _QualificationResult(
                    False,
                    f"prepared executable gradient presence differs for parameter {name!r}",
                    False,
                    False,
                    False,
                )
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
                return _QualificationResult(
                    False,
                    f"prepared executable gradient differs for parameter {name!r}",
                    False,
                    False,
                    False,
                )
        buffers_match = _buffers_equal(actual_buffers, expected_buffers)
        cpu_rng_match = torch.equal(actual_cpu_rng, expected_cpu_rng)
        cuda_rng_match = _rng_equal(actual_cuda_rng, expected_cuda_rng)
        training_flags_match = actual_training_flags == expected_training_flags
        if not buffers_match:
            return _QualificationResult(
                False,
                "prepared executable buffers do not match eager",
                False,
                cpu_rng_match,
                cuda_rng_match,
            )
        if not cpu_rng_match:
            return _QualificationResult(
                False,
                "prepared executable CPU RNG does not match eager",
                True,
                False,
                cuda_rng_match,
            )
        if not cuda_rng_match:
            return _QualificationResult(False, "prepared executable CUDA RNG does not match eager", True, True, False)
        if not training_flags_match:
            return _QualificationResult(
                False,
                "prepared executable module training flags do not match eager",
                True,
                True,
                True,
                False,
            )
        return _QualificationResult(True, None, True, True, True, True)
    except Exception as exc:
        return _QualificationResult(
            False,
            f"training executable qualification failed: {type(exc).__name__}: {exc}",
            False,
            False,
            False,
        )
    finally:
        restore()


def prepare_all_save(
    model: nn.Module,
    *,
    execution_backend: str = "unwrapped_eager_autograd",
) -> PreparedMethod:
    """Prepare the real model with PyTorch's default save-for-backward policy."""
    spec = MethodSpec(
        method_id="all_save",
        is_real=True,
        api="torch.nn.Module.forward",
        policy="PyTorch default save-for-backward",
        region_family=(),
        compiler_protocol="unwrapped_eager_autograd",
    )
    if execution_backend != spec.compiler_protocol:
        return _unsupported(spec, f"execution backend {execution_backend!r} is not integrated")
    try:
        model_sha256 = _model_sha256(model, ())
        executable_sha256 = _executable_sha256(spec, model_sha256, region_paths=())
    except StableSerializationError as exc:
        return _unsupported(spec, f"stable model digest unavailable: {exc}")
    identity = RuntimeIdentity(
        method_id=spec.method_id,
        status="ready",
        is_real=True,
        api=spec.api,
        policy=spec.policy,
        region_paths=(),
        compiler_protocol=spec.compiler_protocol,
        model_sha256=model_sha256,
        executable_sha256=executable_sha256,
        provenance=(("save_policy", "autograd_default"),),
    )
    return PreparedMethod(spec=spec, identity=identity, executable=model)


def prepare_aot_min_cut(
    model: nn.Module,
    capture: CapturedJointGraph,
    *,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    activation_memory_budget: float,
    partitioner_compiler: str = "inductor",
    execution_backend: str = "aot_lowered_graphmodule_eager",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> PreparedMethod:
    if isinstance(activation_memory_budget, bool) or not isinstance(activation_memory_budget, (int, float)):
        raise TypeError("activation_memory_budget must be a number in [0, 1]")
    ratio = float(activation_memory_budget)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("activation_memory_budget must be in [0, 1]")
    api = "torch._functorch.partitioners.min_cut_rematerialization_partition"
    spec = MethodSpec(
        method_id="pytorch_aot_min_cut",
        is_real=True,
        api=api,
        policy=f"activation_memory_budget={ratio:g}",
        region_family=(),
        compiler_protocol="aot_lowered_graphmodule_eager",
    )
    if execution_backend != "aot_lowered_graphmodule_eager":
        return _unsupported(spec, f"execution backend {execution_backend!r} is not integrated")
    if capture.backend != "aot" or capture.num_fwd_outputs < 1:
        return _unsupported(spec, "native min-cut executable requires a complete AOT capture")
    model_mismatch = _validate_capture_model(capture, model)
    if model_mismatch is not None:
        return _unsupported(spec, model_mismatch)
    try:
        model_sha256 = _model_sha256(model, ())
    except StableSerializationError as exc:
        return _unsupported(spec, f"stable model digest unavailable: {exc}")
    try:
        from torch._functorch import config as functorch_config
        from torch._functorch.partitioners import min_cut_rematerialization_partition
    except (ImportError, AttributeError) as exc:
        return _unsupported(spec, f"native min-cut API unavailable: {exc}")

    try:
        working_graph = copy.deepcopy(capture.joint_module)
        with functorch_config.patch(activation_memory_budget=ratio):
            fw_graph, bw_graph = min_cut_rematerialization_partition(
                working_graph,
                None,
                compiler=partitioner_compiler,
                num_fwd_outputs=capture.num_fwd_outputs,
                static_lifetime_input_indices=list(capture.static_lifetime_input_indices),
            )
    except Exception as exc:
        return _unsupported(spec, f"native min-cut partition failed: {type(exc).__name__}: {exc}")

    fw_outputs = _forward_output_names(fw_graph)
    residual_names = fw_outputs[capture.num_fwd_outputs:]
    lowered = LoweredPartition(
        plan_id=f"pytorch_aot_min_cut_ratio_{ratio:g}",
        fw_graph=fw_graph,
        bw_graph=bw_graph,
        partition_abi=PartitionABI(
            fw_output_value_ids=(),
            bw_placeholder_value_ids=(),
            tangent_value_ids=(),
            rng_state_value_ids=(),
        ),
    )
    try:
        executable = build_aot_partition_executable(
            lowered,
            model,
            num_fwd_outputs=capture.num_fwd_outputs,
            kwarg_names=tuple(name for name, _ in capture.kwarg_tree_specs),
            output_tree_spec=capture.output_tree_spec,
            output_tangent_mask=capture.output_tangent_mask,
            arg_tree_specs=capture.arg_tree_specs,
            kwarg_tree_specs=capture.kwarg_tree_specs,
        )
    except Exception as exc:
        return _unsupported(
            spec,
            f"native min-cut executable construction failed: {type(exc).__name__}: {exc}",
            fw_graph=fw_graph,
            bw_graph=bw_graph,
        )
    qualification = _qualify_training_executable(
        executable,
        model,
        example_args,
        example_kwargs,
        loss_fn,
        atol=atol,
        rtol=rtol,
    )
    fw_graph_sha256 = _graph_sha256(fw_graph)
    bw_graph_sha256 = _graph_sha256(bw_graph)
    executable_sha256 = _executable_sha256(
        spec,
        model_sha256,
        region_paths=(),
        fw_graph_sha256=fw_graph_sha256,
        bw_graph_sha256=bw_graph_sha256,
    )
    if not qualification.passed:
        return _unsupported(
            spec,
            qualification.failure_reason or "training executable qualification failed",
            fw_graph=fw_graph,
            bw_graph=bw_graph,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    identity = RuntimeIdentity(
        method_id=spec.method_id,
        status="ready",
        is_real=True,
        api=api,
        policy=spec.policy,
        region_paths=(),
        compiler_protocol=spec.compiler_protocol,
        fw_graph_sha256=fw_graph_sha256,
        bw_graph_sha256=bw_graph_sha256,
        model_sha256=model_sha256,
        executable_sha256=executable_sha256,
        fw_residual_names=residual_names,
        bw_placeholder_names=_placeholder_names(bw_graph),
        provenance=tuple(
            sorted(
                {
                    "api": api,
                    "buffers_match": qualification.buffers_match,
                    "cpu_rng_match": qualification.cpu_rng_match,
                    "cuda_rng_match": qualification.cuda_rng_match,
                    "partitioner_cost_model": partitioner_compiler,
                    "memory_budget_ratio": ratio,
                    "num_fwd_outputs": capture.num_fwd_outputs,
                    "output_tangent_mask": capture.output_tangent_mask,
                    "qualification": "eager_output_loss_parameter_grad",
                    "solver": "PyTorch min-cut rematerialization partitioner",
                    "torch_version": torch.__version__,
                    "training_flags_match": qualification.training_flags_match,
                }.items()
            )
        ),
    )
    return PreparedMethod(
        spec=spec,
        identity=identity,
        executable=executable,
        fw_graph=fw_graph,
        bw_graph=bw_graph,
    )


_BLOCK_REGION_FAMILIES: dict[str, tuple[str, ...]] = {
    "resnet50": ("layer1", "layer2", "layer3", "layer4"),
    "vit_b_16": ("encoder.layers",),
    "bert_base": ("bert.encoder.layer",),
    "bert_like": ("bert.encoder.layer",),
    "gpt2": ("blocks",),
    "gpt2_like": ("blocks",),
}


def resolve_block_regions(model: nn.Module, registry_key: str) -> tuple[str, ...]:
    declared = _BLOCK_REGION_FAMILIES.get(registry_key)
    if declared is None:
        return ()
    try:
        modules = tuple(model.get_submodule(path) for path in declared)
    except (AttributeError, KeyError):
        return ()
    if any(not isinstance(module, nn.Module) for module in modules):
        return ()
    return declared


def _expand_checkpoint_paths(model: nn.Module, region_paths: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for path in region_paths:
        module = model.get_submodule(path)
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            expanded.extend(f"{path}.{index}" for index in range(len(module)))
        else:
            expanded.append(path)
    return tuple(expanded)


_ACTIVE_REGION_OWNERS: weakref.WeakKeyDictionary[nn.Module, set[str]] = weakref.WeakKeyDictionary()
_ACTIVE_REGION_OWNERS_LOCK = threading.Lock()


def _regions_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


@contextmanager
def _region_buffer_transaction(model: nn.Module, region_paths: tuple[str, ...]):
    snapshots: list[tuple[nn.Module, str, torch.Tensor, torch.Tensor]] = []
    seen: set[tuple[int, str]] = set()
    for path in region_paths:
        for module in model.get_submodule(path).modules():
            for name, buffer in module._buffers.items():
                key = (id(module), name)
                if buffer is None or key in seen:
                    continue
                seen.add(key)
                snapshots.append((module, name, buffer, buffer.detach().clone()))
    try:
        yield
    finally:
        with torch.no_grad():
            for module, name, original_buffer, value in snapshots:
                module._buffers[name] = original_buffer
                original_buffer.copy_(value)


def _tracked_checkpoint_context_fn(
    model: nn.Module,
    region_paths: tuple[str, ...],
    tracker: dict[str, int],
    base_context_fn: Callable[[], tuple[Any, Any]] | None = None,
) -> Callable[[], tuple[Any, Any]]:
    def context_fn() -> tuple[Any, Any]:
        forward_context, recompute_context = (
            base_context_fn() if base_context_fn is not None else (nullcontext(), nullcontext())
        )

        @contextmanager
        def tracked_recompute():
            tracker["recompute_count"] = tracker.get("recompute_count", 0) + 1
            with _region_buffer_transaction(model, region_paths):
                with recompute_context:
                    yield

        return forward_context, tracked_recompute()

    return context_fn


class _RegionCheckpointExecutable(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        region_paths: tuple[str, ...],
        *,
        context_fn: Callable[[], tuple[Any, Any]],
        tracker: dict[str, int],
    ) -> None:
        super().__init__()
        self.model = model
        self.region_paths = region_paths
        self.context_fn = context_fn
        self.tracker = tracker

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with _ACTIVE_REGION_OWNERS_LOCK:
            active_regions = _ACTIVE_REGION_OWNERS.setdefault(self.model, set())
            if any(
                _regions_overlap(requested, active)
                for requested in self.region_paths
                for active in active_regions
            ):
                raise RuntimeError(
                    "region checkpoint executable does not support concurrent or reentrant ownership "
                    "of overlapping regions on the same model"
                )
            active_regions.update(self.region_paths)
        restored: list[tuple[nn.Module, bool, Any]] = []
        try:
            for path in self.region_paths:
                region = self.model.get_submodule(path)
                original_forward = region.forward
                had_instance_forward = "forward" in region.__dict__
                original_instance_forward = region.__dict__.get("forward")

                def checkpointed_forward(
                    *region_args: Any,
                    _original: Callable[..., Any] = original_forward,
                    **region_kwargs: Any,
                ) -> Any:
                    self.tracker["checkpoint_call_count"] = self.tracker.get("checkpoint_call_count", 0) + 1
                    return checkpoint.checkpoint(
                        _original,
                        *region_args,
                        use_reentrant=False,
                        context_fn=self.context_fn,
                        **region_kwargs,
                    )

                restored.append((region, had_instance_forward, original_instance_forward))
                region.forward = checkpointed_forward
            return self.model(*args, **kwargs)
        finally:
            for region, had_instance_forward, original_instance_forward in reversed(restored):
                if had_instance_forward:
                    region.forward = original_instance_forward
                elif "forward" in region.__dict__:
                    delattr(region, "forward")
            with _ACTIVE_REGION_OWNERS_LOCK:
                active_regions = _ACTIVE_REGION_OWNERS.get(self.model)
                if active_regions is not None:
                    active_regions.difference_update(self.region_paths)
                    if not active_regions:
                        del _ACTIVE_REGION_OWNERS[self.model]


def prepare_block_activation_checkpoint(
    model: nn.Module,
    registry_key: str,
    *,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> PreparedMethod:
    api = "torch.utils.checkpoint.checkpoint(use_reentrant=False)"
    declared_regions = _BLOCK_REGION_FAMILIES.get(registry_key, ())
    spec = MethodSpec(
        method_id="block_activation_checkpoint",
        is_real=True,
        api=api,
        policy="non_reentrant checkpoint over frozen architecture regions",
        region_family=declared_regions,
        compiler_protocol="eager_non_reentrant_region_checkpoint",
    )
    regions = resolve_block_regions(model, registry_key)
    if not regions:
        return _unsupported(spec, f"no validated block region family for registry key {registry_key!r}")
    checkpoint_paths = _expand_checkpoint_paths(model, regions)
    try:
        model_sha256 = _model_sha256(model, checkpoint_paths)
    except StableSerializationError as exc:
        return _unsupported(
            spec,
            f"stable model digest unavailable: {exc}",
            regions=checkpoint_paths,
        )
    executable_sha256 = _executable_sha256(spec, model_sha256, region_paths=checkpoint_paths)
    tracker: dict[str, int] = {}
    context_fn = _tracked_checkpoint_context_fn(model, checkpoint_paths, tracker)
    executable = _RegionCheckpointExecutable(
        model,
        checkpoint_paths,
        context_fn=context_fn,
        tracker=tracker,
    )
    qualification = _qualify_training_executable(
        executable,
        model,
        example_args,
        example_kwargs,
        loss_fn,
        atol=atol,
        rtol=rtol,
    )
    if not qualification.passed:
        return _unsupported(
            spec,
            qualification.failure_reason or "block AC qualification failed",
            regions=checkpoint_paths,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    if tracker.get("checkpoint_call_count", 0) == 0 or tracker.get("recompute_count", 0) == 0:
        return _unsupported(
            spec,
            "block AC qualification did not observe checkpoint and recompute execution",
            regions=checkpoint_paths,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    identity = RuntimeIdentity(
        method_id=spec.method_id,
        status="ready",
        is_real=True,
        api=api,
        policy=spec.policy,
        region_paths=checkpoint_paths,
        compiler_protocol=spec.compiler_protocol,
        model_sha256=model_sha256,
        executable_sha256=executable_sha256,
        provenance=(
            ("buffers_match", qualification.buffers_match),
            ("checkpoint_call_count", tracker["checkpoint_call_count"]),
            ("cpu_rng_match", qualification.cpu_rng_match),
            ("cuda_rng_match", qualification.cuda_rng_match),
            ("declared_region_family", regions),
            ("recompute_count", tracker["recompute_count"]),
            ("training_flags_match", qualification.training_flags_match),
            ("use_reentrant", False),
        ),
    )
    return PreparedMethod(spec=spec, identity=identity, executable=executable)


_SAC_POLICY_SOURCE = (
    "MUST_SAVE for convolution/mm/addmm/bmm/matmul/scaled-dot-product-attention; "
    "PREFER_RECOMPUTE for all other dispatchable operations"
)


def _aten_op(packet_name: str, overload: str = "default") -> Any | None:
    packet = getattr(torch.ops.aten, packet_name, None)
    return None if packet is None else getattr(packet, overload, None)


_SAC_SAVE_OPS = frozenset(
    op
    for op in (
        _aten_op("convolution"),
        _aten_op("mm"),
        _aten_op("addmm"),
        _aten_op("bmm"),
        _aten_op("matmul"),
        _aten_op("_scaled_dot_product_flash_attention"),
        _aten_op("_scaled_dot_product_efficient_attention"),
        _aten_op("_scaled_dot_product_cudnn_attention"),
    )
    if op is not None
)


def make_sac_policy(decisions: dict[str, int] | None = None) -> Callable[..., checkpoint.CheckpointPolicy]:
    def policy(_ctx: Any, op: Any, *args: Any, **kwargs: Any) -> checkpoint.CheckpointPolicy:
        del args, kwargs
        result = (
            checkpoint.CheckpointPolicy.MUST_SAVE
            if op in _SAC_SAVE_OPS
            else checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        )
        if decisions is not None:
            decisions[result.name] = decisions.get(result.name, 0) + 1
        return result

    return policy


def prepare_selective_activation_checkpoint(
    model: nn.Module,
    registry_key: str,
    *,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> PreparedMethod:
    api = "torch.utils.checkpoint.create_selective_checkpoint_contexts"
    declared_regions = _BLOCK_REGION_FAMILIES.get(registry_key, ())
    policy_hash = hashlib.sha256(_SAC_POLICY_SOURCE.encode("utf-8")).hexdigest()
    spec = MethodSpec(
        method_id="selective_activation_checkpoint",
        is_real=True,
        api=api,
        policy=_SAC_POLICY_SOURCE,
        region_family=declared_regions,
        compiler_protocol="eager_non_reentrant_selective_region_checkpoint",
    )
    if not hasattr(checkpoint, "create_selective_checkpoint_contexts") or not hasattr(checkpoint, "CheckpointPolicy"):
        return _unsupported(spec, "selective activation checkpoint APIs are unavailable")
    regions = resolve_block_regions(model, registry_key)
    if not regions:
        return _unsupported(spec, f"no validated SAC region family for registry key {registry_key!r}")
    checkpoint_paths = _expand_checkpoint_paths(model, regions)
    try:
        model_sha256 = _model_sha256(model, checkpoint_paths)
    except StableSerializationError as exc:
        return _unsupported(
            spec,
            f"stable model digest unavailable: {exc}",
            regions=checkpoint_paths,
        )
    executable_sha256 = _executable_sha256(spec, model_sha256, region_paths=checkpoint_paths)
    tracker: dict[str, int] = {}
    policy = make_sac_policy(tracker)
    sac_context_fn = functools.partial(checkpoint.create_selective_checkpoint_contexts, policy)
    context_fn = _tracked_checkpoint_context_fn(model, checkpoint_paths, tracker, sac_context_fn)
    executable = _RegionCheckpointExecutable(
        model,
        checkpoint_paths,
        context_fn=context_fn,
        tracker=tracker,
    )
    qualification = _qualify_training_executable(
        executable,
        model,
        example_args,
        example_kwargs,
        loss_fn,
        atol=atol,
        rtol=rtol,
    )
    if not qualification.passed:
        return _unsupported(
            spec,
            qualification.failure_reason or "SAC qualification failed",
            regions=checkpoint_paths,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    must_save_count = tracker.get("MUST_SAVE", 0)
    prefer_recompute_count = tracker.get("PREFER_RECOMPUTE", 0)
    if must_save_count == 0 or prefer_recompute_count == 0:
        return _unsupported(
            spec,
            "SAC qualification did not observe both MUST_SAVE and PREFER_RECOMPUTE decisions",
            regions=checkpoint_paths,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    if tracker.get("checkpoint_call_count", 0) == 0 or tracker.get("recompute_count", 0) == 0:
        return _unsupported(
            spec,
            "SAC qualification did not observe checkpoint and recompute execution",
            regions=checkpoint_paths,
            model_sha256=model_sha256,
            executable_sha256=executable_sha256,
        )
    identity = RuntimeIdentity(
        method_id=spec.method_id,
        status="ready",
        is_real=True,
        api=api,
        policy=spec.policy,
        region_paths=checkpoint_paths,
        compiler_protocol=spec.compiler_protocol,
        model_sha256=model_sha256,
        executable_sha256=executable_sha256,
        provenance=(
            ("buffers_match", qualification.buffers_match),
            ("checkpoint_call_count", tracker["checkpoint_call_count"]),
            ("cpu_rng_match", qualification.cpu_rng_match),
            ("cuda_rng_match", qualification.cuda_rng_match),
            ("must_save_count", must_save_count),
            ("policy_hash", policy_hash),
            ("policy_source", _SAC_POLICY_SOURCE),
            ("prefer_recompute_count", prefer_recompute_count),
            ("recompute_count", tracker["recompute_count"]),
            ("training_flags_match", qualification.training_flags_match),
            ("use_reentrant", False),
        ),
    )
    return PreparedMethod(spec=spec, identity=identity, executable=executable)
