from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import os
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Sequence

import torch
from torch import fx, nn
from torch.utils import checkpoint

from peakaware.runtime.stable_callable import register_stable_callable, unregister_stable_callable

from .baselines import (
    MethodSpec,
    PreparedMethod,
    RuntimeIdentity,
    StableSerializationError,
    _buffers_equal,
    _canonical_sha256,
    _clone_buffers,
    _clone_grads,
    _clone_output,
    _cuda_rng_state,
    _executable_sha256,
    _graph_sha256,
    _model_sha256,
    _outputs_match,
    _restore_grads,
    _restore_rng,
    _rng_equal,
    _training_flags,
    _unsupported,
    resolve_publication_regions,
)


PublicationMethod = Literal["all_save", "block_ac", "sac"]
PublicationBackend = Literal["aot_eager", "inductor"]

try:
    from torch._inductor.custom_graph_pass import CustomPartitionerFn as _CustomPartitionerFn
except (ImportError, AttributeError):
    _CustomPartitionerFn = object  # type: ignore[assignment,misc]


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
        _aten_op("scaled_dot_product_attention"),
        _aten_op("native_batch_norm"),
        _aten_op("cudnn_batch_norm"),
        _aten_op("_scaled_dot_product_flash_attention"),
        _aten_op("_scaled_dot_product_efficient_attention"),
        _aten_op("_scaled_dot_product_cudnn_attention"),
    )
    if op is not None
)
_SAC_SAVE_OP_NAMES = tuple(sorted(str(op) for op in _SAC_SAVE_OPS))
_STATEFUL_POLICY_SOURCE = (
    "MUST_SAVE=batch_norm_and_scalar_integer_inplace_counter_update;"
    "PREFER_RECOMPUTE=all_other_ops"
)
_SAC_POLICY_SOURCE = (
    "MUST_SAVE={" + ",".join(_SAC_SAVE_OP_NAMES) + "},scalar_integer_inplace_counter_update;"
    "PREFER_RECOMPUTE=all_other_ops"
)
_SAC_POLICY_HASH = hashlib.sha256(
    _SAC_POLICY_SOURCE.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class FrozenSACPolicy:
    save_ops: frozenset[Any] = _SAC_SAVE_OPS

    def __call__(self, _ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return (
            checkpoint.CheckpointPolicy.MUST_SAVE
            if op in self.save_ops
            else checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        )


def _frozen_sac_policy(_ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
    del kwargs
    scalar_counter_update = (
        op is _aten_op("add_", "Tensor")
        and args
        and isinstance(args[0], torch.Tensor)
        and args[0].ndim == 0
        and not args[0].is_floating_point()
    )
    return (
        checkpoint.CheckpointPolicy.MUST_SAVE
        if op in _SAC_SAVE_OPS or scalar_counter_update
        else checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    )


def _stateful_block_policy(_ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
    del kwargs
    scalar_counter_update = (
        op is _aten_op("add_", "Tensor")
        and args
        and isinstance(args[0], torch.Tensor)
        and args[0].ndim == 0
        and not args[0].is_floating_point()
    )
    batch_norm = op in {_aten_op("native_batch_norm"), _aten_op("cudnn_batch_norm")}
    return (
        checkpoint.CheckpointPolicy.MUST_SAVE
        if batch_norm or scalar_counter_update
        else checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    )


@dataclass
class _CompilerEvidence:
    joint_graph_sha256: list[str] = field(default_factory=list)
    fw_graph_sha256: list[str] = field(default_factory=list)
    bw_graph_sha256: list[str] = field(default_factory=list)
    joint_callback_count: int = 0
    fw_callback_count: int = 0
    bw_callback_count: int = 0
    partition_callback_count: int = 0
    checkpoint_call_count: int = 0
    recompute_count: int = 0
    fw_recompute_count: int = 0
    bw_recompute_count: int = 0
    must_save_count: int = 0
    prefer_recompute_count: int = 0
    static_lifetime_input_indices_observed: bool = False


@dataclass(frozen=True)
class _QualificationEvidence:
    buffers_match: bool
    cpu_rng_match: bool
    cuda_rng_match: bool
    training_flags_match: bool


def _policy_counts(graph_module: fx.GraphModule) -> tuple[int, int]:
    must_save = 0
    prefer_recompute = 0
    for node in graph_module.graph.nodes:
        policy = node.meta.get("recompute")
        if policy is checkpoint.CheckpointPolicy.MUST_SAVE:
            must_save += 1
        elif policy is checkpoint.CheckpointPolicy.PREFER_RECOMPUTE:
            prefer_recompute += 1
    return must_save, prefer_recompute


def _observe_joint(
    graph_module: fx.GraphModule,
    evidence: _CompilerEvidence,
    callback: Callable[[str, fx.GraphModule, str], None] | None,
) -> None:
    digest = _graph_sha256(graph_module)
    ac_graph_ids = {
        node.meta["ac_graph_id"]
        for node in graph_module.graph.nodes
        if node.meta.get("ac_graph_id") is not None
    }
    must_save, prefer_recompute = _policy_counts(graph_module)
    evidence.joint_graph_sha256.append(digest)
    evidence.joint_callback_count += 1
    evidence.checkpoint_call_count = len(ac_graph_ids)
    evidence.recompute_count = prefer_recompute
    evidence.must_save_count = must_save
    evidence.prefer_recompute_count = prefer_recompute
    if callback is not None:
        callback("joint", graph_module, digest)


def _observe_partition_graph(
    phase: Literal["fw", "bw"],
    graph_module: fx.GraphModule,
    evidence: _CompilerEvidence,
    callback: Callable[[str, fx.GraphModule, str], None] | None,
) -> None:
    digest = _graph_sha256(graph_module)
    _, recompute = _policy_counts(graph_module)
    getattr(evidence, f"{phase}_graph_sha256").append(digest)
    setattr(evidence, f"{phase}_callback_count", getattr(evidence, f"{phase}_callback_count") + 1)
    setattr(evidence, f"{phase}_recompute_count", recompute)
    if callback is not None:
        callback(phase, graph_module, digest)


def _default_partition(
    graph: fx.GraphModule,
    inputs: Sequence[object],
    kwargs: dict[str, Any],
) -> tuple[fx.GraphModule, fx.GraphModule]:
    from torch._functorch.partitioners import default_partition

    parameters = inspect.signature(default_partition).parameters
    supported_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return default_partition(graph, inputs, **supported_kwargs)


_MODEL_OWNERS: weakref.WeakKeyDictionary[nn.Module, _StaticRegionInstaller] = weakref.WeakKeyDictionary()
_MODEL_OWNERS_LOCK = threading.Lock()


class _StaticRegionInstaller:
    def __init__(
        self,
        model: nn.Module,
        region_paths: tuple[str, ...],
        method: PublicationMethod,
        *,
        functionalize_resnet_bottleneck: bool = False,
        restore_cpu_rng_after_call: bool = False,
        cudnn_enabled_override: bool | None = None,
    ) -> None:
        self.model = model
        self.region_paths = region_paths
        self.method = method
        self.functionalize_resnet_bottleneck = functionalize_resnet_bottleneck
        self.restore_cpu_rng_after_call = restore_cpu_rng_after_call
        self.cudnn_enabled_override = cudnn_enabled_override
        self._installed: list[tuple[nn.Module, bool, Any, Callable[..., Any], object]] = []
        self._claimed = False
        self._closed = False

    def claim(self, reference_model: nn.Module) -> None:
        with _MODEL_OWNERS_LOCK:
            if self.model in _MODEL_OWNERS:
                raise RuntimeError("model already has an active publication compiler owner")
            if reference_model in _MODEL_OWNERS:
                raise RuntimeError("reference_model has an active publication compiler owner")
            _MODEL_OWNERS[self.model] = self
            self._claimed = True

    def install(self) -> None:
        if self.method == "all_save":
            return
        policy = _frozen_sac_policy if self.method == "sac" else _stateful_block_policy
        context_fn = functools.partial(checkpoint.create_selective_checkpoint_contexts, policy)
        for path in self.region_paths:
            region = self.model.get_submodule(path)
            had_instance_forward = "forward" in region.__dict__
            instance_forward = region.__dict__.get("forward")
            original_forward = region.forward
            checkpoint_forward = (
                functools.partial(_out_of_place_resnet_bottleneck_forward, region)
                if self.functionalize_resnet_bottleneck
                else original_forward
            )

            @functools.wraps(original_forward)
            def checkpointed_forward(
                *region_args: Any,
                _forward: Callable[..., Any] = checkpoint_forward,
                _context_fn: Callable[[], tuple[Any, Any]] | None = context_fn,
                **region_kwargs: Any,
            ) -> Any:
                def invoke(packed_args: tuple[Any, ...], packed_kwargs: dict[str, Any]) -> Any:
                    return _forward(*packed_args, **packed_kwargs)

                checkpoint_options: dict[str, Any] = {"use_reentrant": False}
                if _context_fn is not None:
                    checkpoint_options["context_fn"] = _context_fn
                return checkpoint.checkpoint(
                    invoke,
                    region_args,
                    region_kwargs,
                    **checkpoint_options,
                )

            token = register_stable_callable(checkpointed_forward)
            try:
                region.forward = checkpointed_forward
            except BaseException:
                unregister_stable_callable(checkpointed_forward, token)
                raise
            self._installed.append((region, had_instance_forward, instance_forward, checkpointed_forward, token))

    def close(self) -> None:
        if self._closed:
            return
        for region, had_instance_forward, instance_forward, wrapper, token in reversed(self._installed):
            unregister_stable_callable(wrapper, token)
            if had_instance_forward:
                region.forward = instance_forward
            elif "forward" in region.__dict__:
                delattr(region, "forward")
        self._installed.clear()
        with _MODEL_OWNERS_LOCK:
            if self._claimed and _MODEL_OWNERS.get(self.model) is self:
                del _MODEL_OWNERS[self.model]
        self._claimed = False
        self._closed = True


def _out_of_place_resnet_bottleneck_forward(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
    identity = value
    output = torch.relu(block.bn1(block.conv1(value)))
    output = torch.relu(block.bn2(block.conv2(output)))
    output = block.bn3(block.conv3(output))
    downsample = getattr(block, "downsample", None)
    if downsample is not None:
        identity = downsample(value)
    return torch.relu(output + identity)


class PublicationExecutable:
    def __init__(
        self,
        model: nn.Module,
        compiled: Callable[..., Any],
        installer: _StaticRegionInstaller,
        runtime_observations: dict[str, Any],
    ) -> None:
        self.model = model
        self.compiled = compiled
        self._installer = installer
        self._closed = False
        self._peakaware_runtime_observations = runtime_observations

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("publication executable is closed")
        cudnn_context = (
            torch.backends.cudnn.flags(enabled=self._installer.cudnn_enabled_override)
            if self._installer.cudnn_enabled_override is not None
            else nullcontext()
        )
        with cudnn_context:
            if not self._installer.restore_cpu_rng_after_call:
                return self.compiled(*args, **kwargs)
            cpu_rng = torch.get_rng_state().clone()
            try:
                return self.compiled(*args, **kwargs)
            finally:
                torch.set_rng_state(cpu_rng)

    def close(self) -> None:
        if not self._closed:
            self._installer.close()
            self._closed = True

    def __enter__(self) -> PublicationExecutable:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.parameters(*args, **kwargs)

    def named_buffers(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.named_buffers(*args, **kwargs)

    def buffers(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.buffers(*args, **kwargs)

    def named_modules(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.named_modules(*args, **kwargs)

    def get_submodule(self, target: str) -> nn.Module:
        return self.model.get_submodule(target)

    def train(self, mode: bool = True) -> PublicationExecutable:
        self.model.train(mode)
        return self

    def eval(self) -> PublicationExecutable:
        return self.train(False)


def close_publication_executable(executable: Any) -> None:
    close = getattr(executable, "close", None)
    if callable(close):
        close()


def _aot_eager_backend(
    evidence: _CompilerEvidence,
    callback: Callable[[str, fx.GraphModule, str], None] | None,
) -> Callable[..., Any]:
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    def compile_graph(graph: fx.GraphModule, _inputs: list[Any], phase: Literal["fw", "bw"]):
        _observe_partition_graph(phase, graph, evidence, callback)
        return make_boxed_func(graph.forward)

    def partition(graph: fx.GraphModule, inputs: Sequence[Any], **kwargs: Any):
        evidence.partition_callback_count += 1
        evidence.static_lifetime_input_indices_observed = "static_lifetime_input_indices" in kwargs
        _observe_joint(graph, evidence, callback)
        return _default_partition(graph, inputs, kwargs)

    compiler = aot_autograd(
        fw_compiler=lambda graph, inputs: compile_graph(graph, inputs, "fw"),
        bw_compiler=lambda graph, inputs: compile_graph(graph, inputs, "bw"),
        partition_fn=partition,
        keep_inference_input_mutations=True,
    )

    def backend(graph: fx.GraphModule, inputs: list[Any]) -> Any:
        return compiler(graph, inputs)

    return backend


class PublicationDefaultPartitionerFn(_CustomPartitionerFn):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        evidence: _CompilerEvidence,
        callback: Callable[[str, fx.GraphModule, str], None] | None,
    ) -> None:
        self._evidence = evidence
        self._callback = callback

    def __call__(self, graph: fx.GraphModule, inputs: Sequence[object], **kwargs: Any):
        self._evidence.partition_callback_count += 1
        self._evidence.static_lifetime_input_indices_observed = (
            "static_lifetime_input_indices" in kwargs
        )
        _observe_joint(graph, self._evidence, self._callback)
        fw_graph, bw_graph = _default_partition(graph, inputs, kwargs)
        _observe_partition_graph("fw", fw_graph, self._evidence, self._callback)
        _observe_partition_graph("bw", bw_graph, self._evidence, self._callback)
        return fw_graph, bw_graph

    def uuid(self) -> str:
        from torch._functorch.partitioners import default_partition

        payload = "peakaware-default-partition-v2\n" + inspect.getsource(default_partition)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __getstate__(self) -> dict[str, Any]:
        return {}


def _inductor_options(
    evidence: _CompilerEvidence,
    callback: Callable[[str, fx.GraphModule, str], None] | None,
) -> tuple[dict[str, Any], str]:
    try:
        from torch._inductor.custom_graph_pass import CustomPartitionerFn
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Inductor custom partitioner API is unavailable: {exc}") from exc
    if _CustomPartitionerFn is object or not issubclass(PublicationDefaultPartitionerFn, CustomPartitionerFn):
        raise RuntimeError("Inductor CustomPartitionerFn ABI does not match the adapter")
    return {"custom_partitioner_fn": PublicationDefaultPartitionerFn(evidence, callback)}, (
        "torch._inductor.custom_graph_pass.CustomPartitionerFn(default_partition)"
    )


@contextmanager
def _cold_compile_cache(backend: PublicationBackend) -> Iterator[None]:
    torch._dynamo.reset()
    if backend != "inductor":
        yield
        return
    try:
        from torch._inductor.utils import fresh_inductor_cache
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"fresh Inductor cache API is unavailable: {exc}") from exc
    with fresh_inductor_cache():
        yield


def _checkpoint_regions(
    model: nn.Module,
    registry_key: str,
    publication_blocks: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    del model, registry_key
    return publication_blocks, "whole_publication_block"


def _states_match(reference_model: nn.Module, model: nn.Module) -> bool:
    reference = reference_model.state_dict()
    candidate = model.state_dict()
    return reference.keys() == candidate.keys() and all(
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.detach(), right.detach())
        for left, right in zip(reference.values(), candidate.values())
    )


def _qualify_against_reference(
    executable: Callable[..., Any],
    reference_model: nn.Module,
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> _QualificationEvidence:
    if reference_model is model:
        raise RuntimeError("reference_model must be independent from the candidate model")
    if not _states_match(reference_model, model):
        raise RuntimeError("reference_model and candidate initial states differ")
    reference_state = copy.deepcopy(reference_model.state_dict())
    candidate_state = copy.deepcopy(model.state_dict())
    reference_grads = _clone_grads(reference_model)
    candidate_grads = _clone_grads(model)
    reference_flags = tuple((module, module.training) for module in reference_model.modules())
    candidate_flags = tuple((module, module.training) for module in model.modules())
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_state()

    def restore() -> None:
        reference_model.load_state_dict(reference_state)
        model.load_state_dict(candidate_state)
        _restore_grads(reference_model, reference_grads)
        _restore_grads(model, candidate_grads)
        for module, training in (*reference_flags, *candidate_flags):
            module.training = training
        _restore_rng(cpu_rng, cuda_rng)

    try:
        restore()
        reference_model.zero_grad(set_to_none=True)
        expected_output = reference_model(*example_args, **example_kwargs)
        expected_loss = loss_fn(expected_output)
        if not isinstance(expected_loss, torch.Tensor) or expected_loss.ndim != 0:
            raise RuntimeError("loss_fn must return a scalar tensor")
        expected_loss.backward()
        expected_output_copy = _clone_output(expected_output)
        expected_loss_copy = expected_loss.detach().clone()
        expected_grads = _clone_grads(reference_model)
        expected_buffers = _clone_buffers(reference_model)
        expected_cpu_rng = torch.get_rng_state().clone()
        expected_cuda_rng = _cuda_rng_state()
        expected_flags = _training_flags(reference_model)

        restore()
        model.zero_grad(set_to_none=True)
        actual_output = executable(*example_args, **example_kwargs)
        actual_loss = loss_fn(actual_output)
        if not isinstance(actual_loss, torch.Tensor) or actual_loss.ndim != 0:
            raise RuntimeError("compiled loss_fn must return a scalar tensor")
        actual_loss.backward()
        actual_grads = _clone_grads(model)
        actual_buffers = _clone_buffers(model)
        actual_cpu_rng = torch.get_rng_state().clone()
        actual_cuda_rng = _cuda_rng_state()
        actual_flags = _training_flags(model)

        if not _outputs_match(actual_output, expected_output_copy, atol=atol, rtol=rtol):
            raise RuntimeError("compiled outputs do not match independent eager reference")
        if not torch.allclose(actual_loss.detach(), expected_loss_copy, atol=atol, rtol=rtol, equal_nan=True):
            raise RuntimeError("compiled loss does not match independent eager reference")
        for (name, _), actual, expected in zip(model.named_parameters(), actual_grads, expected_grads):
            if actual is None and expected is None:
                continue
            if actual is None or expected is None or not torch.allclose(
                actual, expected, atol=atol, rtol=rtol, equal_nan=True
            ):
                raise RuntimeError(f"compiled gradient differs for parameter {name!r}")
        buffers_match = _buffers_equal(actual_buffers, expected_buffers)
        cpu_rng_match = torch.equal(actual_cpu_rng, expected_cpu_rng)
        cuda_rng_match = _rng_equal(actual_cuda_rng, expected_cuda_rng)
        training_flags_match = actual_flags == expected_flags
        if not buffers_match:
            raise RuntimeError("compiled buffers do not match independent eager reference")
        if not cpu_rng_match or not cuda_rng_match:
            raise RuntimeError("compiled RNG does not match independent eager reference")
        if not training_flags_match:
            raise RuntimeError("compiled training flags do not match independent eager reference")
        return _QualificationEvidence(True, True, True, True)
    finally:
        restore()


def _method_spec(
    method: PublicationMethod,
    backend: PublicationBackend,
    publication_blocks: tuple[str, ...],
) -> MethodSpec:
    policies = {
        "all_save": "default_partition_without_activation_checkpoint",
        "block_ac": "non_reentrant_checkpoint_over_frozen_regions",
        "sac": _SAC_POLICY_SOURCE,
    }
    return MethodSpec(
        method_id=method,
        is_real=True,
        api="torch.compile(fullgraph=True)",
        policy=policies[method],
        region_family=publication_blocks,
        compiler_protocol=f"{backend}:matched_publication_v2_default_partition",
    )


def prepare_publication_compiler(
    model: nn.Module,
    registry_key: str,
    *,
    reference_model: nn.Module,
    method: PublicationMethod,
    backend: PublicationBackend,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    loss_fn: Callable[[Any], torch.Tensor],
    cache_identity: str | None = None,
    graph_callback: Callable[[str, fx.GraphModule, str], None] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> PreparedMethod:
    if method not in {"all_save", "block_ac", "sac"}:
        raise ValueError(f"unknown publication method {method!r}")
    if backend not in {"aot_eager", "inductor"}:
        raise ValueError(f"unknown publication backend {backend!r}")
    publication_blocks = () if method == "all_save" else resolve_publication_regions(model, registry_key)
    spec = _method_spec(method, backend, publication_blocks)
    if reference_model is model:
        return _unsupported(spec, "reference_model must be an independent object")
    if method != "all_save" and not publication_blocks:
        return _unsupported(spec, f"no complete publication region family for registry key {registry_key!r}")
    checkpoint_regions, region_boundary = _checkpoint_regions(model, registry_key, publication_blocks)
    if method != "all_save" and not checkpoint_regions:
        return _unsupported(spec, "no safe checkpoint boundary covers the publication blocks")
    if method == "sac" and (
        not hasattr(checkpoint, "create_selective_checkpoint_contexts")
        or not hasattr(checkpoint, "CheckpointPolicy")
    ):
        return _unsupported(spec, "selective activation checkpoint APIs are unavailable")

    try:
        model_sha256 = _model_sha256(model, checkpoint_regions)
    except StableSerializationError as exc:
        return _unsupported(spec, f"stable model digest unavailable: {exc}", regions=checkpoint_regions)
    stable_cache_identity = cache_identity or _canonical_sha256(
        {"backend": backend, "method": method, "model_sha256": model_sha256}
    )
    evidence = _CompilerEvidence()
    cudnn_enabled_override = False if registry_key == "resnet50" else None
    installer = _StaticRegionInstaller(
        model,
        checkpoint_regions,
        method,
        functionalize_resnet_bottleneck=registry_key == "resnet50" and method != "all_save",
        restore_cpu_rng_after_call=method == "block_ac",
        cudnn_enabled_override=cudnn_enabled_override,
    )
    state_keys = tuple(model.state_dict())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    buffer_names = tuple(name for name, _ in model.named_buffers())
    try:
        installer.claim(reference_model)
        if backend == "aot_eager":
            compiler_backend = _aot_eager_backend(evidence, graph_callback)
            compile_options: dict[str, Any] = {}
            partition_api = "torch._functorch.partitioners.default_partition"
        else:
            compile_options, partition_api = _inductor_options(evidence, graph_callback)
            compiler_backend = "inductor"
        installer.install()
        if tuple(model.state_dict()) != state_keys:
            raise RuntimeError("static region installation changed state_dict keys")
        if tuple(name for name, _ in model.named_parameters()) != parameter_names:
            raise RuntimeError("static region installation changed parameter FQNs")
        if tuple(name for name, _ in model.named_buffers()) != buffer_names:
            raise RuntimeError("static region installation changed buffer FQNs")
        cudnn_context = (
            torch.backends.cudnn.flags(enabled=cudnn_enabled_override)
            if cudnn_enabled_override is not None
            else nullcontext()
        )
        with _cold_compile_cache(backend), cudnn_context:
            compiled = torch.compile(
                model,
                backend=compiler_backend,
                fullgraph=True,
                options=compile_options or None,
            )
            runtime_observations: dict[str, Any] = {}
            executable = PublicationExecutable(model, compiled, installer, runtime_observations)
            qualification = _qualify_against_reference(
                executable,
                reference_model,
                model,
                example_args,
                example_kwargs,
                loss_fn,
                atol=atol,
                rtol=rtol,
            )
        if not evidence.joint_graph_sha256 or not evidence.fw_graph_sha256 or not evidence.bw_graph_sha256:
            raise RuntimeError("joint/FW/BW compiler callbacks were not all observed")
        if evidence.partition_callback_count < 1:
            raise RuntimeError("explicit default_partition callback was not observed")
        if method == "all_save":
            if evidence.checkpoint_call_count != 0 or evidence.recompute_count != 0:
                raise RuntimeError("all-save graph unexpectedly contains checkpoint annotations")
        elif evidence.checkpoint_call_count < 1 or evidence.recompute_count < 1:
            raise RuntimeError("checkpoint/recompute annotations were not observed in the joint graph")
        if method == "sac" and (evidence.must_save_count < 1 or evidence.prefer_recompute_count < 1):
            raise RuntimeError("SAC joint graph did not observe both policy classes")
    except Exception as exc:
        installer.close()
        return _unsupported(
            spec,
            f"matched compiler preparation failed: {type(exc).__name__}: {exc}",
            regions=checkpoint_regions,
            model_sha256=model_sha256,
        )

    fw_hash = evidence.fw_graph_sha256[0]
    bw_hash = evidence.bw_graph_sha256[0]
    executable_sha256 = _executable_sha256(
        spec,
        model_sha256,
        region_paths=checkpoint_regions,
        fw_graph_sha256=fw_hash,
        bw_graph_sha256=bw_hash,
    )
    provenance = {
        "backend": backend,
        "buffers_match": qualification.buffers_match,
        "bw_callback_count": evidence.bw_callback_count,
        "bw_recompute_count": evidence.bw_recompute_count,
        "cache_identity": stable_cache_identity,
        "cache_reuse_disabled": True,
        "checkpoint_call_count": evidence.checkpoint_call_count,
        "checkpoint_region_count": len(checkpoint_regions),
        "checkpoint_region_paths": checkpoint_regions,
        "compiler_callback_pid": os.getpid(),
        "compiler_callbacks_observed_in_process": True,
        "cpu_rng_match": qualification.cpu_rng_match,
        "cuda_rng_match": qualification.cuda_rng_match,
        "fresh_cache_per_preparation": True,
        "fullgraph": True,
        "fw_callback_count": evidence.fw_callback_count,
        "fw_recompute_count": evidence.fw_recompute_count,
        "graph_break_count": 0,
        "joint_callback_count": evidence.joint_callback_count,
        "joint_graph_sha256": evidence.joint_graph_sha256[0],
        "cudnn_enabled_override": cudnn_enabled_override,
        "must_save_count": evidence.must_save_count if method == "sac" else 0,
        "partition_callback_count": evidence.partition_callback_count,
        "partition_fn": partition_api,
        "policy_hash": _SAC_POLICY_HASH if method == "sac" else None,
        "policy_save_ops": _SAC_SAVE_OP_NAMES if method == "sac" else (),
        "policy_source": _SAC_POLICY_SOURCE if method == "sac" else None,
        "prefer_recompute_count": evidence.prefer_recompute_count if method == "sac" else 0,
        "publication_block_count": len(publication_blocks),
        "publication_block_paths": publication_blocks,
        "recompute_count": evidence.recompute_count,
        "region_boundary": region_boundary,
        "restore_cpu_rng_after_call": method == "block_ac" if method != "all_save" else None,
        "state_dict_keys_preserved": True,
        "static_lifetime_input_indices_observed": evidence.static_lifetime_input_indices_observed,
        "static_region_installation": True,
        "training_flags_match": qualification.training_flags_match,
        "use_reentrant": False if method != "all_save" else None,
    }
    runtime_observations.update(
        {
            "joint_compile_count": evidence.joint_callback_count,
            "fw_compile_count": evidence.fw_callback_count,
            "bw_compile_count": evidence.bw_callback_count,
            "joint_graph_sha256": tuple(evidence.joint_graph_sha256),
            "fw_graph_sha256": tuple(evidence.fw_graph_sha256),
            "bw_graph_sha256": tuple(evidence.bw_graph_sha256),
            "compiler_callback_pid": os.getpid(),
            "cache_identity": stable_cache_identity,
            "cache_reuse_disabled": True,
        }
    )
    identity = RuntimeIdentity(
        method_id=method,
        status="ready",
        is_real=True,
        api=spec.api,
        policy=spec.policy,
        region_paths=checkpoint_regions,
        compiler_protocol=spec.compiler_protocol,
        fw_graph_sha256=fw_hash,
        bw_graph_sha256=bw_hash,
        model_sha256=model_sha256,
        executable_sha256=executable_sha256,
        provenance=tuple(sorted(provenance.items())),
    )
    return PreparedMethod(spec=spec, identity=identity, executable=executable)


compile_publication_method = prepare_publication_compiler


__all__ = [
    "FrozenSACPolicy",
    "PublicationBackend",
    "PublicationDefaultPartitionerFn",
    "PublicationExecutable",
    "PublicationMethod",
    "close_publication_executable",
    "compile_publication_method",
    "prepare_publication_compiler",
]
