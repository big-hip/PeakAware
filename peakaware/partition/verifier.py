from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils import _pytree

from peakaware.contracts import DryRunResult, JointTrainingIR, LoweredPartition


class PartitionReplayUnsupported(RuntimeError):
    pass


def verify_partition_abi(lowered: LoweredPartition) -> tuple[bool, str | None]:
    abi = lowered.partition_abi
    if len(abi.fw_output_value_ids) != len(abi.bw_placeholder_value_ids):
        return False, "FW outputs and BW placeholders have different lengths"
    if tuple(abi.fw_output_value_ids) != tuple(abi.bw_placeholder_value_ids):
        return False, "FW output value ids do not match BW placeholder value ids in M0 ABI"
    return True, None


def verify_partition_structure(lowered: LoweredPartition, ir: JointTrainingIR) -> tuple[bool, str | None]:
    abi_valid, abi_reason = verify_partition_abi(lowered)
    if not abi_valid:
        return False, abi_reason
    if len(list(lowered.fw_graph.graph.nodes)) == 0:
        return False, "FW graph is empty"
    if len(list(lowered.bw_graph.graph.nodes)) == 0:
        return False, "BW graph is empty"
    known_value_ids = {value.id for value in ir.values}
    abi_value_ids = (
        tuple(lowered.partition_abi.fw_output_value_ids)
        + tuple(lowered.partition_abi.bw_placeholder_value_ids)
        + tuple(lowered.partition_abi.tangent_value_ids)
        + tuple(lowered.partition_abi.rng_state_value_ids)
    )
    unknown = sorted(value_id for value_id in abi_value_ids if value_id not in known_value_ids)
    if unknown:
        return False, f"partition ABI references unknown IR value ids: {unknown}"
    if len(set(lowered.partition_abi.fw_output_value_ids)) != len(lowered.partition_abi.fw_output_value_ids):
        return False, "FW output value ids contain duplicates"
    return True, None


def verify_recomputed_nodes(lowered: LoweredPartition, ir: JointTrainingIR) -> tuple[bool, str | None]:
    saved_value_ids = set(lowered.partition_abi.fw_output_value_ids)
    illegally_dropped = sorted(
        value.id
        for value in ir.values
        if value.phase == "fw"
        and value.crosses_fw_bw
        and value.id not in saved_value_ids
        and (not value.recomputable or value.mandatory_save_reason is not None)
    )
    if illegally_dropped:
        return False, f"partition dropped non-recomputable or mandatory FW values: {illegally_dropped}"
    return True, None


def verify_rng_and_tangents(lowered: LoweredPartition) -> tuple[bool, str | None]:
    if lowered.partition_abi.tangent_value_ids:
        return False, "M0 eager partition does not model tangent ABI"
    missing_rng_outputs = sorted(
        set(lowered.partition_abi.rng_state_value_ids) - set(lowered.partition_abi.fw_output_value_ids)
    )
    if missing_rng_outputs:
        return False, f"RNG state value ids must be forwarded to BW placeholders: {missing_rng_outputs}"
    return True, None


def _clone_grads(model: nn.Module) -> tuple[Tensor | None, ...]:
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in model.parameters())


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _restore_grads(model: nn.Module, grads: tuple[Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


def _zero_grads(model: nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _cuda_rng_state() -> list[Tensor] | None:
    return torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _restore_rng(cpu_rng: Tensor, cuda_rng: list[Tensor] | None) -> None:
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


def _compare_nested_state(left: Any, right: Any, path: str, *, atol: float, rtol: float) -> str | None:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return None if torch.allclose(left, right, atol=atol, rtol=rtol) else f"state mismatch at {path}"
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return f"state keys mismatch at {path}"
        for key in left:
            reason = _compare_nested_state(left[key], right[key], f"{path}.{key}", atol=atol, rtol=rtol)
            if reason is not None:
                return reason
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return f"state length mismatch at {path}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            reason = _compare_nested_state(left_item, right_item, f"{path}[{index}]", atol=atol, rtol=rtol)
            if reason is not None:
                return reason
        return None
    return None if left == right else f"state mismatch at {path}"


@contextmanager
def _deterministic_backend_for_comparison() -> Any:
    if not torch.cuda.is_available():
        yield
        return
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic


def compare_dry_run_with_baseline(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loss_fn: Any,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, str | None]:
    original_model_state = _clone_model_state(model)
    original_grads = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_state()

    def restore_all() -> None:
        model.load_state_dict(original_model_state)
        _restore_grads(model, original_grads)
        _restore_rng(cpu_rng, cuda_rng)

    try:
        with _deterministic_backend_for_comparison():
            restore_all()
            _zero_grads(model)
            baseline_loss = loss_fn(model(*args, **kwargs))
            if baseline_loss.ndim != 0:
                return False, "loss_fn must return a scalar tensor"
            baseline_loss.backward()
            baseline_grads = _clone_grads(model)

            restore_all()
            _zero_grads(model)
            candidate_loss = loss_fn(model(*args, **kwargs))
            candidate_loss.backward()
            candidate_grads = _clone_grads(model)

        if not torch.allclose(baseline_loss.detach(), candidate_loss.detach(), atol=atol, rtol=rtol):
            return False, "loss mismatch"
        for index, (left, right) in enumerate(zip(baseline_grads, candidate_grads)):
            if left is None and right is None:
                continue
            if left is None or right is None:
                return False, f"gradient presence mismatch at parameter {index}"
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                return False, f"gradient mismatch at parameter {index}"
        return True, None
    finally:
        restore_all()


def _placeholder_names(graph: torch.fx.GraphModule) -> tuple[str, ...]:
    return tuple(str(node.target) for node in graph.graph.nodes if node.op == "placeholder")


def _aot_partition_replay_supported(lowered: LoweredPartition, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    flat_args, _ = _pytree.tree_flatten(args)
    flat_kwargs, _ = _pytree.tree_flatten(kwargs)
    if any(not isinstance(arg, Tensor) for arg in flat_args):
        return False
    if any(not isinstance(value, Tensor) for value in flat_kwargs):
        return False
    fw_placeholders = _placeholder_names(lowered.fw_graph)
    bw_placeholders = _placeholder_names(lowered.bw_graph)
    return (
        bool(fw_placeholders)
        and bool(bw_placeholders)
        and all(name.startswith("primals_") for name in fw_placeholders)
        and any(name.startswith("tangents_") for name in bw_placeholders)
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _loss_input_from_user_outputs(outputs: tuple[Any, ...], output_tree_spec: Any | None = None) -> Any:
    if output_tree_spec is not None:
        return _pytree.tree_unflatten(list(outputs), output_tree_spec)
    if len(outputs) != 1:
        return outputs
    return outputs[0]


def _detach_for_tangent(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().requires_grad_(value.is_floating_point())
    if isinstance(value, tuple):
        return tuple(_detach_for_tangent(item) for item in value)
    if isinstance(value, list):
        return [_detach_for_tangent(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_for_tangent(item) for key, item in value.items()}
    raise PartitionReplayUnsupported("lowered partition replay only supports tensor outputs")


def _flatten_tensors(value: Any) -> tuple[Tensor, ...]:
    flat_values, _ = _pytree.tree_flatten(value)
    if any(not isinstance(item, Tensor) for item in flat_values):
        raise PartitionReplayUnsupported("lowered partition replay only supports tensor outputs")
    return tuple(flat_values)


def compare_lowered_partition_with_baseline(
    lowered: LoweredPartition,
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loss_fn: Any,
    *,
    num_fwd_outputs: int = 1,
    kwarg_names: tuple[str, ...] | None = None,
    output_tree_spec: Any | None = None,
    arg_tree_specs: tuple[Any, ...] = (),
    kwarg_tree_specs: tuple[tuple[str, Any], ...] = (),
    atol: float,
    rtol: float,
) -> tuple[bool, str | None]:
    if not _aot_partition_replay_supported(lowered, args, kwargs):
        raise PartitionReplayUnsupported("lowered partition replay requires AOT primals/tangents ABI")
    if num_fwd_outputs < 1:
        raise PartitionReplayUnsupported("lowered partition replay requires at least one FW user output")

    original_model_state = _clone_model_state(model)
    original_grads = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_state()
    params = tuple(model.parameters())
    state_inputs = params + tuple(model.buffers())
    if arg_tree_specs:
        if len(args) != len(arg_tree_specs):
            raise PartitionReplayUnsupported(
                "lowered partition replay args must match captured args: "
                f"expected {len(arg_tree_specs)}, got {len(args)}"
            )
        flat_args: list[Any] = []
        for arg, expected_spec in zip(args, arg_tree_specs):
            flat_values, actual_spec = _pytree.tree_flatten(arg)
            if actual_spec != expected_spec:
                raise PartitionReplayUnsupported("lowered partition replay arg pytree structure changed")
            flat_args.extend(flat_values)
    else:
        flat_args = list(args)
    ordered_kwarg_names = (
        tuple(name for name, _ in kwarg_tree_specs)
        if kwarg_tree_specs
        else (tuple(kwargs) if kwarg_names is None else kwarg_names)
    )
    if set(kwargs) != set(ordered_kwarg_names):
        raise PartitionReplayUnsupported(
            "lowered partition replay kwargs must match captured kwargs: "
            f"expected {sorted(ordered_kwarg_names)}, got {sorted(kwargs)}"
        )
    if kwarg_tree_specs:
        flat_kwargs: list[Any] = []
        for name, expected_spec in kwarg_tree_specs:
            flat_values, actual_spec = _pytree.tree_flatten(kwargs[name])
            if actual_spec != expected_spec:
                raise PartitionReplayUnsupported("lowered partition replay kwarg pytree structure changed")
            flat_kwargs.extend(flat_values)
    else:
        flat_kwargs = list(kwargs[name] for name in ordered_kwarg_names)
    if any(not isinstance(value, Tensor) for value in tuple(flat_args) + tuple(flat_kwargs)):
        raise PartitionReplayUnsupported("lowered partition replay only supports tensor inputs")
    flat_user_inputs = tuple(flat_args) + tuple(flat_kwargs)

    def restore_all() -> None:
        model.load_state_dict(original_model_state)
        _restore_grads(model, original_grads)
        _restore_rng(cpu_rng, cuda_rng)

    try:
        with _deterministic_backend_for_comparison():
            restore_all()
            _zero_grads(model)
            baseline_loss = loss_fn(model(*args, **kwargs))
            if baseline_loss.ndim != 0:
                return False, "loss_fn must return a scalar tensor"
            baseline_loss.backward()
            baseline_grads = _clone_grads(model)

            restore_all()
            _zero_grads(model)
            fw_outputs = _as_tuple(lowered.fw_graph(*(state_inputs + flat_user_inputs)))
            if len(fw_outputs) < num_fwd_outputs:
                return False, "lowered FW graph returned fewer user outputs than expected"
            user_outputs = fw_outputs[:num_fwd_outputs]
            saved_for_bw = fw_outputs[num_fwd_outputs:]
            tangent_outputs = _detach_for_tangent(_loss_input_from_user_outputs(user_outputs, output_tree_spec))
            candidate_loss = loss_fn(tangent_outputs)
            if candidate_loss.ndim != 0:
                return False, "loss_fn must return a scalar tensor"
            tangent_tensors = _flatten_tensors(tangent_outputs)
            tangents = torch.autograd.grad(candidate_loss, tangent_tensors, allow_unused=False)
            bw_outputs = _as_tuple(lowered.bw_graph(*(saved_for_bw + tuple(tangents))))

        if not torch.allclose(baseline_loss.detach(), candidate_loss.detach(), atol=atol, rtol=rtol):
            return False, "loss mismatch"
        if len(bw_outputs) < len(params):
            return False, "lowered BW graph returned fewer gradients than model parameters"
        for index, (expected, actual) in enumerate(zip(baseline_grads, bw_outputs[: len(params)])):
            if expected is None and actual is None:
                continue
            if expected is None or actual is None:
                return False, f"gradient presence mismatch at parameter {index}"
            if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
                return False, f"lowered gradient mismatch at parameter {index}"
        return True, None
    finally:
        restore_all()


def compare_multistep_training_with_baseline(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    *,
    step_count: int,
    atol: float,
    rtol: float,
    zero_grad_set_to_none: bool = True,
) -> tuple[bool, str | None]:
    original_model_state = _clone_model_state(model)
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
    original_grads = _clone_grads(model)
    original_cpu_rng = torch.get_rng_state()
    original_cuda_rng = _cuda_rng_state()

    step_count = max(int(step_count), 1)

    def restore_all() -> None:
        model.load_state_dict(original_model_state)
        optimizer.load_state_dict(copy.deepcopy(original_optimizer_state))
        _restore_grads(model, original_grads)
        _restore_rng(original_cpu_rng, original_cuda_rng)

    def run_steps() -> tuple[tuple[Tensor, ...], dict[str, Tensor], dict[str, Any]]:
        losses: list[Tensor] = []
        for _ in range(step_count):
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            loss = loss_fn(model(*args, **kwargs))
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
        return tuple(losses), _clone_model_state(model), copy.deepcopy(optimizer.state_dict())

    try:
        restore_all()
        baseline_losses, baseline_state, baseline_optimizer_state = run_steps()
        restore_all()
        candidate_losses, candidate_state, candidate_optimizer_state = run_steps()

        for index, (left, right) in enumerate(zip(baseline_losses, candidate_losses)):
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                return False, f"loss mismatch at step {index}"
        for name, left in baseline_state.items():
            right = candidate_state[name]
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                return False, f"parameter or buffer mismatch after step {step_count}: {name}"
        optimizer_reason = _compare_nested_state(
            baseline_optimizer_state,
            candidate_optimizer_state,
            "optimizer",
            atol=atol,
            rtol=rtol,
        )
        if optimizer_reason is not None:
            return False, optimizer_reason
        return True, None
    finally:
        restore_all()


def run_aot_eager_dry_run(
    lowered: LoweredPartition,
    *,
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loss_fn: Any,
    atol: float,
    rtol: float,
    ir: JointTrainingIR | None = None,
    num_fwd_outputs: int = 1,
    kwarg_names: tuple[str, ...] | None = None,
    output_tree_spec: Any | None = None,
    arg_tree_specs: tuple[Any, ...] = (),
    kwarg_tree_specs: tuple[tuple[str, Any], ...] = (),
) -> DryRunResult:
    if ir is None:
        structure_valid, structure_reason = verify_partition_abi(lowered)
    else:
        structure_valid, structure_reason = verify_partition_structure(lowered, ir)
        if structure_valid:
            structure_valid, structure_reason = verify_recomputed_nodes(lowered, ir)
    if not structure_valid:
        return DryRunResult(lowered.plan_id, False, False, False, False, structure_reason, "not_run")
    rng_valid, rng_reason = verify_rng_and_tangents(lowered)
    if not rng_valid:
        return DryRunResult(lowered.plan_id, False, False, False, False, rng_reason, "not_run")
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_state()
    replay_mode = "lowered_aot"
    try:
        try:
            ok, reason = compare_lowered_partition_with_baseline(
                lowered,
                model,
                args,
                kwargs,
                loss_fn,
                num_fwd_outputs=num_fwd_outputs,
                kwarg_names=kwarg_names,
                output_tree_spec=output_tree_spec,
                arg_tree_specs=arg_tree_specs,
                kwarg_tree_specs=kwarg_tree_specs,
                atol=atol,
                rtol=rtol,
            )
        except PartitionReplayUnsupported:
            replay_mode = "eager_baseline"
            ok, reason = compare_dry_run_with_baseline(model, args, kwargs, loss_fn, atol=atol, rtol=rtol)
    except Exception as exc:
        _restore_rng(cpu_rng, cuda_rng)
        return DryRunResult(lowered.plan_id, True, False, False, False, str(exc), replay_mode)
    rng_match = torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        rng_match = rng_match and all(
            torch.equal(expected, actual)
            for expected, actual in zip(cuda_rng, torch.cuda.get_rng_state_all())
        )
    _restore_rng(cpu_rng, cuda_rng)
    return DryRunResult(
        plan_id=lowered.plan_id,
        abi_valid=True,
        outputs_match=ok,
        gradients_match=ok,
        rng_match=rng_match or ok,
        failure_reason=reason,
        replay_mode=replay_mode,
    )


def record_partition_abi_failure(plan_id: str, reason: str) -> DryRunResult:
    return DryRunResult(plan_id, False, False, False, False, reason, "not_run")
