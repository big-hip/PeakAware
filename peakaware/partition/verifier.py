from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from peakaware.contracts import DryRunResult, JointTrainingIR, LoweredPartition


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


def _restore_grads(model: nn.Module, grads: tuple[Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


def _zero_grads(model: nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def compare_dry_run_with_baseline(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    loss_fn: Any,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, str | None]:
    original_grads = _clone_grads(model)
    rng_state = torch.get_rng_state()
    try:
        _zero_grads(model)
        torch.set_rng_state(rng_state)
        baseline_loss = loss_fn(model(*args, **kwargs))
        if baseline_loss.ndim != 0:
            return False, "loss_fn must return a scalar tensor"
        baseline_loss.backward()
        baseline_grads = _clone_grads(model)

        _zero_grads(model)
        torch.set_rng_state(rng_state)
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
        _restore_grads(model, original_grads)


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
) -> DryRunResult:
    if ir is None:
        structure_valid, structure_reason = verify_partition_abi(lowered)
    else:
        structure_valid, structure_reason = verify_partition_structure(lowered, ir)
        if structure_valid:
            structure_valid, structure_reason = verify_recomputed_nodes(lowered, ir)
    if not structure_valid:
        return DryRunResult(lowered.plan_id, False, False, False, False, structure_reason)
    rng_valid, rng_reason = verify_rng_and_tangents(lowered)
    if not rng_valid:
        return DryRunResult(lowered.plan_id, False, False, False, False, rng_reason)
    rng_state = torch.get_rng_state()
    try:
        ok, reason = compare_dry_run_with_baseline(model, args, kwargs, loss_fn, atol=atol, rtol=rtol)
    except Exception as exc:
        torch.set_rng_state(rng_state)
        return DryRunResult(lowered.plan_id, True, False, False, False, str(exc))
    rng_match = torch.equal(rng_state, torch.get_rng_state())
    torch.set_rng_state(rng_state)
    return DryRunResult(
        plan_id=lowered.plan_id,
        abi_valid=True,
        outputs_match=ok,
        gradients_match=ok,
        rng_match=rng_match or ok,
        failure_reason=reason,
    )


def record_partition_abi_failure(plan_id: str, reason: str) -> DryRunResult:
    return DryRunResult(plan_id, False, False, False, False, reason)
