from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint as checkpoint

from peakaware.models import TrainingTaskRegistry
from peakaware.runtime.measure import measure_training_step_phases


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _tensor_output(output: Any) -> torch.Tensor:
    return output if isinstance(output, torch.Tensor) else getattr(output, "logits")


def _clone_grads(model: torch.nn.Module) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if param.grad is None else param.grad.detach().clone() for param in model.parameters())


def _restore_grads(model: torch.nn.Module, grads: tuple[torch.Tensor | None, ...]) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = None if grad is None else grad.detach().clone()


def _forward_backward_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    executable: Any,
    loss_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    optimizer.zero_grad(set_to_none=True)
    output = executable(*args, **kwargs)
    loss = loss_fn(output)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    loss.backward()
    grads = _clone_grads(model)
    return _tensor_output(output).detach().clone(), grads


def _correctness_passed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    eager_executable: Any,
    sac_executable: Any,
    loss_fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    grad_state = _clone_grads(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore() -> None:
        model.load_state_dict(copy.deepcopy(model_state))
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        _restore_grads(model, grad_state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    restore()
    eager_output, eager_grads = _forward_backward_snapshot(model, optimizer, eager_executable, loss_fn, args, kwargs)
    restore()
    sac_output, sac_grads = _forward_backward_snapshot(model, optimizer, sac_executable, loss_fn, args, kwargs)
    restore()
    if not torch.allclose(eager_output, sac_output, atol=1e-5, rtol=1e-4):
        return False
    for eager_grad, sac_grad in zip(eager_grads, sac_grads):
        if eager_grad is None or sac_grad is None:
            if eager_grad is not sac_grad:
                return False
            continue
        if not torch.allclose(eager_grad, sac_grad, atol=1e-5, rtol=1e-4):
            return False
    return True


def run_sac_baseline(
    *,
    task_name: str,
    microbatch_size: int,
    device: str = "cpu",
    warmup_steps: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    if not hasattr(checkpoint, "create_selective_checkpoint_contexts"):
        return {
            "task_name": task_name,
            "status": "unavailable",
            "unavailable_reason": "torch.utils.checkpoint.create_selective_checkpoint_contexts is absent",
        }
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device}")
    task = TrainingTaskRegistry.with_defaults().get(task_name)
    torch.manual_seed(1337)
    model = task.build_model().to(resolved)
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(microbatch_size)
    args = _move_to_device(args, resolved)
    kwargs = _move_to_device(kwargs, resolved)
    context_fn = functools.partial(checkpoint.create_selective_checkpoint_contexts, [])
    eager_executable = model
    sac_executable = lambda *call_args, **call_kwargs: checkpoint.checkpoint(
        lambda *inner: model(*inner, **call_kwargs),
        *call_args,
        use_reentrant=False,
        context_fn=context_fn,
    )
    correctness = _correctness_passed(
        model,
        optimizer,
        eager_executable,
        sac_executable,
        task.loss_fn,
        args,
        kwargs,
    )
    eager_metrics = measure_training_step_phases(
        model,
        optimizer,
        eager_executable,
        task.loss_fn,
        args,
        kwargs,
        zero_grad_set_to_none=True,
        warmup_steps=warmup_steps,
        repeat_count=repeats,
    )
    sac_metrics = measure_training_step_phases(
        model,
        optimizer,
        sac_executable,
        task.loss_fn,
        args,
        kwargs,
        zero_grad_set_to_none=True,
        warmup_steps=warmup_steps,
        repeat_count=repeats,
    )
    eager_peak = int(eager_metrics["overall_peak_bytes"])
    sac_peak = int(sac_metrics["overall_peak_bytes"])
    eager_step = float(eager_metrics["step_us"])
    sac_step = float(sac_metrics["step_us"])
    return {
        "task_name": task_name,
        "microbatch_size": microbatch_size,
        "device": str(resolved),
        "status": "ok",
        "baseline_id": "pytorch_sac_prefer_recompute",
        "policy_provenance": {
            "api": "torch.utils.checkpoint.create_selective_checkpoint_contexts",
            "ops_to_save": [],
            "default_policy": "PREFER_RECOMPUTE",
        },
        "correctness_passed": correctness,
        "performance_result_usable": correctness,
        "correctness_note": None
        if correctness
        else "SAC output or gradients differ from eager; do not use this row as a performance baseline.",
        "eager_overall_peak_bytes": eager_peak,
        "sac_overall_peak_bytes": sac_peak,
        "peak_reduction_bytes": eager_peak - sac_peak,
        "eager_step_us": eager_step,
        "sac_step_us": sac_step,
        "samples_per_second_speedup_vs_eager": eager_step / max(sac_step, 1.0),
        "eager_metrics": eager_metrics,
        "sac_metrics": sac_metrics,
    }


def run_sac_baseline_matrix(
    *,
    task_names: tuple[str, ...],
    microbatch_sizes: tuple[int, ...],
    device: str = "cpu",
    warmup_steps: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    rows = []
    for task_name in task_names:
        for microbatch_size in microbatch_sizes:
            try:
                row = run_sac_baseline(
                    task_name=task_name,
                    microbatch_size=microbatch_size,
                    device=device,
                    warmup_steps=warmup_steps,
                    repeats=repeats,
                )
            except Exception as exc:
                row = {
                    "task_name": task_name,
                    "microbatch_size": microbatch_size,
                    "device": device,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "performance_result_usable": False,
                }
            row.setdefault("task_name", task_name)
            row.setdefault("microbatch_size", microbatch_size)
            row.setdefault("device", device)
            rows.append(row)
    usable = [row for row in rows if row.get("status") == "ok" and row.get("performance_result_usable")]
    return {
        "baseline_id": "pytorch_sac_prefer_recompute",
        "task_names": list(task_names),
        "microbatch_sizes": list(microbatch_sizes),
        "device": device,
        "measurement_warmup_steps": warmup_steps,
        "measurement_repeats": repeats,
        "row_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "usable_count": len(usable),
        "unusable_count": len(rows) - len(usable),
        "mean_peak_reduction_bytes": None
        if not usable
        else sum(float(row["peak_reduction_bytes"]) for row in usable) / len(usable),
        "mean_samples_per_second_speedup_vs_eager": None
        if not usable
        else sum(float(row["samples_per_second_speedup_vs_eager"]) for row in usable) / len(usable),
        "rows": rows,
    }


def write_sac_baseline_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
