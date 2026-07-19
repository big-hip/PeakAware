import pytest
import torch
from torch import nn

from peakaware import PeakAwareConfig
from peakaware.contracts import GuardSpec
from peakaware.runtime.executor import build_training_step_executor, validate_runtime_guards


def test_runtime_guards_validate_positional_and_keyword_tensors():
    args = (torch.ones(2, 3),)
    kwargs = {"mask": torch.ones(2, 3, dtype=torch.bool)}
    guards = (
        GuardSpec("arg0.shape", "(2, 3)"),
        GuardSpec("arg0.dtype", "torch.float32"),
        GuardSpec("arg0.device", "cpu"),
        GuardSpec("kw.mask.shape", "(2, 3)"),
        GuardSpec("kw.mask.dtype", "torch.bool"),
    )

    validate_runtime_guards(guards, args, kwargs)


def test_runtime_guards_reject_shape_drift():
    guards = (GuardSpec("arg0.shape", "(2, 3)"),)

    with pytest.raises(ValueError, match="runtime guard failed"):
        validate_runtime_guards(guards, (torch.ones(4, 3),), {})


def test_executor_rejects_guard_drift_before_zero_grad_or_step():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        guards=(GuardSpec("arg0.shape", "(2, 3)"),),
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    params_before = tuple(param.detach().clone() for param in model.parameters())
    grads_before = tuple(param.grad.detach().clone() for param in model.parameters())

    with pytest.raises(ValueError, match="runtime guard failed"):
        executor.step(torch.ones(4, 3))

    assert all(torch.equal(before, after) for before, after in zip(params_before, model.parameters()))
    assert all(torch.equal(before, param.grad) for before, param in zip(grads_before, model.parameters()))
