import pickle

import torch

from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, OptimizerSpec, TrainingRequest
from peakaware.ir import build_joint_ir
from peakaware.models import TrainingTaskRegistry


def _capture_task(task_name: str):
    registry = TrainingTaskRegistry.with_defaults()
    task = registry.get(task_name)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(),
        optimizer_spec=OptimizerSpec("sgd", 1, sum(1 for _ in model.parameters()), 0, 0),
        hardware=HardwareSpec("cpu", False, None),
        request_key=task.name,
    )
    capture = capture_joint_graph(request)
    return build_joint_ir(capture)


def test_default_registry_includes_explanatory_models():
    registry = TrainingTaskRegistry.with_defaults()

    assert registry.names() == ("tiny_attention_w8_s4", "tiny_mlp_w8_d3", "tiny_residual_w8")


def test_default_tasks_are_picklable_for_worker_isolation():
    registry = TrainingTaskRegistry.with_defaults()

    for name in registry.names():
        task = registry.get(name)
        restored = pickle.loads(pickle.dumps(task))
        args, kwargs = restored.build_batch(2)
        output = restored.build_model()(*args, **kwargs)

        assert output.shape == (2, 1)


def test_registered_mlp_and_attention_build_valid_ir():
    torch.manual_seed(0)

    for task_name in ("tiny_mlp_w8_d3", "tiny_attention_w8_s4"):
        ir, report = _capture_task(task_name)

        assert report.valid
        assert ir.values
