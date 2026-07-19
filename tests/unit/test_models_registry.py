import pickle

import torch

from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, OptimizerSpec, TrainingRequest
from peakaware.ir import build_joint_ir
from peakaware.models import (
    TrainingTaskRegistry,
    build_bert_base_task,
    build_gpt2_task,
    build_resnet50_task,
    build_vit_b16_task,
)


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

    assert registry.names() == (
        "bert_base",
        "gpt2",
        "resnet50",
        "tiny_attention_w8_s4",
        "tiny_mlp_w8_d3",
        "tiny_residual_w8",
        "vit_b_16",
    )


def test_tiny_default_tasks_are_picklable_for_worker_isolation():
    registry = TrainingTaskRegistry.with_defaults()

    for name in ("tiny_attention_w8_s4", "tiny_mlp_w8_d3", "tiny_residual_w8"):
        task = registry.get(name)
        restored = pickle.loads(pickle.dumps(task))
        args, kwargs = restored.build_batch(2)
        output = restored.build_model()(*args, **kwargs)

        assert output.shape == (2, 1)


def test_main_model_task_specs_are_picklable_without_downloads():
    registry = TrainingTaskRegistry.with_defaults()

    for name in ("bert_base", "gpt2", "resnet50", "vit_b_16"):
        task = registry.get(name)
        restored = pickle.loads(pickle.dumps(task))
        args, kwargs = restored.build_batch(2)

        assert restored.name == name
        assert restored.workload is not None
        assert restored.workload.registry_key == name
        assert args[0].shape[0] == 2
        assert kwargs == {}


def test_main_model_display_names_match_actual_default_configurations():
    registry = TrainingTaskRegistry.with_defaults()

    assert registry.get("resnet50").workload.display_name == "ResNet-50"
    assert registry.get("vit_b_16").workload.display_name == "ViT-B/16"
    assert registry.get("bert_base").workload.display_name == "BERT-like-2L-64H"
    assert registry.get("gpt2").workload.display_name == "GPT2-like-2L-64H"


def test_main_model_tasks_build_fixed_shape_training_losses():
    torch.manual_seed(0)

    tasks = (
        build_resnet50_task(image_size=64, num_classes=7),
        build_vit_b16_task(image_size=32, num_classes=7),
        build_bert_base_task(
            sequence_length=8,
            vocab_size=101,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
        ),
        build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4),
    )

    for task in tasks:
        model = task.build_model()
        args, kwargs = task.build_batch(2)
        loss = task.loss_fn(model(*args, **kwargs))

        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_registered_mlp_and_attention_build_valid_ir():
    torch.manual_seed(0)

    for task_name in ("tiny_mlp_w8_d3", "tiny_attention_w8_s4"):
        ir, report = _capture_task(task_name)

        assert report.valid
        assert ir.values


def test_gpt2_like_task_builds_valid_ir_without_transformers_fx_capture():
    torch.manual_seed(0)
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
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
        optimizer_spec=OptimizerSpec("adamw", 1, sum(1 for _ in model.parameters()), 0, 0),
        hardware=HardwareSpec("cpu", False, None),
        request_key=task.name,
    )

    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)

    assert report.valid
    assert ir.values
    assert any(value.phase == "fw" and value.crosses_fw_bw for value in ir.values)
