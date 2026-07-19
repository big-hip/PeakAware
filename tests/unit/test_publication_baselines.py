import copy
import json
import threading

import pytest
import torch
from torch import nn
from torch.utils import checkpoint

import peakaware.publication.baselines as baseline_module
from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import HardwareSpec, TrainingRequest
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.publication.baselines import (
    RuntimeIdentity,
    UnsupportedMethodError,
    make_sac_policy,
    prepare_aot_min_cut,
    prepare_block_activation_checkpoint,
    prepare_selective_activation_checkpoint,
    resolve_block_regions,
)


def _capture_mlp():
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.GELU(),
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    )
    args = (torch.randn(8, 16),)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda output: output.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="publication-min-cut-test",
    )
    return model, args, request.loss_fn, capture_joint_graph(request)


def _capture_model(model, args, kwargs, loss_fn):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="publication-min-cut-abi-test",
    )
    return capture_joint_graph(request)


def _region_prepare_kwargs(shape=(2, 4)):
    return {
        "example_args": (torch.randn(*shape),),
        "example_kwargs": {},
        "loss_fn": lambda output: output.pow(2).mean(),
    }


def test_native_min_cut_uses_real_partition_outputs_and_restores_config():
    from torch._functorch import config as functorch_config

    model, args, loss_fn, capture = _capture_mlp()
    original_budget = functorch_config.activation_memory_budget
    low = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        activation_memory_budget=0.0,
    ).require_supported()
    high = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        activation_memory_budget=1.0,
    ).require_supported()

    assert functorch_config.activation_memory_budget == original_budget
    assert low.fw_graph is not None and low.bw_graph is not None
    assert high.fw_graph is not None and high.bw_graph is not None
    assert low.identity.fw_residual_names != high.identity.fw_residual_names
    assert low.identity.fw_graph_sha256 != high.identity.fw_graph_sha256
    assert low.identity.bw_graph_sha256 != high.identity.bw_graph_sha256
    assert low.identity.model_sha256
    assert low.identity.executable_sha256
    assert low.identity.bw_placeholder_names
    assert dict(low.identity.provenance)["api"].endswith("min_cut_rematerialization_partition")
    assert dict(low.identity.provenance)["memory_budget_ratio"] == 0.0
    assert dict(low.identity.provenance)["buffers_match"] is True
    assert dict(low.identity.provenance)["cpu_rng_match"] is True
    assert dict(low.identity.provenance)["cuda_rng_match"] is True
    assert low.identity.compiler_protocol == "aot_lowered_graphmodule_eager"
    assert dict(low.identity.provenance)["partitioner_cost_model"] == "inductor"

    model.zero_grad(set_to_none=True)
    eager_output = model(*args)
    eager_loss = loss_fn(eager_output)
    eager_loss.backward()
    eager_grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual_output = high.executable(*args)
    actual_loss = loss_fn(actual_output)
    actual_loss.backward()

    assert torch.allclose(actual_output, eager_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_loss, eager_loss, atol=1e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.grad, eager_grads[name], atol=1e-6, rtol=1e-5)


def test_native_min_cut_training_executable_supports_buffers_kwargs_and_nested_outputs():
    class NestedKwargModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 2)
            self.right = nn.Linear(4, 2)
            self.register_buffer("offset", torch.arange(4, dtype=torch.float32))

        def forward(self, x, *, scale):
            hidden = (x + self.offset) * scale
            return {"left": self.left(hidden), "right": (torch.relu(self.right(hidden)),)}

    def loss_fn(outputs):
        return outputs["left"].pow(2).mean() + outputs["right"][0].pow(2).mean()

    model = NestedKwargModel()
    args = (torch.randn(3, 4),)
    kwargs = {"scale": torch.full((3, 4), 0.5)}
    capture = _capture_model(model, args, kwargs, loss_fn)
    prepared = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=loss_fn,
        activation_memory_budget=1.0,
    ).require_supported()

    outputs = prepared.executable(*args, **kwargs)
    loss = loss_fn(outputs)
    loss.backward()

    assert set(outputs) == {"left", "right"}
    assert isinstance(outputs["right"], tuple)
    assert loss.ndim == 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_native_min_cut_bad_training_executable_is_unsupported(monkeypatch):
    model, args, loss_fn, capture = _capture_mlp()
    monkeypatch.setattr(
        baseline_module,
        "build_aot_partition_executable",
        lambda *unused_args, **unused_kwargs: lambda *args, **kwargs: torch.zeros((), requires_grad=True),
    )

    prepared = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        activation_memory_budget=1.0,
    )

    assert not prepared.supported
    assert prepared.executable is None
    assert prepared.identity.fw_graph_sha256
    assert prepared.identity.bw_graph_sha256
    assert "outputs do not match eager" in prepared.identity.fallback_reason
    with pytest.raises(UnsupportedMethodError):
        prepared.require_supported()


def test_native_min_cut_extra_random_draw_is_unsupported(monkeypatch):
    model, args, loss_fn, capture = _capture_mlp()
    real_builder = baseline_module.build_aot_partition_executable

    def rng_changing_builder(*builder_args, **builder_kwargs):
        executable = real_builder(*builder_args, **builder_kwargs)

        def rng_changing_executable(*call_args, **call_kwargs):
            output = executable(*call_args, **call_kwargs)
            torch.rand(())
            return output

        return rng_changing_executable

    monkeypatch.setattr(baseline_module, "build_aot_partition_executable", rng_changing_builder)
    prepared = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        activation_memory_budget=1.0,
    )

    assert not prepared.supported
    assert "CPU RNG" in prepared.identity.fallback_reason


def test_native_min_cut_rejects_unintegrated_inductor_execution_identity():
    model, args, loss_fn, capture = _capture_mlp()
    prepared = prepare_aot_min_cut(
        model,
        capture,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        activation_memory_budget=1.0,
        execution_backend="inductor",
    )

    assert not prepared.supported
    assert prepared.identity.compiler_protocol == "aot_lowered_graphmodule_eager"
    assert "not integrated" in prepared.identity.fallback_reason


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_native_min_cut_rejects_invalid_ratio(ratio):
    model, args, loss_fn, capture = _capture_mlp()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_aot_min_cut(
            model,
            capture,
            example_args=args,
            example_kwargs={},
            loss_fn=loss_fn,
            activation_memory_budget=ratio,
        )


class _FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return x


class _FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.Sequential(nn.Linear(4, 4), nn.ReLU())


class _FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


class _FakeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Sequential(nn.Linear(4, 4), nn.ReLU())]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _BatchNormResNet(nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(1, 5):
            setattr(
                self,
                f"layer{index}",
                nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU()),
            )

    def forward(self, x):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return x


class _ConvResNet(nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(1, 5):
            setattr(self, f"layer{index}", nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.ReLU()))

    def forward(self, x):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return x


class _ConfigurableGPT(nn.Module):
    def __init__(self, extra_relu):
        super().__init__()
        layers = [nn.Linear(4, 4), nn.ReLU()]
        if extra_relu:
            layers.append(nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(*layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _DropoutGPT(nn.Module):
    def __init__(self, probability=0.5):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.Dropout(probability), nn.ReLU())]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _ActivationGPT(nn.Module):
    def __init__(self, use_gelu):
        super().__init__()
        activation = nn.GELU() if use_gelu else nn.ReLU()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), activation)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _SelfEvalGPT(_FakeGPT):
    def forward(self, x):
        self.eval()
        return super().forward(x)


class _ScaledGPT(_FakeGPT):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return super().forward(x) * self.scale


class _UnserializableAttributeGPT(_FakeGPT):
    def __init__(self):
        super().__init__()
        self.behavior_state = object()


_BLOCKING_STATES = {}


class _BlockingGPT(_FakeGPT):

    def forward(self, x):
        state = _BLOCKING_STATES.get(id(self))
        if state is not None:
            state["entered"].set()
            if not state["release"].wait(timeout=5):
                raise RuntimeError("blocking test timed out")
        return super().forward(x)


def _enable_blocking(model):
    state = {"entered": threading.Event(), "release": threading.Event()}
    _BLOCKING_STATES[id(model)] = state
    return state


@pytest.mark.parametrize(
    ("model", "registry_key", "expected"),
    [
        (_FakeResNet(), "resnet50", ("layer1", "layer2", "layer3", "layer4")),
        (_FakeViT(), "vit_b_16", ("encoder.layers",)),
        (_FakeBert(), "bert_base", ("bert.encoder.layer",)),
        (_FakeGPT(), "gpt2", ("blocks",)),
    ],
)
def test_resolve_block_regions_validates_architecture_paths(model, registry_key, expected):
    assert resolve_block_regions(model, registry_key) == expected


def test_block_checkpoint_wraps_only_resolved_non_reentrant_regions(monkeypatch):
    calls = []
    real_checkpoint = checkpoint.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs["use_reentrant"])
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "checkpoint", recording_checkpoint)
    prepared = prepare_block_activation_checkpoint(
        _FakeGPT(),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    result = prepared.executable(torch.randn(2, 4, requires_grad=True))
    result.sum().backward()

    assert prepared.identity.region_paths == ("blocks.0", "blocks.1")
    assert calls and set(calls) == {False}
    provenance = dict(prepared.identity.provenance)
    assert provenance["checkpoint_call_count"] > 0
    assert provenance["recompute_count"] > 0
    assert provenance["buffers_match"] is True
    assert provenance["cpu_rng_match"] is True
    assert provenance["cuda_rng_match"] is True


def test_sequential_stage_is_kept_as_one_frozen_block_region():
    prepared = prepare_block_activation_checkpoint(
        _FakeResNet(),
        "resnet50",
        **_region_prepare_kwargs(),
    ).require_supported()

    assert prepared.identity.region_paths == ("layer1", "layer2", "layer3", "layer4")


@pytest.mark.parametrize(
    "prepare_method",
    [prepare_block_activation_checkpoint, prepare_selective_activation_checkpoint],
)
def test_region_checkpoint_recompute_restores_batchnorm_buffers(prepare_method):
    candidate = _BatchNormResNet()
    reference = copy.deepcopy(candidate)
    kwargs = _region_prepare_kwargs(shape=(4, 4))
    prepared = prepare_method(candidate, "resnet50", **kwargs).require_supported()

    reference_output = reference(*kwargs["example_args"])
    kwargs["loss_fn"](reference_output).backward()
    actual_output = prepared.executable(*kwargs["example_args"])
    kwargs["loss_fn"](actual_output).backward()

    expected_buffers = dict(reference.named_buffers())
    actual_buffers = dict(candidate.named_buffers())
    assert expected_buffers.keys() == actual_buffers.keys()
    for name in expected_buffers:
        assert torch.equal(actual_buffers[name], expected_buffers[name]), name


@pytest.mark.parametrize(
    "prepare_method",
    [prepare_block_activation_checkpoint, prepare_selective_activation_checkpoint],
)
def test_region_checkpoint_dropout_preserves_rng_semantics(prepare_method):
    prepared = prepare_method(_DropoutGPT(), "gpt2", **_region_prepare_kwargs()).require_supported()
    provenance = dict(prepared.identity.provenance)

    assert provenance["cpu_rng_match"] is True
    assert provenance["cuda_rng_match"] is True


def test_prepare_restores_every_module_training_flag_after_forward_mutation():
    model = _SelfEvalGPT()
    model.train()
    model.blocks[0].eval()
    before = tuple((name, module.training) for name, module in model.named_modules())
    prepared = prepare_block_activation_checkpoint(model, "gpt2", **_region_prepare_kwargs()).require_supported()
    after = tuple((name, module.training) for name, module in model.named_modules())

    assert after == before
    assert dict(prepared.identity.provenance)["training_flags_match"] is True


def test_region_forward_instance_state_is_restored_exactly():
    model = _FakeGPT()
    model.blocks[0].forward = model.blocks[0].forward
    regions = tuple(model.get_submodule(path) for path in ("blocks.0", "blocks.1"))
    before = tuple({key: id(value) for key, value in region.__dict__.items()} for region in regions)
    prepared = prepare_block_activation_checkpoint(model, "gpt2", **_region_prepare_kwargs()).require_supported()

    output = prepared.executable(torch.randn(2, 4))
    output.sum().backward()
    after = tuple({key: id(value) for key, value in region.__dict__.items()} for region in regions)

    assert after == before


def test_region_checkpoint_rejects_concurrent_execution():
    model = _BlockingGPT()
    first = prepare_block_activation_checkpoint(model, "gpt2", **_region_prepare_kwargs()).require_supported()
    second = prepare_block_activation_checkpoint(model, "gpt2", **_region_prepare_kwargs()).require_supported()
    state = _enable_blocking(model)
    failures = []

    def run_first_execution():
        try:
            first.executable(torch.randn(2, 4))
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_first_execution)
    worker.start()
    assert state["entered"].wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="concurrent or reentrant"):
            second.executable(torch.randn(2, 4))
    finally:
        state["release"].set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    _BLOCKING_STATES.pop(id(model), None)
    second.executable(torch.randn(2, 4))


def test_region_checkpoint_allows_different_models_to_execute_concurrently():
    first_model = _BlockingGPT()
    second_model = _BlockingGPT()
    first = prepare_block_activation_checkpoint(
        first_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    second = prepare_block_activation_checkpoint(
        second_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    first_state = _enable_blocking(first_model)
    second_state = _enable_blocking(second_model)
    failures = []

    def run(executable):
        try:
            executable(torch.randn(2, 4))
        except Exception as exc:
            failures.append(exc)

    workers = [
        threading.Thread(target=run, args=(first.executable,)),
        threading.Thread(target=run, args=(second.executable,)),
    ]
    for worker in workers:
        worker.start()
    assert first_state["entered"].wait(timeout=5)
    assert second_state["entered"].wait(timeout=5)
    first_state["release"].set()
    second_state["release"].set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert failures == []
    _BLOCKING_STATES.pop(id(first_model), None)
    _BLOCKING_STATES.pop(id(second_model), None)


def test_same_region_paths_with_different_module_trees_have_different_identity():
    first = prepare_block_activation_checkpoint(
        _ConfigurableGPT(extra_relu=False),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    second = prepare_block_activation_checkpoint(
        _ConfigurableGPT(extra_relu=True),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()

    assert first.identity.region_paths == second.identity.region_paths
    assert first.identity.model_sha256 != second.identity.model_sha256
    assert first.identity.executable_sha256 != second.identity.executable_sha256


def test_behavior_digest_tracks_stable_module_configuration():
    original_model = _DropoutGPT(probability=0.1)
    copied_model = copy.deepcopy(original_model)
    high_dropout_model = _DropoutGPT(probability=0.9)
    original = prepare_block_activation_checkpoint(
        original_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    copied = prepare_block_activation_checkpoint(
        copied_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    high_dropout = prepare_block_activation_checkpoint(
        high_dropout_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    relu = prepare_block_activation_checkpoint(
        _ActivationGPT(use_gelu=False),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    gelu = prepare_block_activation_checkpoint(
        _ActivationGPT(use_gelu=True),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()

    assert original.identity.model_sha256 == copied.identity.model_sha256
    assert original.identity.executable_sha256 == copied.identity.executable_sha256
    assert original.identity.model_sha256 != high_dropout.identity.model_sha256
    assert relu.identity.model_sha256 != gelu.identity.model_sha256


def test_behavior_digest_tracks_custom_instance_attributes_stably():
    scale_one_model = _ScaledGPT(scale=1)
    copied_model = copy.deepcopy(scale_one_model)
    scale_two_model = _ScaledGPT(scale=2)
    scale_one = prepare_block_activation_checkpoint(
        scale_one_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    copied = prepare_block_activation_checkpoint(
        copied_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    scale_two = prepare_block_activation_checkpoint(
        scale_two_model,
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()

    assert scale_one.identity.model_sha256 == copied.identity.model_sha256
    assert scale_one.identity.executable_sha256 == copied.identity.executable_sha256
    assert scale_one.identity.model_sha256 != scale_two.identity.model_sha256


def test_unserializable_behavior_attribute_fails_closed_without_address_repr():
    prepared = prepare_block_activation_checkpoint(
        _UnserializableAttributeGPT(),
        "gpt2",
        **_region_prepare_kwargs(),
    )

    assert not prepared.supported
    assert prepared.executable is None
    assert "stable model digest unavailable" in prepared.identity.fallback_reason
    assert "builtins.object" in prepared.identity.fallback_reason
    assert "0x" not in prepared.identity.fallback_reason


def test_sac_policy_has_explicit_save_and_recompute_decisions():
    decisions = {}
    policy = make_sac_policy(decisions)

    assert policy(None, torch.ops.aten.mm.default) is checkpoint.CheckpointPolicy.MUST_SAVE
    assert policy(None, torch.ops.aten.relu.default) is checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    assert decisions == {"MUST_SAVE": 1, "PREFER_RECOMPUTE": 1}


def test_sac_adapter_records_nonempty_policy_and_region():
    prepared = prepare_selective_activation_checkpoint(
        _FakeGPT(),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()
    provenance = dict(prepared.identity.provenance)

    assert prepared.identity.region_paths == ("blocks.0", "blocks.1")
    assert provenance["must_save_count"] > 0
    assert provenance["prefer_recompute_count"] > 0
    assert provenance["checkpoint_call_count"] > 0
    assert provenance["recompute_count"] > 0
    assert provenance["buffers_match"] is True
    assert provenance["cpu_rng_match"] is True
    assert provenance["policy_hash"]
    assert "all other" in provenance["policy_source"]


def test_sac_conv_region_records_actual_save_and_recompute_decisions():
    prepared = prepare_selective_activation_checkpoint(
        _ConvResNet(),
        "resnet50",
        **_region_prepare_kwargs(shape=(2, 3, 8, 8)),
    ).require_supported()
    provenance = dict(prepared.identity.provenance)

    assert provenance["must_save_count"] > 0
    assert provenance["prefer_recompute_count"] > 0


def test_sac_executable_observes_both_policy_classes(monkeypatch):
    decisions = {}
    policy_factory = make_sac_policy
    def recording_policy_factory(tracker):
        class CombinedCounts(dict):
            def get(self, key, default=None):
                return decisions.get(key, tracker.get(key, default))

            def __setitem__(self, key, value):
                decisions[key] = value
                tracker[key] = value

        return policy_factory(CombinedCounts())

    monkeypatch.setattr(baseline_module, "make_sac_policy", recording_policy_factory)
    prepared = prepare_selective_activation_checkpoint(
        _FakeGPT(),
        "gpt2",
        **_region_prepare_kwargs(),
    ).require_supported()

    result = prepared.executable(torch.randn(2, 4, requires_grad=True))
    result.sum().backward()

    assert decisions["MUST_SAVE"] > 0
    assert decisions["PREFER_RECOMPUTE"] > 0


def test_unknown_model_is_unsupported_and_fails_closed():
    prepared = prepare_block_activation_checkpoint(
        nn.Linear(4, 4),
        "unknown",
        **_region_prepare_kwargs(),
    )

    assert not prepared.supported
    assert prepared.executable is None
    assert prepared.identity.fallback_reason
    with pytest.raises(UnsupportedMethodError, match="unsupported"):
        prepared.require_supported()


def test_runtime_identity_serialization_is_stable_and_explicit():
    identity = RuntimeIdentity(
        method_id="example",
        status="ready",
        is_real=True,
        api="example.api",
        policy="frozen",
        region_paths=("blocks.0",),
        compiler_protocol="aot:nop",
        fw_graph_sha256="a" * 64,
        bw_graph_sha256="b" * 64,
        provenance=(("torch_version", torch.__version__),),
    )

    payload = json.loads(identity.to_json())
    assert payload == identity.to_dict()
    assert payload["is_real"] is True
    assert payload["fallback_reason"] is None
    assert payload["provenance"]["torch_version"] == torch.__version__
