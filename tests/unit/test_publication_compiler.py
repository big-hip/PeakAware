import copy

import pytest
import torch
from torch import nn

import peakaware.publication.compiler as compiler_module
from peakaware.publication import prepare_publication_compiler, resolve_publication_regions
from peakaware.publication.qualification import qualify_three_step_correctness
from peakaware.runtime import measure as measurement


class _TinyGPT(nn.Module):
    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 4), nn.Dropout(dropout), nn.ReLU())
            for _ in range(2)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _prepare(model, method, *, callback=None):
    reference_model = copy.deepcopy(model)
    return prepare_publication_compiler(
        model,
        "gpt2",
        reference_model=reference_model,
        method=method,
        backend="aot_eager",
        example_args=(torch.randn(2, 4),),
        example_kwargs={},
        loss_fn=lambda output: output.square().mean(),
        graph_callback=callback,
    )


def test_aot_eager_methods_have_distinct_graphs_and_explicit_evidence():
    seed = _TinyGPT()
    prepared = {
        method: _prepare(copy.deepcopy(seed), method).require_supported()
        for method in ("all_save", "block_ac", "sac")
    }

    graph_pairs = {
        (item.identity.fw_graph_sha256, item.identity.bw_graph_sha256)
        for item in prepared.values()
    }
    assert len(graph_pairs) == 3
    for method, item in prepared.items():
        provenance = dict(item.identity.provenance)
        assert provenance["graph_break_count"] == 0
        assert provenance["fullgraph"] is True
        assert provenance["partition_callback_count"] == 1
        assert provenance["partition_fn"].endswith("default_partition")
        assert provenance["fw_callback_count"] == 1
        assert provenance["bw_callback_count"] == 1
        if method == "all_save":
            assert provenance["checkpoint_call_count"] == 0
            assert provenance["recompute_count"] == 0
            assert provenance["restore_cpu_rng_after_call"] is None
        elif method == "block_ac":
            assert provenance["checkpoint_call_count"] == 2
            assert provenance["recompute_count"] > 0
            assert provenance["use_reentrant"] is False
            assert provenance["restore_cpu_rng_after_call"] is True
        else:
            assert provenance["checkpoint_call_count"] == 2
            assert provenance["recompute_count"] > 0
            assert provenance["use_reentrant"] is False
            assert provenance["restore_cpu_rng_after_call"] is False

    sac = dict(prepared["sac"].identity.provenance)
    assert sac["policy_hash"]
    assert sac["policy_source"]
    assert sac["must_save_count"] > 0
    assert sac["prefer_recompute_count"] > 0
    if hasattr(torch.ops.aten, "scaled_dot_product_attention"):
        assert "aten.scaled_dot_product_attention.default" in sac["policy_save_ops"]
    for item in prepared.values():
        item.executable.close()


def test_graph_callback_receives_stable_fw_and_bw_hashes():
    calls = []
    prepared = _prepare(
        _TinyGPT(),
        "block_ac",
        callback=lambda phase, _graph, digest: calls.append((phase, digest)),
    ).require_supported()

    assert calls == [
        ("joint", dict(prepared.identity.provenance)["joint_graph_sha256"]),
        ("fw", prepared.identity.fw_graph_sha256),
        ("bw", prepared.identity.bw_graph_sha256),
    ]


def test_static_regions_preserve_state_dict_fqns_and_forward_identity():
    model = _TinyGPT()
    state_keys = tuple(model.state_dict())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    original_forward = model.blocks[0].forward
    prepared = _prepare(model, "block_ac").require_supported()
    installed_forward = model.blocks[0].forward

    for _ in range(2):
        output = prepared.executable(torch.randn(2, 4))
        output.sum().backward()
        model.zero_grad(set_to_none=True)
        assert model.blocks[0].forward is installed_forward

    assert installed_forward is not original_forward
    assert tuple(prepared.executable.state_dict()) == state_keys
    assert tuple(name for name, _ in prepared.executable.named_parameters()) == parameter_names
    assert tuple(name for name, _ in prepared.executable.named_modules()) == tuple(
        name for name, _ in model.named_modules()
    )
    assert prepared.executable.get_submodule("blocks.0") is model.blocks[0]
    assert tuple(model.state_dict()) == state_keys
    prepared.executable.close()


def test_dropout_rng_and_identity_are_stable_across_copied_models():
    first_model = _TinyGPT(dropout=0.25)
    second_model = copy.deepcopy(first_model)
    first = _prepare(first_model, "sac").require_supported()
    second = _prepare(second_model, "sac").require_supported()

    assert dict(first.identity.provenance)["cpu_rng_match"] is True
    assert first.identity.model_sha256 == second.identity.model_sha256
    assert first.identity.executable_sha256 == second.identity.executable_sha256


def test_training_batchnorm_whole_region_matches_independent_reference_on_213():
    model = _TinyGPT()
    model.blocks[0] = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
    reference = copy.deepcopy(model)
    args = (torch.randn(4, 4),)

    prepared = prepare_publication_compiler(
        model,
        "gpt2",
        reference_model=reference,
        method="block_ac",
        backend="aot_eager",
        example_args=args,
        example_kwargs={},
        loss_fn=lambda output: output.square().mean(),
    ).require_supported()

    report = qualify_three_step_correctness(
        reference,
        model,
        prepared.executable,
        torch.optim.SGD(reference.parameters(), lr=0.01),
        torch.optim.SGD(model.parameters(), lr=0.01),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )
    assert report["passed"] is True
    assert report["buffers_finite"] is True
    prepared.executable.close()


def test_publication_executable_applies_cudnn_override_only_during_call():
    model = nn.Identity()
    installer = compiler_module._StaticRegionInstaller(
        model,
        (),
        "all_save",
        cudnn_enabled_override=False,
    )
    executable = compiler_module.PublicationExecutable(
        model,
        lambda x: (x, torch.backends.cudnn.enabled),
        installer,
        {},
    )
    original = torch.backends.cudnn.enabled

    try:
        torch.backends.cudnn.enabled = True
        output, cudnn_enabled = executable(torch.ones(1))
        assert torch.equal(output, torch.ones(1))
        assert cudnn_enabled is False
        assert torch.backends.cudnn.enabled is True
    finally:
        torch.backends.cudnn.enabled = original
        executable.close()


class _SafeBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(4, 4, bias=False)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Linear(4, 4, bias=False)
        self.bn2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Linear(4, 4, bias=False)
        self.bn3 = nn.BatchNorm1d(4)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.relu(x + residual)


class _SafeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        for stage, depth in enumerate((3, 4, 6, 3), start=1):
            setattr(self, f"layer{stage}", nn.Sequential(*(_SafeBottleneck() for _ in range(depth))))

    def forward(self, x):
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = stage(x)
        return x


@pytest.mark.parametrize("method", ["block_ac", "sac"])
def test_resnet_bn_uses_complete_bottleneck_boundaries_and_passes_three_steps(method):
    model = _SafeResNet()
    reference = copy.deepcopy(model)
    args = (torch.randn(4, 4),)
    prepared = prepare_publication_compiler(
        model,
        "resnet50",
        reference_model=reference,
        method=method,
        backend="aot_eager",
        example_args=args,
        example_kwargs={},
        loss_fn=lambda output: output.square().mean(),
    ).require_supported()
    provenance = dict(prepared.identity.provenance)

    assert provenance["publication_block_count"] == 16
    assert provenance["checkpoint_region_count"] == 16
    assert provenance["region_boundary"] == "whole_publication_block"
    assert provenance["cudnn_enabled_override"] is False
    report = qualify_three_step_correctness(
        reference,
        model,
        prepared.executable,
        torch.optim.SGD(reference.parameters(), lr=0.01),
        torch.optim.SGD(model.parameters(), lr=0.01),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )
    assert report["passed"] is True
    assert report["buffers_finite"] is True
    prepared.executable.close()


def test_reference_model_must_be_independent():
    model = _TinyGPT()
    prepared = prepare_publication_compiler(
        model,
        "gpt2",
        reference_model=model,
        method="all_save",
        backend="aot_eager",
        example_args=(torch.randn(2, 4),),
        example_kwargs={},
        loss_fn=lambda output: output.square().mean(),
    )

    assert prepared.supported is False
    assert "independent" in prepared.identity.fallback_reason


def test_model_owner_rejects_block_sac_and_all_save_until_close():
    model = _TinyGPT()
    block = _prepare(model, "block_ac").require_supported()

    sac = _prepare(model, "sac")
    all_save = _prepare(model, "all_save")
    assert not sac.supported and "active publication compiler owner" in sac.identity.fallback_reason
    assert not all_save.supported and "active publication compiler owner" in all_save.identity.fallback_reason

    block.executable.close()
    sac = _prepare(model, "sac").require_supported()
    sac.executable.close()
    all_save = _prepare(model, "all_save").require_supported()
    all_save.executable.close()
    assert "forward" not in model.blocks[0].__dict__


def test_compiler_wrapper_is_strictly_snapshotable_until_owner_close():
    model = _TinyGPT()
    prepared = _prepare(model, "block_ac").require_supported()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    state = measurement._capture_training_state(model, optimizer, None, include_python_state=True)

    measurement._restore_training_state(model, optimizer, state, None)
    prepared.executable.close()
    with pytest.raises(ValueError, match="identity or ownership token changed"):
        measurement._restore_training_state(model, optimizer, state, None)


def test_region_control_named_kwargs_are_forwarded_without_checkpoint_collision():
    class ControlNamesBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x, *, context_fn, debug, use_reentrant):
            scale = context_fn + debug + use_reentrant
            return self.linear(x) * scale

    class ControlNamesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([ControlNamesBlock()])

        def forward(self, x, *, context_fn, debug, use_reentrant):
            return self.blocks[0](
                x,
                context_fn=context_fn,
                debug=debug,
                use_reentrant=use_reentrant,
            )

    model = ControlNamesModel()
    reference = copy.deepcopy(model)
    kwargs = {"context_fn": 1.0, "debug": 2.0, "use_reentrant": 3.0}
    prepared = prepare_publication_compiler(
        model,
        "gpt2",
        reference_model=reference,
        method="block_ac",
        backend="aot_eager",
        example_args=(torch.randn(2, 4),),
        example_kwargs=kwargs,
        loss_fn=lambda output: output.square().mean(),
    ).require_supported()

    assert torch.allclose(
        prepared.executable(torch.ones(2, 4), **kwargs),
        reference(torch.ones(2, 4), **kwargs),
    )
    prepared.executable.close()


def test_compiler_region_resolver_expands_frozen_architecture_depths():
    resnet = nn.Module()
    for stage, depth in enumerate((3, 4, 6, 3), start=1):
        setattr(resnet, f"layer{stage}", nn.Sequential(*(nn.Identity() for _ in range(depth))))
    vit = nn.Module()
    vit.encoder = nn.Module()
    vit.encoder.layers = nn.Sequential()
    for index in range(12):
        vit.encoder.layers.add_module(f"encoder_layer_{index}", nn.Identity())

    assert len(resolve_publication_regions(resnet, "resnet50")) == 16
    assert resolve_publication_regions(vit, "vit_b_16") == tuple(
        f"encoder.layers.encoder_layer_{index}" for index in range(12)
    )
    assert resolve_publication_regions(_TinyGPT(), "gpt2") == ("blocks.0", "blocks.1")
    resnet.layer4 = nn.Sequential(nn.Identity(), nn.Identity())
    assert resolve_publication_regions(resnet, "resnet50") == ()


def test_default_partition_preserves_supported_static_lifetime_indices(monkeypatch):
    observed = {}

    def fake_default_partition(
        graph,
        inputs,
        *,
        num_fwd_outputs,
        static_lifetime_input_indices=None,
    ):
        observed.update(
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
        )
        return graph, graph

    monkeypatch.setattr(torch._functorch.partitioners, "default_partition", fake_default_partition)
    graph = torch.fx.symbolic_trace(nn.Identity())
    compiler_module._default_partition(
        graph,
        (),
        {
            "compiler": "inductor",
            "num_fwd_outputs": 1,
            "static_lifetime_input_indices": [0, 2],
        },
    )

    assert observed == {
        "num_fwd_outputs": 1,
        "static_lifetime_input_indices": [0, 2],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_inductor_repeated_identity_uses_cold_cache_and_fires_callbacks():
    callback_counts = []
    for _ in range(2):
        model = _TinyGPT().cuda()
        prepared = prepare_publication_compiler(
            model,
            "gpt2",
            reference_model=copy.deepcopy(model),
            method="block_ac",
            backend="inductor",
            example_args=(torch.randn(2, 4, device="cuda"),),
            example_kwargs={},
            loss_fn=lambda output: output.square().mean(),
            cache_identity="repeated-cuda-cache-probe",
        )
        if not prepared.supported and "custom partitioner API is unavailable" in (
            prepared.identity.fallback_reason or ""
        ):
            pytest.skip(prepared.identity.fallback_reason)
        item = prepared.require_supported()
        try:
            provenance = dict(item.identity.provenance)
            assert "CustomPartitionerFn" in provenance["partition_fn"]
            assert provenance["graph_break_count"] == 0
            assert provenance["fresh_cache_per_preparation"] is True
            assert provenance["cache_reuse_disabled"] is True
            assert provenance["static_lifetime_input_indices_observed"] is True
            assert provenance["compiler_callback_pid"] == __import__("os").getpid()
            callback_counts.append(provenance["partition_callback_count"])
        finally:
            item.executable.close()

    assert callback_counts == [1, 1]
