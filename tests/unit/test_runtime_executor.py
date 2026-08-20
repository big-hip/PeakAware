from types import SimpleNamespace

import pytest
import torch
from torch import nn

from peakaware.runtime import executor as executor_module
from peakaware import PeakAwareConfig
from peakaware.contracts import GuardSpec, LoweredPartition, PartitionABI
from peakaware.runtime.executor import (
    EagerTrainingStepExecutor,
    build_aot_partition_executable,
    build_training_step_executor,
    make_measured_executable,
    validate_runtime_guards,
)


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


def test_runtime_guards_validate_nested_tensor_leaves():
    args = ({"x": torch.ones(2, 3), "y": torch.ones(2, 3)},)
    kwargs = {"scale": {"s": torch.ones(2, 3)}}
    guards = (
        GuardSpec("arg0.flat0.shape", "(2, 3)"),
        GuardSpec("arg0.flat1.dtype", "torch.float32"),
        GuardSpec("kw.scale.flat0.device", "cpu"),
    )

    validate_runtime_guards(guards, args, kwargs)

    with pytest.raises(ValueError, match="runtime guard failed"):
        validate_runtime_guards(guards, ({"x": torch.ones(4, 3), "y": torch.ones(2, 3)},), kwargs)


def test_runtime_guards_reject_nested_static_value_drift():
    args = ({"x": torch.ones(2, 3), "scale": 2.5},)
    kwargs = {"config": {"bias": 1.0, "training": True}}
    guards = (
        GuardSpec("arg0.flat1.value", "builtins.float:2.5"),
        GuardSpec("kw.config.flat0.value", "builtins.float:1.0"),
        GuardSpec("kw.config.flat1.value", "builtins.bool:true"),
    )

    validate_runtime_guards(guards, args, kwargs)

    with pytest.raises(ValueError, match="runtime guard failed"):
        validate_runtime_guards(guards, ({"x": torch.ones(2, 3), "scale": 3.0},), kwargs)


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


def test_activation_checkpoint_executor_runs_and_reports_plan_marker():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        activation_checkpoint=True,
    )

    measured = make_measured_executable("checkpointed", executor, (torch.ones(2, 3),), {}, simulated_peak_bytes=1024)

    assert executor.activation_checkpoint is True
    assert measured.correctness_passed
    assert measured.phase_metrics["activation_checkpoint"] == 1
    assert measured.phase_metrics["candidate_measurement_protocol"] == "legacy_phase"
    assert measured.phase_metrics["runtime_step_source"] == "legacy_phase_wall_sum"


def test_publication_candidate_measurement_uses_overall_cuda_event_median(
    monkeypatch: pytest.MonkeyPatch,
):
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = PeakAwareConfig(
        enable_compile=True,
        measurement_warmup_steps=5,
        measurement_repeats=20,
        candidate_measurement_protocol="publication_overall",
    )
    executor = EagerTrainingStepExecutor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        model,
        config,
    )

    monkeypatch.setattr(
        executor_module,
        "measure_publication_training_step_phases",
        lambda *args, **kwargs: {
            "overall_event_us": 123.0,
            "overall_wall_us": 150.0,
            "overall_peak_bytes": 456,
        },
    )

    measured = make_measured_executable(
        "publication",
        executor,
        (torch.ones(2, 3),),
        {},
        simulated_peak_bytes=1024,
    )

    assert measured.measured_step_us == 123.0
    assert measured.measured_peak_bytes == 456
    assert measured.phase_metrics["runtime_step_us"] == 123.0
    assert measured.phase_metrics["runtime_step_source"] == "overall_cuda_event_median"
    assert measured.phase_metrics["candidate_measurement_protocol"] == "publication_overall"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_measurement_protocol": "unknown"}, "candidate_measurement_protocol"),
        (
            {"candidate_measurement_protocol": "publication_overall"},
            "requires enable_compile=True",
        ),
        (
            {
                "enable_compile": True,
                "candidate_measurement_protocol": "publication_overall",
                "measurement_warmup_steps": 4,
                "measurement_repeats": 20,
            },
            "measurement_warmup_steps=5",
        ),
        (
            {
                "enable_compile": True,
                "candidate_measurement_protocol": "publication_overall",
                "measurement_warmup_steps": 5,
                "measurement_repeats": 19,
            },
            "measurement_repeats>=20",
        ),
    ],
)
def test_publication_candidate_measurement_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        PeakAwareConfig(**overrides).validate()


def test_activation_checkpoint_executor_preserves_object_outputs():
    torch.manual_seed(0)

    class LogitsModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x):
            return SimpleNamespace(logits=self.linear(x))

    model = LogitsModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    executor = build_training_step_executor(
        model,
        optimizer,
        lambda out: out.logits.pow(2).mean(),
        PeakAwareConfig(enable_compile=False),
        activation_checkpoint=True,
    )

    step = executor.step(torch.ones(2, 3))

    assert step.optimizer_step_performed
    assert executor.activation_checkpoint is True
    assert step.metrics["activation_checkpoint"] == 1


def test_executor_switches_to_verified_fallback_after_runtime_peak_breach():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = {"selected": 0, "fallback": 0}

    def selected(x):
        calls["selected"] += 1
        return model(x)

    def fallback(x):
        calls["fallback"] += 1
        return model(x)

    executor = EagerTrainingStepExecutor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        selected,
        PeakAwareConfig(enable_compile=False),
        plan_id="selected",
        fallback_executables=(("fallback", fallback),),
        runtime_peak_threshold_bytes=5,
        runtime_peak_observer=lambda: 10,
    )
    optimizer_id = id(executor.optimizer)

    first = executor.step(torch.ones(2, 3))
    executor.runtime_peak_observer = lambda: 0
    second = executor.step(torch.ones(2, 3))

    assert first.metrics["fallback_plan_id"] == "fallback"
    assert "exceeded threshold" in first.metrics["fallback_reason"]
    assert executor.current_plan_id == "fallback"
    assert calls == {"selected": 1, "fallback": 1}
    assert second.metrics["plan_id"] == "fallback"
    assert id(executor.optimizer) == optimizer_id


def test_executor_fallback_refreshes_activation_checkpoint_marker():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def selected(x):
        return model(x)

    def fallback(x):
        return model(x)

    executor = EagerTrainingStepExecutor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        selected,
        PeakAwareConfig(enable_compile=False),
        plan_id="checkpointed",
        fallback_executables=(("all_save", fallback),),
        runtime_peak_threshold_bytes=5,
        runtime_peak_observer=lambda: 10,
        activation_checkpoint=True,
        fallback_activation_checkpoints={"all_save": False},
    )

    first = executor.step(torch.ones(2, 3))
    executor.runtime_peak_observer = lambda: 0
    second = executor.step(torch.ones(2, 3))

    assert first.metrics["activation_checkpoint"] == 1
    assert first.metrics["fallback_plan_id"] == "all_save"
    assert first.metrics["fallback_activation_checkpoint"] == 0
    assert second.metrics["plan_id"] == "all_save"
    assert second.metrics["activation_checkpoint"] == 0
    assert executor.activation_checkpoint is False


def test_executor_fallback_refreshes_aot_partition_runtime_marker():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def selected(x):
        return model(x)

    def fallback(x):
        return model(x)

    executor = EagerTrainingStepExecutor(
        model,
        optimizer,
        lambda out: out.pow(2).mean(),
        selected,
        PeakAwareConfig(enable_compile=False),
        plan_id="checkpointed",
        fallback_executables=(("aot_partition", fallback),),
        runtime_peak_threshold_bytes=5,
        runtime_peak_observer=lambda: 10,
        activation_checkpoint=True,
        aot_partition_runtime=False,
        fallback_activation_checkpoints={"aot_partition": False},
        fallback_aot_partition_runtimes={"aot_partition": True},
    )

    first = executor.step(torch.ones(2, 3))
    executor.runtime_peak_observer = lambda: 0
    second = executor.step(torch.ones(2, 3))

    assert first.metrics["activation_checkpoint"] == 1
    assert first.metrics["aot_partition_runtime"] == 0
    assert first.metrics["fallback_plan_id"] == "aot_partition"
    assert first.metrics["fallback_activation_checkpoint"] == 0
    assert first.metrics["fallback_aot_partition_runtime"] == 1
    assert second.metrics["plan_id"] == "aot_partition"
    assert second.metrics["activation_checkpoint"] == 0
    assert second.metrics["aot_partition_runtime"] == 1
    assert executor.activation_checkpoint is False
    assert executor.aot_partition_runtime is True


class _BufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running", torch.zeros(2))


class _FwGraph(nn.Module):
    def forward(self, running, value):
        return running + 1, value * 2, value


class _BwGraph(nn.Module):
    def forward(self, saved_value, grad_output):
        return None, grad_output * 2


def test_aot_partition_executable_applies_mutated_buffer_output_prefix():
    model = _BufferModel()
    lowered = LoweredPartition(
        plan_id="mutated-buffer-prefix",
        fw_graph=torch.fx.symbolic_trace(_FwGraph()),
        bw_graph=torch.fx.symbolic_trace(_BwGraph()),
        partition_abi=PartitionABI(
            fw_output_value_ids=(),
            bw_placeholder_value_ids=(),
            tangent_value_ids=(),
            rng_state_value_ids=(),
        ),
    )
    executable = build_aot_partition_executable(
        lowered,
        model,
        num_fwd_outputs=2,
        output_tangent_mask=(True,),
    )
    value = torch.tensor([3.0, 4.0], requires_grad=True)

    output = executable(value)
    output.sum().backward()

    assert torch.equal(output, torch.tensor([6.0, 8.0]))
    assert torch.equal(model.running, torch.ones(2))
    assert torch.equal(value.grad, torch.tensor([2.0, 2.0]))
