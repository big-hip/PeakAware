import copy
import json
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

from peakaware.runtime import measure as measurement
from peakaware.runtime.measure import (
    measure_publication_training_step_phases,
    measure_training_step_phases,
)


class _StatefulLinear(nn.Module):
    def __init__(self, observations: list[int]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.int64))
        self.observations = observations

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.observations.append(int(self.steps.item()))
        self.steps.add_(1)
        return self.weight * value + torch.rand((), device=value.device) * 0


class _AliasingDeepcopyState:
    def __init__(self) -> None:
        self.counter = 0

    def __deepcopy__(self, memo):
        del memo
        return self


def test_legacy_measurement_preserves_original_call_count_and_state_semantics():
    torch.manual_seed(17)
    observations: list[int] = []
    model = _StatefulLinear(observations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.weight.grad = torch.tensor(7.0)
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = torch.get_rng_state().clone()

    metrics = measure_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square(),
        (torch.tensor(2.0),),
        {},
        zero_grad_set_to_none=True,
        warmup_steps=2,
        repeat_count=3,
    )

    assert observations == [0, 0, 0, 0, 0]
    assert metrics["measurement_warmup_steps"] == 2
    assert metrics["measurement_repeats"] == 3
    assert "raw_samples" not in metrics
    json.dumps(metrics)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in initial_model.items())
    assert optimizer.state_dict() == initial_optimizer
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert torch.equal(model.weight.grad, torch.tensor(7.0))


def test_overall_peak_comes_from_independent_full_step_window(monkeypatch: pytest.MonkeyPatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    allocated_calls = 0
    reserved_calls = 0
    resets: list[None] = []

    def allocated(*_):
        nonlocal allocated_calls
        allocated_calls += 1
        return 99 if allocated_calls <= 20 else (10, 20, 30)[(allocated_calls - 21) % 3]

    def reserved(*_):
        nonlocal reserved_calls
        reserved_calls += 1
        return 199 if reserved_calls <= 20 else (110, 120, 130)[(reserved_calls - 21) % 3]

    monkeypatch.setattr(measurement, "_reset_cuda_peak", lambda *_: resets.append(None))
    monkeypatch.setattr(measurement, "_cuda_allocated_peak_or_zero", allocated)
    monkeypatch.setattr(measurement, "_cuda_reserved_peak_or_zero", reserved)

    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(1, 1),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
    )

    assert len(resets) == 80
    assert metrics["fw_peak_bytes"] == 10
    assert metrics["bw_peak_bytes"] == 20
    assert metrics["optimizer_peak_bytes"] == 30
    assert metrics["overall_peak_bytes"] == 99
    assert metrics["overall_reserved_peak_bytes"] == 199
    assert metrics["trajectory_order"] == ["warmup", "overall", "phase"]
    assert metrics["raw_samples"][0]["trajectory_order"] == ["overall", "phase"]


def test_cpu_measurement_has_wall_time_and_explicitly_unavailable_events():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(2, 2),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=False,
        warmup_steps=5,
        repeat_count=20,
    )

    for window in ("fw", "bw", "optimizer", "overall"):
        assert metrics[f"{window}_wall_us"] > 0
        assert metrics[f"{window}_event_us"] is None
        assert metrics[f"{window}_event_wall_abs_diff_us"] is None
        assert metrics[f"{window}_event_wall_relative_diff"] is None
    assert metrics["fw_us"] == metrics["fw_wall_us"]
    assert metrics["step_us"] == metrics["phase_step_wall_us"]


@pytest.mark.parametrize(
    ("backend", "warmup_steps", "repeat_count", "message"),
    [
        ("eager", 5, 20, "backend"),
        ("aot_eager", 4, 20, "warmup_steps == 5"),
        ("inductor", 9, 20, "warmup_steps == 10"),
        ("aot_eager", 31, 20, "warmup_steps == 5"),
        ("aot_eager", 5, 19, "repeat_count >= 20"),
    ],
)
def test_publication_wrapper_enforces_protocol_thresholds(
    backend: str,
    warmup_steps: int,
    repeat_count: int,
    message: str,
):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=message):
        measure_publication_training_step_phases(
            model,
            optimizer,
            model,
            lambda output: output.square().mean(),
            (torch.ones(1, 1),),
            {},
            backend=backend,
            zero_grad_set_to_none=True,
            warmup_steps=warmup_steps,
            repeat_count=repeat_count,
        )


def test_publication_warmup_extends_when_first_five_steps_strictly_decrease(
    monkeypatch: pytest.MonkeyPatch,
):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    warmup_times = iter([100.0, 99.0, 98.0, 97.0, 96.0, 96.0])
    calls = 0

    def timed_window(fn, cuda_device):
        nonlocal calls
        del cuda_device
        calls += 1
        value = next(warmup_times) if calls <= 6 else 96.0
        return fn(), value, value

    monkeypatch.setattr(measurement, "_measure_window", timed_window)
    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(1, 1),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
    )

    assert [sample["warmup_wall_us"] for sample in metrics["warmup_samples"][:5]] == [
        100.0,
        99.0,
        98.0,
        97.0,
        96.0,
    ]
    assert metrics["warmup_samples"][4]["last5_wall_strictly_decreasing"] is True
    assert metrics["warmup_samples"][5]["last5_wall_strictly_decreasing"] is False
    assert len(metrics["warmup_samples"]) == 6
    assert metrics["warmup_auto_extended_steps"] == 1


def test_publication_warmup_fails_closed_when_still_decreasing_at_step_30(
    monkeypatch: pytest.MonkeyPatch,
):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    calls = 0

    def decreasing_window(fn, cuda_device):
        nonlocal calls
        del cuda_device
        calls += 1
        value = 1000.0 - calls if calls <= 30 else 100.0
        return fn(), value, value

    monkeypatch.setattr(measurement, "_measure_window", decreasing_window)
    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(1, 1),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
    )

    assert len(metrics["warmup_samples"]) == 30
    assert metrics["warmup_reached_max_steps"] is True
    assert metrics["warmup_last5_wall_strictly_decreasing"] is True
    assert metrics["warmup_trend_qualified"] is False
    assert metrics["publication_status"] == "warmup_unqualified"


@pytest.mark.parametrize(("warmup_steps", "repeat_count"), [(-1, 1), (0, 0), (0, -1)])
def test_compatible_api_rejects_negative_or_empty_measurement_counts(warmup_steps: int, repeat_count: int):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError):
        measure_training_step_phases(
            model,
            optimizer,
            model,
            lambda output: output.square().mean(),
            (torch.ones(1, 1),),
            {},
            zero_grad_set_to_none=True,
            warmup_steps=warmup_steps,
            repeat_count=repeat_count,
        )


def test_inputs_python_counter_and_training_mode_are_restored_across_trajectories(
    monkeypatch: pytest.MonkeyPatch,
):
    observations: list[tuple[float, float, bool, int]] = []
    model = nn.Linear(1, 1, bias=False)
    model.counter = 0
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    value = torch.tensor([[1.0]])
    nested = {"state": [torch.tensor(2.0)]}

    def stable_window(fn, cuda_device):
        del cuda_device
        return fn(), 100.0, 100.0

    monkeypatch.setattr(measurement, "_measure_window", stable_window)

    def executable(input_value: torch.Tensor, *, metadata: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        observations.append(
            (float(input_value.item()), float(metadata["state"][0].item()), model.training, model.counter)
        )
        input_value.add_(1)
        metadata["state"][0].add_(10)
        model.counter += 1
        model.eval()
        return model(input_value)

    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        executable,
        lambda output: output.square().mean(),
        (value,),
        {"metadata": nested},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
    )

    assert [item[3] for item in observations[:5]] == list(range(5))
    assert [item[3] for item in observations[5:25]] == list(range(5, 25))
    assert [item[3] for item in observations[25:45]] == list(range(5, 25))
    assert observations[5][:2] == (6.0, 52.0)
    assert observations[25][:2] == (6.0, 52.0)
    assert observations[5][2] is False
    assert observations[25][2] is False
    assert torch.equal(value, torch.tensor([[1.0]]))
    assert torch.equal(nested["state"][0], torch.tensor(2.0))
    assert model.training is True
    assert model.counter == 0
    assert metrics["trajectory_order"] == ["warmup", "overall", "phase"]
    assert metrics["callable_state_assumption"] == "pure_or_state_dict_owned"
    assert metrics["python_module_state_policy"] == "public_attributes_snapshotted_or_fail_closed"


def test_publication_rejects_aliasing_custom_python_state_before_execution():
    model = nn.Linear(1, 1)
    model.evil = _AliasingDeepcopyState()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    executions = 0

    def executable(value):
        nonlocal executions
        executions += 1
        model.evil.counter += 1
        return model(value)

    with pytest.raises(ValueError, match="cannot snapshot Python module attribute evil"):
        measure_publication_training_step_phases(
            model,
            optimizer,
            executable,
            lambda output: output.square().mean(),
            (torch.ones(1, 1),),
            {},
            backend="aot_eager",
            zero_grad_set_to_none=True,
            warmup_steps=5,
            repeat_count=20,
        )

    assert executions == 0
    assert model.evil.counter == 0


def test_phase_samples_record_after_forward_retained_memory(monkeypatch: pytest.MonkeyPatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    monkeypatch.setattr(measurement, "_cuda_allocated_or_zero", lambda *_: 123)
    monkeypatch.setattr(measurement, "_cuda_reserved_or_zero", lambda *_: 456)

    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(1, 1),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
    )

    assert metrics["after_fw_allocated_bytes"] == 123
    assert metrics["after_fw_reserved_bytes"] == 456
    assert all(sample["after_fw_allocated_bytes"] == 123 for sample in metrics["phase_samples"])
    assert all(sample["after_fw_reserved_bytes"] == 456 for sample in metrics["phase_samples"])


def test_publication_wrapper_marks_large_event_wall_gap_unqualified(monkeypatch: pytest.MonkeyPatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def mismatched_window(fn, cuda_device):
        del cuda_device
        return fn(), 100.0, 10.0

    monkeypatch.setattr(measurement, "_measure_window", mismatched_window)
    metrics = measure_publication_training_step_phases(
        model,
        optimizer,
        model,
        lambda output: output.square().mean(),
        (torch.ones(1, 1),),
        {},
        backend="aot_eager",
        zero_grad_set_to_none=True,
        warmup_steps=5,
        repeat_count=20,
        max_event_wall_relative_gap=0.20,
    )

    assert len(metrics["warmup_samples"]) == 5
    assert metrics["warmup_last5_wall_relative_slope"] == pytest.approx(0.0)
    assert metrics["warmup_last5_event_relative_slope"] == pytest.approx(0.0)
    assert metrics["overall_samples"][0]["overall_event_wall_relative_diff"] == pytest.approx(9.0)
    assert metrics["overall_timing_qualified"] is False
    assert metrics["timing_qualified"] is False
    assert metrics["publication_qualified"] is False
    assert metrics["publication_status"] == "timing_unqualified"
    assert "event_wall_gap_or_event_unavailable" in metrics["publication_unqualified_reasons"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real Event timing")
def test_cuda_measurement_records_real_event_and_wall_timing():
    script = textwrap.dedent(
        """
        import torch
        from torch import nn
        from peakaware.runtime.measure import measure_publication_training_step_phases

        device = torch.device("cuda")
        model = nn.Linear(32, 32).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = measure_publication_training_step_phases(
            model,
            optimizer,
            model,
            lambda output: output.square().mean(),
            (torch.randn(32, 32, device=device),),
            {},
            backend="aot_eager",
            zero_grad_set_to_none=True,
            warmup_steps=5,
            repeat_count=20,
        )
        for window in ("fw", "bw", "optimizer", "overall"):
            assert metrics[f"{window}_event_us"] > 0
            assert metrics[f"{window}_wall_us"] > 0
            assert metrics[f"{window}_event_wall_abs_diff_us"] >= 0
            assert metrics[f"{window}_event_wall_relative_diff"] >= 0
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
