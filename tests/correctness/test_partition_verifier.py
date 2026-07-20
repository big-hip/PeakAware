import torch
from torch import nn

from peakaware.capture.joint import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import (
    HardwareSpec,
    JointTrainingIR,
    LoweredPartition,
    PartitionABI,
    TrainingRequest,
    ValueInfo,
)
from peakaware.ir.builder import build_joint_ir
from peakaware.memory.fixed_frontier import build_optimizer_spec
from peakaware.partition.aot import partition_joint_graph
from peakaware.partition.verifier import (
    compare_lowered_partition_with_baseline,
    compare_dry_run_with_baseline,
    compare_multistep_training_with_baseline,
    run_aot_eager_dry_run,
)
from peakaware.runtime.executor import build_aot_partition_executable
from peakaware.search.plan import build_recompute_plan


class TiedLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        hidden = self.shared(x).relu()
        reused = self.shared(hidden).relu()
        return self.out(reused)


def test_dry_run_compares_loss_and_gradients():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(p=0.1), nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 3),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    capture = capture_joint_graph(request)
    ir, _ = build_joint_ir(capture)
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(capture.joint_module, plan, ir)

    result = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={},
        loss_fn=lambda out: out.sum(),
        atol=1e-5,
        rtol=1e-4,
        ir=ir,
    )

    assert result.abi_valid
    assert result.outputs_match
    assert result.gradients_match
    assert result.replay_mode == "lowered_aot"


def test_dry_run_handles_tied_shared_weights():
    torch.manual_seed(0)
    model = TiedLinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="tied",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(capture.joint_module, plan, ir)

    result = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={},
        loss_fn=lambda out: out.pow(2).mean(),
        atol=1e-5,
        rtol=1e-4,
        ir=ir,
    )

    assert result.abi_valid
    assert result.outputs_match
    assert result.gradients_match
    assert result.replay_mode == "lowered_aot"


def test_multistep_correctness_handles_dropout_and_restores_state():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(p=0.4), nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = (torch.randn(4, 3),)
    model_state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng_before = torch.get_rng_state()

    ok, reason = compare_multistep_training_with_baseline(
        model,
        args,
        {},
        lambda out: out.pow(2).mean(),
        optimizer,
        step_count=3,
        atol=1e-6,
        rtol=1e-5,
    )

    assert ok, reason
    assert torch.equal(rng_before, torch.get_rng_state())
    assert optimizer.state_dict()["state"] == {}
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, model_state_before[name])


def test_dry_run_restores_batchnorm_buffers_and_rng():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.BatchNorm1d(4),
        nn.Dropout(p=0.25),
        nn.Linear(4, 1),
    )
    args = (torch.randn(8, 3),)
    model_state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng_before = torch.get_rng_state()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        ok, reason = compare_dry_run_with_baseline(
            model,
            args,
            {},
            lambda out: out.pow(2).mean(),
            atol=1e-6,
            rtol=1e-5,
        )
    finally:
        restored_benchmark = torch.backends.cudnn.benchmark
        restored_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic

    assert ok, reason
    assert restored_benchmark is True
    assert restored_deterministic is False
    assert torch.equal(rng_before, torch.get_rng_state())
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, model_state_before[name])


def test_saved_value_policy_changes_aot_partition_shape():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    all_save = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    mandatory_only = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.mandatory_save_reason),
        label="mandatory_only",
    )

    all_save_partition = partition_joint_graph(
        capture.joint_module,
        all_save,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )
    mandatory_partition = partition_joint_graph(
        capture.joint_module,
        mandatory_only,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )

    mandatory_bw_nodes = len(list(mandatory_partition.bw_graph.graph.nodes))
    all_save_bw_nodes = len(list(all_save_partition.bw_graph.graph.nodes))

    assert mandatory_bw_nodes > all_save_bw_nodes
    assert (
        mandatory_partition.partition_abi.fw_output_value_ids
        != all_save_partition.partition_abi.fw_output_value_ids
    )


def test_lowered_aot_partition_replay_matches_baseline_gradients():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-replay",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )

    ok, reason = compare_lowered_partition_with_baseline(
        lowered,
        model,
        args,
        {},
        lambda out: out.sum(),
        num_fwd_outputs=capture.num_fwd_outputs,
        atol=1e-5,
        rtol=1e-4,
    )

    assert ok, reason


def test_lowered_aot_partition_executable_matches_eager_backward():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-executable",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )
    executable = build_aot_partition_executable(lowered, model, num_fwd_outputs=capture.num_fwd_outputs)

    optimizer.zero_grad(set_to_none=True)
    eager_loss = request.loss_fn(model(*args))
    eager_loss.backward()
    eager_grads = tuple(param.grad.detach().clone() for param in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    partition_loss = request.loss_fn(executable(*args))
    partition_loss.backward()
    partition_grads = tuple(param.grad.detach().clone() for param in model.parameters())

    assert torch.allclose(partition_loss.detach(), eager_loss.detach(), atol=1e-6, rtol=1e-5)
    for actual, expected in zip(partition_grads, eager_grads):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lowered_aot_partition_executable_supports_tensor_kwargs():
    torch.manual_seed(0)

    class KwargModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x, scale=None, bias=None):
            return self.linear(x * scale + bias)

    model = KwargModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    kwargs = {
        "scale": torch.full((2, 4), 2.0),
        "bias": torch.full((2, 4), 5.0),
    }
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-executable-kwargs",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )
    dry_run = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={"bias": kwargs["bias"], "scale": kwargs["scale"]},
        loss_fn=request.loss_fn,
        atol=1e-6,
        rtol=1e-5,
        ir=ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        kwarg_names=tuple(kwargs),
    )
    executable = build_aot_partition_executable(
        lowered,
        model,
        num_fwd_outputs=capture.num_fwd_outputs,
        kwarg_names=tuple(kwargs),
    )

    optimizer.zero_grad(set_to_none=True)
    eager_loss = request.loss_fn(model(*args, **kwargs))
    eager_loss.backward()
    eager_grads = tuple(param.grad.detach().clone() for param in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    partition_loss = request.loss_fn(executable(*args, bias=kwargs["bias"], scale=kwargs["scale"]))
    partition_loss.backward()
    partition_grads = tuple(param.grad.detach().clone() for param in model.parameters())

    assert dry_run.replay_mode == "lowered_aot"
    assert dry_run.gradients_match
    assert torch.allclose(partition_loss.detach(), eager_loss.detach(), atol=1e-6, rtol=1e-5)
    for actual, expected in zip(partition_grads, eager_grads):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lowered_aot_partition_executable_supports_multiple_tensor_outputs():
    torch.manual_seed(0)

    class MultiOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 2)
            self.right = nn.Linear(4, 2)

        def forward(self, x):
            return self.left(x), self.right(x).relu()

    def loss_fn(outputs):
        left, right = outputs
        return left.pow(2).mean() + right.pow(2).mean()

    model = MultiOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-executable-multi-output",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    assert capture.num_fwd_outputs == 2
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )
    dry_run = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={},
        loss_fn=loss_fn,
        atol=1e-6,
        rtol=1e-5,
        ir=ir,
        num_fwd_outputs=capture.num_fwd_outputs,
    )
    executable = build_aot_partition_executable(
        lowered,
        model,
        num_fwd_outputs=capture.num_fwd_outputs,
    )

    optimizer.zero_grad(set_to_none=True)
    eager_loss = loss_fn(model(*args))
    eager_loss.backward()
    eager_grads = tuple(param.grad.detach().clone() for param in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    partition_outputs = executable(*args)
    partition_loss = loss_fn(partition_outputs)
    partition_loss.backward()
    partition_grads = tuple(param.grad.detach().clone() for param in model.parameters())

    assert dry_run.replay_mode == "lowered_aot"
    assert dry_run.gradients_match
    assert isinstance(partition_outputs, tuple)
    assert len(partition_outputs) == 2
    assert torch.allclose(partition_loss.detach(), eager_loss.detach(), atol=1e-6, rtol=1e-5)
    for actual, expected in zip(partition_grads, eager_grads):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lowered_aot_partition_executable_reconstructs_nested_tensor_outputs():
    torch.manual_seed(0)

    class NestedOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 2)
            self.right = nn.Linear(4, 2)

        def forward(self, x):
            return {"left": self.left(x), "right": (self.right(x).relu(),)}

    def loss_fn(outputs):
        return outputs["left"].pow(2).mean() + outputs["right"][0].pow(2).mean()

    model = NestedOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-executable-nested-output",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    assert capture.num_fwd_outputs == 2
    assert capture.output_tree_spec is not None
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )
    dry_run = run_aot_eager_dry_run(
        lowered,
        model=model,
        args=args,
        kwargs={},
        loss_fn=loss_fn,
        atol=1e-6,
        rtol=1e-5,
        ir=ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        output_tree_spec=capture.output_tree_spec,
    )
    executable = build_aot_partition_executable(
        lowered,
        model,
        num_fwd_outputs=capture.num_fwd_outputs,
        output_tree_spec=capture.output_tree_spec,
    )

    optimizer.zero_grad(set_to_none=True)
    eager_loss = loss_fn(model(*args))
    eager_loss.backward()
    eager_grads = tuple(param.grad.detach().clone() for param in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    partition_outputs = executable(*args)
    partition_loss = loss_fn(partition_outputs)
    partition_loss.backward()
    partition_grads = tuple(param.grad.detach().clone() for param in model.parameters())

    assert dry_run.replay_mode == "lowered_aot"
    assert dry_run.gradients_match
    assert set(partition_outputs) == {"left", "right"}
    assert isinstance(partition_outputs["right"], tuple)
    assert torch.allclose(partition_loss.detach(), eager_loss.detach(), atol=1e-6, rtol=1e-5)
    for actual, expected in zip(partition_grads, eager_grads):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_lowered_aot_partition_replay_passes_model_buffers_as_primals():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(3, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.pow(2).mean(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-replay-buffers",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )

    ok, reason = compare_lowered_partition_with_baseline(
        lowered,
        model,
        args,
        {},
        lambda out: out.pow(2).mean(),
        num_fwd_outputs=capture.num_fwd_outputs,
        atol=1e-5,
        rtol=1e-4,
    )

    assert ok, reason


def test_lowered_aot_partition_replay_applies_mutated_buffer_output_prefix():
    class BufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running", torch.zeros(2))

        def forward(self, value):
            with torch.no_grad():
                self.running.copy_(self.running + 1)
            return value * 2

    class FwGraph(nn.Module):
        def forward(self, primals_1, primals_2):
            return primals_1 + 1, primals_2 * 2, primals_2

    class BwGraph(nn.Module):
        def forward(self, saved_value, tangents_1):
            return ()

    model = BufferModel()
    value = torch.tensor([3.0, 4.0], requires_grad=True)
    lowered = LoweredPartition(
        plan_id="mutated-buffer-prefix",
        fw_graph=torch.fx.symbolic_trace(FwGraph()),
        bw_graph=torch.fx.symbolic_trace(BwGraph()),
        partition_abi=PartitionABI((), (), (), ()),
    )

    ok, reason = compare_lowered_partition_with_baseline(
        lowered,
        model,
        (value,),
        {},
        lambda out: out.sum(),
        num_fwd_outputs=2,
        output_tangent_mask=(True,),
        atol=1e-6,
        rtol=1e-5,
    )

    assert ok, reason
    assert torch.equal(model.running, torch.zeros(2))


def test_lowered_aot_partition_replay_detects_bad_backward_gradient():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 4),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(capture_backend="aot"),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="partition-replay-bad",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(value.id for value in ir.values if value.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(
        capture.joint_module,
        plan,
        ir,
        num_fwd_outputs=capture.num_fwd_outputs,
        static_lifetime_input_indices=capture.static_lifetime_input_indices,
    )

    class BadBackward:
        graph = lowered.bw_graph.graph

        def __call__(self, *inputs):
            outputs = list(lowered.bw_graph(*inputs))
            outputs[0] = torch.zeros_like(outputs[0])
            return tuple(outputs)

    bad = LoweredPartition(
        lowered.plan_id,
        lowered.fw_graph,
        BadBackward(),
        lowered.partition_abi,
    )

    ok, reason = compare_lowered_partition_with_baseline(
        bad,
        model,
        args,
        {},
        lambda out: out.sum(),
        num_fwd_outputs=capture.num_fwd_outputs,
        atol=1e-5,
        rtol=1e-4,
    )

    assert not ok
    assert reason == "lowered gradient mismatch at parameter 0"


def test_dry_run_rejects_unknown_partition_abi_value():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 3),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(capture.joint_module, plan, ir)
    bad = LoweredPartition(
        "bad",
        lowered.fw_graph,
        lowered.bw_graph,
        PartitionABI((999,), (999,), (), ()),
    )

    result = run_aot_eager_dry_run(
        bad,
        model=model,
        args=args,
        kwargs={},
        loss_fn=lambda out: out.sum(),
        atol=1e-5,
        rtol=1e-4,
        ir=ir,
    )

    assert not result.abi_valid
    assert result.replay_mode == "not_run"
    assert "unknown IR value ids" in result.failure_reason


def test_dry_run_rejects_dropped_non_recomputable_forward_value():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (torch.randn(2, 3),)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs={},
        loss_fn=lambda out: out.sum(),
        optimizer=optimizer,
        memory_budget_bytes=1 << 30,
        config=PeakAwareConfig(),
        optimizer_spec=build_optimizer_spec(optimizer, model),
        hardware=HardwareSpec("cpu", False, None),
        request_key="test",
    )
    capture = capture_joint_graph(request)
    ir, report = build_joint_ir(capture)
    assert report.valid
    plan = build_recompute_plan(
        ir,
        budget_bytes=1 << 30,
        saved_value_ids=frozenset(v.id for v in ir.values if v.phase == "fw"),
        label="all_save",
    )
    lowered = partition_joint_graph(capture.joint_module, plan, ir)
    synthetic_ir = JointTrainingIR(
        ops=(),
        values=(
            ValueInfo(
                id=1,
                producer_id=0,
                consumer_ids=(1,),
                storage_id=1,
                logical_nbytes=4,
                phase="fw",
                crosses_fw_bw=True,
                recomputable=False,
                mandatory_save_reason=None,
            ),
        ),
        storages=(),
        regions=(),
        graph_key="synthetic",
    )
    bad = LoweredPartition(
        "bad",
        lowered.fw_graph,
        lowered.bw_graph,
        PartitionABI((), (), (), ()),
    )

    result = run_aot_eager_dry_run(
        bad,
        model=model,
        args=args,
        kwargs={},
        loss_fn=lambda out: out.sum(),
        atol=1e-5,
        rtol=1e-4,
        ir=synthetic_ir,
    )

    assert not result.abi_valid
    assert result.replay_mode == "not_run"
    assert "non-recomputable or mandatory" in result.failure_reason
