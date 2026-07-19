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
from peakaware.partition.verifier import run_aot_eager_dry_run
from peakaware.search.plan import build_recompute_plan


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
    assert "non-recomputable or mandatory" in result.failure_reason
