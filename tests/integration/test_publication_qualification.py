from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import torch
import pytest
from torch import nn

from peakaware.contracts import FrozenConfig, TrainingTaskSpec
from peakaware.models.registry import build_tiny_mlp_task
from peakaware.publication.qualification import (
    QualificationRecord,
    _build_seeded_batch,
    _build_seeded_model_optimizer,
    _marker_process_ids,
    _terminate_process,
    _timeout_stage,
    build_qualification_slots,
    cleanup_uncommitted_qualification_artifacts,
    convert_physical_budget_to_activation_memory_budget,
    qualify_three_step_correctness,
    run_qualification_slots,
    summarize_qualification,
    validate_qualification_artifact_bundle,
    validate_qualification_record,
    validate_qualification_slot,
    validate_worker_task_binding,
    write_qualification_artifacts,
)


def _descendant_process_target(channel, marker):
    os.environ["PEAKAWARE_QUALIFICATION_MARKER"] = marker
    if os.name == "posix":
        os.setsid()
    command = [sys.executable, "-c", "import time; time.sleep(180)", marker]
    same_group = subprocess.Popen(command)
    escaped_group = subprocess.Popen(command, start_new_session=True)
    channel.send((same_group.pid, escaped_group.pid))
    channel.close()
    time.sleep(60)


def _slots(
    *,
    methods=("all_save",),
    backends=("aot_eager",),
    replicates=1,
    timeout_s=30.0,
    attempt_index=0,
):
    return build_qualification_slots(
        (build_tiny_mlp_task(),),
        run_id="test-e0",
        backends=backends,
        methods=methods,
        memory_budgets_bytes=(1 << 20,),
        replicates=replicates,
        microbatch_size=2,
        device="cpu",
        base_seed=17,
        repeat_count=20,
        timeout_s=timeout_s,
        attempt_index=attempt_index,
    )


def test_slots_are_deterministic_and_pair_replicates_across_methods():
    first_manifest, first = _slots(methods=("all_save", "peakaware"), replicates=2)
    second_manifest, second = _slots(methods=("peakaware", "all_save"), replicates=2)

    assert first_manifest == second_manifest
    assert [slot.to_dict() for slot in first] == [slot.to_dict() for slot in second]
    assert len(first) == 4
    by_replicate = {}
    for slot in first:
        by_replicate.setdefault(slot.replicate_index, []).append(slot)
    for paired in by_replicate.values():
        assert len({slot.replicate_id for slot in paired}) == 1
        assert len({slot.case_id for slot in paired}) == 2
        assert len({slot.attempt_id for slot in paired}) == 1
        assert len({slot.slot_id for slot in paired}) == 2
    run = first_manifest["qualification_run"]
    assert run["run_id"] == "test-e0"
    assert run["expected_slot_count"] == 4
    assert run["expected_slot_ids"] == sorted(slot.slot_id for slot in first)
    assert run["execution_order"] == [slot.slot_id for slot in first]
    assert all(block["shuffle_seed"] >= 0 for block in run["pairing_blocks"])
    assert run["matrix"]["repeat_count"] == 20
    assert run["matrix"]["timeout_s"] == 30.0
    assert run["matrix"]["warmup_steps_by_backend"] == {"aot_eager": 5}
    assert run["matrix"]["spawn_start_interruptible"] is False
    assert "cannot be safely interrupted" in run["matrix"]["spawn_start_boundary"]


def test_build_slots_rejects_duplicate_registry_key_and_workload_fingerprint():
    task = build_tiny_mlp_task()
    try:
        build_qualification_slots(
            (task, task),
            run_id="duplicates",
            backends=("aot_eager",),
            methods=("all_save",),
            memory_budgets_bytes=(1 << 20,),
            replicates=1,
            microbatch_size=1,
            device="cpu",
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate workload registration was accepted")


@pytest.mark.parametrize(
    "override",
    (
        {"tasks": ()},
        {"backends": ()},
        {"methods": ()},
        {"memory_budgets_bytes": ()},
    ),
)
def test_build_slots_rejects_empty_matrix_dimensions(override):
    arguments = {
        "tasks": (build_tiny_mlp_task(),),
        "run_id": "empty-dimension",
        "backends": ("aot_eager",),
        "methods": ("all_save",),
        "memory_budgets_bytes": (1 << 20,),
        "replicates": 1,
        "microbatch_size": 1,
        "device": "cpu",
    }
    arguments.update(override)

    with pytest.raises(ValueError, match="must not be empty"):
        build_qualification_slots(**arguments)


def test_task_binding_rejects_same_workload_with_replaced_factories_and_loss():
    task = build_tiny_mlp_task()
    manifest, slots = _slots(backends=("inductor",))
    slot = slots[0]

    def zero_model():
        model = task.build_model()
        with torch.no_grad():
            for value in model.state_dict().values():
                value.zero_()
        return model

    def zero_batch(batch_size):
        return (torch.zeros(batch_size, 8),), {}

    replacements = (
        replace(task, build_model=zero_model),
        replace(task, build_batch=zero_batch),
        replace(task, build_optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.5)),
        replace(task, loss_fn=lambda output: output.abs().mean()),
    )
    for replacement in replacements:
        with pytest.raises(ValueError, match="task binding"):
            validate_worker_task_binding(replacement, slot, manifest)


def test_build_slots_and_task_binding_probe_restore_all_caller_rng_states():
    np = __import__("numpy")
    random.seed(301)
    np.random.seed(302)
    torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(304)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    manifest, slots = _slots(backends=("inductor",))
    validate_worker_task_binding(build_tiny_mlp_task(), slots[0], manifest)

    restored_numpy = np.random.get_state()
    assert random.getstate() == python_state
    assert restored_numpy[0] == numpy_state[0] and restored_numpy[2:] == numpy_state[2:]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(
        torch.equal(restored, expected)
        for restored, expected in zip(torch.cuda.get_rng_state_all(), cuda_states)
    )


def test_worker_fails_closed_when_registry_factory_does_not_match_preregistered_workload():
    width16 = build_tiny_mlp_task(width=16)
    forged_workload = replace(
        width16.workload,
        registry_key="tiny_mlp_w8_d3",
    )
    forged_task = TrainingTaskSpec(
        name="tiny_mlp_w8_d3",
        build_model=width16.build_model,
        build_batch=width16.build_batch,
        loss_fn=width16.loss_fn,
        build_optimizer=width16.build_optimizer,
        workload=forged_workload,
    )
    manifest, slots = build_qualification_slots(
        (forged_task,),
        run_id="factory-mismatch",
        backends=("inductor",),
        methods=("all_save",),
        memory_budgets_bytes=(1 << 20,),
        replicates=1,
        microbatch_size=1,
        device="cpu",
        timeout_s=30.0,
    )
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)

    assert records[0].status == "infra_failure"
    assert "WorkloadSpec does not match" in records[0].error_message


def test_slot_identity_recomputation_rejects_tampering():
    manifest, slots = _slots(methods=("all_save", "peakaware"))
    slot = slots[0]
    workload_config = dict(slot.workload_config)
    workload_config["display_name"] = "forged-name"
    execution_config = dict(slot.execution_config)
    execution_config["device"] = "cuda:7"
    for tampered in (
        replace(slot, workload_fingerprint="f" * 64),
        replace(slot, execution_fingerprint="e" * 64),
        replace(slot, workload_config=FrozenConfig(workload_config)),
        replace(slot, execution_config=FrozenConfig(execution_config)),
        replace(slot, case_id="0" * 64),
        replace(slot, replicate_id="1" * 64),
        replace(slot, attempt_id="2" * 64),
        replace(slot, slot_id="3" * 64),
        replace(slot, backend="inductor"),
        replace(slot, execution_order_index=1 - slot.execution_order_index),
    ):
        try:
            validate_qualification_slot(tampered, manifest)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"tampered slot was accepted: {tampered}")


def test_manifest_cannot_shrink_expected_slots_and_claim_complete_coverage():
    manifest, slots = _slots(methods=("all_save", "peakaware"))
    tampered = json.loads(json.dumps(manifest))
    run = tampered["qualification_run"]
    run["expected_slot_ids"] = run["expected_slot_ids"][:1]
    run["expected_slot_count"] = 1
    run["execution_order"] = run["execution_order"][:1]
    run["pairing_blocks"][0]["slot_ids"] = run["pairing_blocks"][0]["slot_ids"][:1]
    run["pairing_blocks"][0]["method_order"] = run["pairing_blocks"][0]["method_order"][:1]

    try:
        summarize_qualification((), tampered)
    except ValueError as exc:
        assert "Cartesian" in str(exc) or "pairing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shrunken preregistration was accepted")


def test_attempt_index_changes_attempt_and_slot_but_not_paired_replicate():
    first_manifest, first = _slots(attempt_index=0)
    second_manifest, second = _slots(attempt_index=1)

    assert first[0].replicate_id == second[0].replicate_id
    assert first[0].attempt_id != second[0].attempt_id
    assert first[0].slot_id != second[0].slot_id
    assert first_manifest["qualification_run"]["matrix"]["attempt_index"] == 0
    assert second_manifest["qualification_run"]["matrix"]["attempt_index"] == 1


def test_summary_uses_manifest_expected_slots_for_empty_subset_and_complete_coverage(tmp_path: Path):
    manifest, slots = _slots(methods=("all_save", "pytorch_min_cut"), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)

    assert summarize_qualification((), manifest)["complete_slot_coverage"] is False
    subset = summarize_qualification(records[:1], manifest)
    assert subset["complete_slot_coverage"] is False
    assert len(subset["missing_slot_ids"]) == 1
    complete = summarize_qualification(records, manifest)
    assert complete["complete_slot_coverage"] is True
    assert complete["qualification_passed"] is False
    assert complete["method_qualification"]["inductor"]["all_save"]["unsupported"] == 1
    assert complete["required_method_coverage"]["inductor"]["all_save"] is False
    assert [record.slot.slot_id for record in records] == manifest["qualification_run"]["execution_order"]
    with pytest.raises(ValueError, match="execution order"):
        write_qualification_artifacts(
            tuple(reversed(records)),
            manifest,
            output_jsonl=tmp_path / "records.jsonl",
            manifest_json=tmp_path / "manifest.json",
            t1_output=tmp_path / "T-1.md",
        )


def test_runner_rejects_measurement_protocol_different_from_preregistration():
    manifest, slots = _slots()
    for kwargs in ({"timeout_s": 31.0, "repeat_count": 20}, {"timeout_s": 30.0, "repeat_count": 21}):
        try:
            run_qualification_slots(slots, manifest, **kwargs)
        except ValueError as exc:
            assert "preregistration" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("runner accepted a non-preregistered measurement protocol")


def test_inductor_methods_fail_closed_as_unsupported():
    manifest, slots = _slots(methods=("all_save", "pytorch_min_cut"), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)

    assert {record.status for record in records} == {"unsupported"}
    assert all(record.error_stage == "method_prepare" for record in records)
    assert all(record.measurement_raw is None for record in records)
    assert all(record.process_id and record.process_id != 0 for record in records)
    for record in records:
        identity = json.loads(record.runtime_identity)
        assert identity["status"] == "unsupported"
        assert identity["fallback_reason"]


def test_aot_methods_without_matched_budget_or_compiler_protocol_are_explicitly_unsupported():
    methods = ("pytorch_min_cut", "block_ac", "sac")
    manifest, slots = _slots(methods=methods)
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)

    assert [record.slot.slot_id for record in records] == manifest["qualification_run"]["execution_order"]
    assert {record.slot.method for record in records} == set(methods)
    assert all(record.status == "unsupported" for record in records)
    for record in records:
        identity = json.loads(record.runtime_identity)
        assert identity["status"] == "unsupported"
        assert identity["fallback_reason"]


def test_cpu_tiny_runner_preserves_twenty_repeats_and_uses_new_processes():
    manifest, slots = _slots(replicates=2, timeout_s=60.0)
    records = run_qualification_slots(slots, manifest, timeout_s=60.0, repeat_count=20)

    assert len({record.process_id for record in records}) == 2
    assert all(record.status == "runtime_failure" for record in records)
    assert all(json.loads(record.correctness_report)["passed"] for record in records)
    for record in records:
        raw = json.loads(record.measurement_raw)
        assert len(raw["overall_samples"]) == 20
        assert len(raw["phase_samples"]) == 20
        assert len(record.measurement_raw.encode("utf-8")) > 32_768
        assert record.error_stage == "measurement"
        assert record.last_progress_stage == "measurement"
        assert record.elapsed_seconds is not None and record.elapsed_seconds > 0
        identity = json.loads(record.runtime_identity)
        assert identity["compiler_protocol"].endswith("custom_aot_autograd_default_partition")
        assert identity["provenance"]["partition_fn"].endswith("default_partition")
        assert "no-rematerialization" in identity["provenance"]["all_save_naming_scope"]
        assert len(identity["provenance"]["partition_fn_source_sha256"]) == 64
        assert identity["runtime_observations"]["fw_compile_count"] >= 1
        assert identity["runtime_observations"]["bw_compile_count"] >= 1

    source = records[0]
    fake_identity = json.loads(source.runtime_identity)
    fake_identity["method_id"] = "selective_activation_checkpoint"
    forged_method = replace(source, runtime_identity=json.dumps(fake_identity))
    with pytest.raises(ValueError, match="method"):
        validate_qualification_record(forged_method, manifest)

    raw = json.loads(source.measurement_raw)
    aggregate = json.loads(source.measurement_aggregate)
    fractional_raw = json.loads(source.measurement_raw)
    fractional_raw["overall_samples"][0]["overall_peak_bytes"] = 0.5
    fractional_raw["raw_samples"][0]["overall"]["overall_peak_bytes"] = 0.5
    with pytest.raises(ValueError, match="integer"):
        validate_qualification_record(
            replace(source, measurement_raw=json.dumps(fractional_raw)),
            manifest,
        )
    wrong_backend = dict(aggregate)
    wrong_backend["publication_backend"] = "inductor"
    with pytest.raises(ValueError, match="backend"):
        validate_qualification_record(
            replace(source, measurement_aggregate=json.dumps(wrong_backend)),
            manifest,
        )
    aggregate["publication_qualified"] = True
    aggregate["per_run_max_overall_allocated_bytes"] = max(
        sample["overall_peak_bytes"] for sample in raw["overall_samples"]
    )
    short_raw = dict(raw)
    short_raw["overall_samples"] = short_raw["overall_samples"][:1]
    short_raw["phase_samples"] = short_raw["phase_samples"][:1]
    short_success = replace(
        source,
        status="ok",
        error_stage=None,
        error_type=None,
        error_message=None,
        measurement_raw=json.dumps(short_raw),
        measurement_aggregate=json.dumps(aggregate),
    )
    with pytest.raises(ValueError, match="sample count"):
        validate_qualification_record(short_success, manifest)

    wrong_budget = replace(
        source,
        status="budget_violation",
        error_stage=None,
        error_type=None,
        error_message=None,
        measurement_aggregate=json.dumps(aggregate),
    )
    with pytest.raises(ValueError, match="budget status"):
        validate_qualification_record(wrong_budget, manifest)


def test_native_min_cut_absolute_budget_fails_closed_without_conversion_evidence():
    manifest, slots = _slots(methods=("pytorch_min_cut",), timeout_s=90.0)
    records = run_qualification_slots(slots, manifest, timeout_s=90.0, repeat_count=20)

    record = records[0]
    identity = json.loads(record.runtime_identity)
    conversion = identity["provenance"]["budget_conversion"]
    assert record.status == "unsupported"
    assert identity["status"] == "unsupported"
    assert identity["fallback_reason"]
    assert conversion["physical_budget_bytes"] == 1 << 20
    assert conversion["ratio"] is None
    assert conversion["status"] == "unavailable"
    assert record.correctness_report is None
    assert record.measurement_raw is None


def test_absolute_budget_conversion_requires_complete_auditable_evidence():
    unavailable = convert_physical_budget_to_activation_memory_budget(
        800,
        fixed_physical_bytes=None,
        maximum_saved_activation_bytes=None,
    )
    converted = convert_physical_budget_to_activation_memory_budget(
        800,
        fixed_physical_bytes=400,
        maximum_saved_activation_bytes=800,
    )

    assert unavailable["status"] == "unavailable"
    assert unavailable["ratio"] is None
    assert converted["status"] == "converted"
    assert converted["ratio"] == 0.5


def test_parent_timeout_terminates_worker_and_records_null_measurement():
    manifest, slots = _slots(timeout_s=0.001)
    records = run_qualification_slots(slots, manifest, timeout_s=0.001, repeat_count=20)

    record = records[0]
    assert record.status == "timeout"
    assert record.process_id is not None
    assert record.measurement_raw is None
    assert record.correctness_report is None
    assert record.process_id not in {child.pid for child in mp.active_children()}
    assert record.error_stage in {"infra_timeout", "compile_timeout", "runtime_timeout"}
    assert record.elapsed_seconds is not None


def test_timeout_stage_distinguishes_compile_runtime_and_infra():
    assert _timeout_stage("method_prepare") == "compile_timeout"
    assert _timeout_stage("measurement_setup") == "compile_timeout"
    assert _timeout_stage("correctness") == "runtime_timeout"
    assert _timeout_stage("measurement") == "runtime_timeout"
    assert _timeout_stage("spawn") == "infra_timeout"


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup requires POSIX")
def test_process_group_termination_cleans_controlled_descendant():
    context = mp.get_context("spawn")
    marker = f"peakaware-test-{uuid.uuid4().hex}"
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_descendant_process_target, args=(sender, marker))
    process.start()
    sender.close()
    assert receiver.poll(10.0)
    descendant_pids = receiver.recv()
    receiver.close()

    _terminate_process(process, marker)

    assert not process.is_alive()
    assert _marker_process_ids(marker) == ()
    for descendant_pid in descendant_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(descendant_pid, 0)


def test_three_step_correctness_rejects_wrong_output():
    torch.manual_seed(3)
    reference = nn.Linear(4, 2)
    candidate = nn.Linear(4, 2)
    candidate.load_state_dict(reference.state_dict())
    args = (torch.randn(2, 4),)

    report = qualify_three_step_correctness(
        reference,
        candidate,
        lambda value: candidate(value) + 1.0,
        torch.optim.SGD(reference.parameters(), lr=0.1),
        torch.optim.SGD(candidate.parameters(), lr=0.1),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )

    assert report["passed"] is False
    assert report["values_match"] is False
    assert report["first_mismatch"] is not None


def test_three_step_correctness_rejects_nonfinite_optimizer_state():
    class NonFiniteSGD(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            parameter = self.param_groups[0]["params"][0]
            self.state[parameter]["poison"] = torch.tensor(float("inf"))
            return result

    torch.manual_seed(4)
    reference = nn.Linear(4, 2)
    candidate = nn.Linear(4, 2)
    candidate.load_state_dict(reference.state_dict())
    args = (torch.randn(2, 4),)
    report = qualify_three_step_correctness(
        reference,
        candidate,
        candidate,
        torch.optim.SGD(reference.parameters(), lr=0.1),
        NonFiniteSGD(candidate.parameters(), lr=0.1),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )

    assert report["passed"] is False
    assert report["optimizer_state_finite"] is False


def test_three_step_correctness_rejects_rng_drift_with_equal_values():
    torch.manual_seed(5)
    reference = nn.Linear(4, 2)
    candidate = nn.Linear(4, 2)
    candidate.load_state_dict(reference.state_dict())
    args = (torch.randn(2, 4),)

    def consume_rng(value):
        output = candidate(value)
        torch.rand((), device=output.device)
        return output

    report = qualify_three_step_correctness(
        reference,
        candidate,
        consume_rng,
        torch.optim.SGD(reference.parameters(), lr=0.1),
        torch.optim.SGD(candidate.parameters(), lr=0.1),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )

    assert report["passed"] is False
    assert report["values_match"] is True
    assert report["rng_match"] is False


def test_three_step_correctness_rejects_python_and_numpy_rng_drift_and_restores_caller():
    np = __import__("numpy")
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    reference = nn.Linear(4, 2)
    candidate = nn.Linear(4, 2)
    candidate.load_state_dict(reference.state_dict())
    args = (torch.randn(2, 4),)
    caller_python = random.getstate()
    caller_numpy = np.random.get_state()
    caller_torch = torch.get_rng_state().clone()

    def consume_non_torch_rng(value):
        output = candidate(value)
        random.random()
        np.random.random()
        return output

    report = qualify_three_step_correctness(
        reference,
        candidate,
        consume_non_torch_rng,
        torch.optim.SGD(reference.parameters(), lr=0.1),
        torch.optim.SGD(candidate.parameters(), lr=0.1),
        lambda output: output.square().mean(),
        args,
        {},
        device=torch.device("cpu"),
    )

    restored_numpy = np.random.get_state()
    assert report["passed"] is False
    assert report["values_match"] is True
    assert report["python_random_match"] is False
    assert report["numpy_random_match"] is False
    assert random.getstate() == caller_python
    assert np.array_equal(restored_numpy[1], caller_numpy[1])
    assert restored_numpy[0] == caller_numpy[0] and restored_numpy[2:] == caller_numpy[2:]
    assert torch.equal(torch.get_rng_state(), caller_torch)


def test_seeded_rebuild_replays_python_numpy_and_torch_dependent_factories():
    np = __import__("numpy")

    def build_model():
        model = nn.Linear(2, 1)
        value = random.random() + float(np.random.random()) + float(torch.rand(()))
        with torch.no_grad():
            model.weight.fill_(value)
        return model

    def build_batch(batch_size):
        value = random.random() + float(np.random.random()) + float(torch.rand(()))
        return (torch.full((batch_size, 2), value),), {}

    task = TrainingTaskSpec(
        name="seeded",
        build_model=build_model,
        build_batch=build_batch,
        loss_fn=lambda output: output.square().mean(),
        build_optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    first_model, _ = _build_seeded_model_optimizer(task, torch.device("cpu"), 91)
    second_model, _ = _build_seeded_model_optimizer(task, torch.device("cpu"), 91)
    first_batch = _build_seeded_batch(task, 2, torch.device("cpu"), 91)
    second_batch = _build_seeded_batch(task, 2, torch.device("cpu"), 91)

    assert torch.equal(first_model.weight, second_model.weight)
    assert torch.equal(first_batch[0][0], second_batch[0][0])


def test_jsonl_summary_manifest_and_t1_are_immutable(tmp_path: Path):
    manifest, slots = _slots(methods=("all_save",), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)
    output = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    t1_path = tmp_path / "T-1.md"

    summary_path = write_qualification_artifacts(
        records,
        manifest,
        output_jsonl=output,
        manifest_json=manifest_path,
        t1_output=t1_path,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["slot_id"] == slots[0].slot_id
    assert payload["status"] == "unsupported"
    assert payload["measurement_raw"] is None
    committed_summary = validate_qualification_artifact_bundle(
        output_jsonl=output,
        manifest_json=manifest_path,
        t1_output=t1_path,
    )
    expected_summary = summarize_qualification(records, manifest)
    for key, value in expected_summary.items():
        assert committed_summary[key] == value
    assert committed_summary["artifact_commit"]["committed"] is True
    assert len(committed_summary["artifact_commit"]["files"]) == 3
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert "TinyMLP-W8-D3" in t1_path.read_text(encoding="utf-8")

    try:
        write_qualification_artifacts(
            records,
            manifest,
            output_jsonl=output,
            manifest_json=manifest_path,
            t1_output=t1_path,
        )
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("immutable artifact writer overwrote an existing run")


def test_artifact_validator_recomputes_marker_summary_and_validates_jsonl(tmp_path: Path):
    manifest, slots = _slots(methods=("all_save",), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)
    output = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    t1_path = tmp_path / "T-1.md"
    summary_path = write_qualification_artifacts(
        records,
        manifest,
        output_jsonl=output,
        manifest_json=manifest_path,
        t1_output=t1_path,
    )
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for field, value in (("record_count", 99), ("qualification_passed", True)):
        tampered_summary = json.loads(json.dumps(original_summary))
        tampered_summary[field] = value
        summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
        with pytest.raises(ValueError, match="summary"):
            validate_qualification_artifact_bundle(
                output_jsonl=output,
                manifest_json=manifest_path,
                t1_output=t1_path,
            )

    tampered_record = json.loads(output.read_text(encoding="utf-8"))
    tampered_record["task_name"] = "forged-task"
    tampered_output = json.dumps(tampered_record, sort_keys=True) + "\n"
    output.write_text(tampered_output, encoding="utf-8")
    rehashed_summary = json.loads(json.dumps(original_summary))
    output_entry = next(
        entry
        for entry in rehashed_summary["artifact_commit"]["files"]
        if entry["name"] == output.name
    )
    output_entry["size_bytes"] = len(tampered_output.encode("utf-8"))
    output_entry["sha256"] = hashlib.sha256(tampered_output.encode("utf-8")).hexdigest()
    summary_path.write_text(json.dumps(rehashed_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL record"):
        validate_qualification_artifact_bundle(
            output_jsonl=output,
            manifest_json=manifest_path,
            t1_output=t1_path,
        )


def test_artifact_writer_rejects_path_aliases(tmp_path: Path):
    manifest, slots = _slots(methods=("all_save",), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)
    shared = tmp_path / "shared.json"

    try:
        write_qualification_artifacts(
            records,
            manifest,
            output_jsonl=shared,
            manifest_json=tmp_path / "." / "shared.json",
            t1_output=tmp_path / "T-1.md",
        )
    except ValueError as exc:
        assert "alias" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("artifact path alias was accepted")
    assert not shared.exists()


def test_artifact_publish_rolls_back_caught_partial_failure(tmp_path: Path, monkeypatch):
    manifest, slots = _slots(methods=("all_save",), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)
    targets = (
        tmp_path / "records.jsonl",
        tmp_path / "manifest.json",
        tmp_path / "T-1.md",
        tmp_path / "records.jsonl.summary.json",
    )
    real_link = os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return real_link(source, target)

    monkeypatch.setattr(os, "link", fail_second_link)
    try:
        write_qualification_artifacts(
            records,
            manifest,
            output_jsonl=targets[0],
            manifest_json=targets[1],
            t1_output=targets[2],
        )
    except OSError as exc:
        assert "injected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("injected transaction failure did not propagate")
    assert all(not path.exists() for path in targets)
    assert not tuple(path for path in tmp_path.iterdir() if path.suffix == ".tmp")


def test_uncommitted_stale_partial_is_rejected_and_explicitly_recoverable(tmp_path: Path):
    output = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    t1 = tmp_path / "T-1.md"
    output.write_text("partial\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted"):
        validate_qualification_artifact_bundle(
            output_jsonl=output,
            manifest_json=manifest,
            t1_output=t1,
        )
    cleanup_uncommitted_qualification_artifacts(
        output_jsonl=output,
        manifest_json=manifest,
        t1_output=t1,
    )
    assert not output.exists() and not manifest.exists() and not t1.exists()


def test_artifact_writer_rejects_cross_parent_commit_marker_layout(tmp_path: Path):
    manifest, slots = _slots(methods=("all_save",), backends=("inductor",))
    records = run_qualification_slots(slots, manifest, timeout_s=30.0, repeat_count=20)
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(ValueError, match="one parent"):
        write_qualification_artifacts(
            records,
            manifest,
            output_jsonl=tmp_path / "records.jsonl",
            manifest_json=other / "manifest.json",
            t1_output=tmp_path / "T-1.md",
        )


def test_cli_preserves_failure_artifacts_and_exits_nonzero(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    t1 = tmp_path / "T-1.md"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "qualify_publication_runtime.py"),
            "--tasks",
            "tiny_mlp_w8_d3",
            "--backends",
            "inductor",
            "--methods",
            "all_save",
            "--budget-mib",
            "1",
            "--replicates",
            "1",
            "--device",
            "cpu",
            "--timeout",
            "30",
            "--attempt-index",
            "2",
            "--output-jsonl",
            str(output),
            "--manifest-json",
            str(manifest),
            "--t1-output",
            str(t1),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 2
    summary = validate_qualification_artifact_bundle(
        output_jsonl=output,
        manifest_json=manifest,
        t1_output=t1,
    )
    assert summary["complete_slot_coverage"] is True
    assert summary["qualification_passed"] is False
    assert json.loads(manifest.read_text(encoding="utf-8"))["qualification_run"]["matrix"]["attempt_index"] == 2


def test_record_rejects_missing_failure_fields():
    _, slots = _slots()
    try:
        QualificationRecord(slot=slots[0], process_id=None, status="infra_failure")
    except ValueError as exc:
        assert "error" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("failure record accepted null diagnostics")


def test_record_rejects_unqualified_success_and_unsupported_without_identity():
    _, slots = _slots()
    for status in ("ok", "budget_violation", "unsupported"):
        try:
            QualificationRecord(
                slot=slots[0],
                process_id=1,
                status=status,
                error_stage=None if status != "unsupported" else "method_prepare",
                error_type=None if status != "unsupported" else "Unsupported",
                error_message=None if status != "unsupported" else "missing identity",
            )
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid {status} record was accepted")
