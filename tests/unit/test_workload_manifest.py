from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from peakaware.contracts import ExecutionSpec, FrozenConfig
from peakaware.models import (
    build_bert_base_task,
    build_gpt2_task,
    build_resnet50_task,
    build_vit_b16_task,
)
from peakaware.workload_manifest import (
    attempt_fingerprint,
    build_manifest_entry,
    build_workload_manifest,
    canonical_json,
    case_id,
    execution_config_fingerprint,
    replicate_fingerprint,
    render_t1_markdown,
    validate_record_manifest_reference,
    workload_fingerprint,
)


def _fingerprint(task, *, microbatch_size: int = 2) -> str:
    assert task.workload is not None
    return workload_fingerprint(task.workload, microbatch_size=microbatch_size)


def _execution(backend: str = "aot_eager") -> ExecutionSpec:
    return ExecutionSpec(
        schema_version="1.0",
        backend=backend,
        device="cpu",
        compiler_protocol="aot-autograd-v1",
        precision_protocol="fp32-v1",
        measurement_protocol="publication-v1",
    )


def test_workload_fingerprint_tracks_model_input_optimizer_and_microbatch_but_not_seed():
    base = build_bert_base_task(sequence_length=16, hidden_size=32)

    assert _fingerprint(base) != _fingerprint(build_bert_base_task(sequence_length=16, hidden_size=48))
    assert _fingerprint(base) != _fingerprint(build_bert_base_task(sequence_length=24, hidden_size=32))
    assert _fingerprint(base, microbatch_size=2) != _fingerprint(base, microbatch_size=4)

    assert base.workload is not None
    changed_optimizer = replace(
        base.workload,
        optimizer_config=FrozenConfig({"name": "AdamW", "lr": 0.001}),
    )
    assert workload_fingerprint(changed_optimizer, microbatch_size=2) != _fingerprint(base)
    changed_dtype = replace(
        base.workload,
        input_config=FrozenConfig({**dict(base.workload.input_config), "dtype": "torch.int32"}),
    )
    assert workload_fingerprint(changed_dtype, microbatch_size=2) != _fingerprint(base)
    bf16_compute = replace(base.workload, compute_dtype="torch.bfloat16")
    bf16_parameters = replace(base.workload, parameter_dtype="torch.bfloat16")
    assert workload_fingerprint(bf16_compute, microbatch_size=2) != _fingerprint(base)
    assert workload_fingerprint(bf16_parameters, microbatch_size=2) != _fingerprint(base)


def test_execution_and_case_identity_have_separate_boundaries():
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
    workload_id = _fingerprint(task)
    eager_id = execution_config_fingerprint(_execution("aot_eager"))
    inductor_id = execution_config_fingerprint(_execution("inductor"))

    assert eager_id != inductor_id
    assert workload_id == _fingerprint(task)
    base_case = case_id(workload_id, eager_id, memory_budget_bytes=1000, strategy="all_save")
    assert base_case != case_id(workload_id, eager_id, memory_budget_bytes=900, strategy="all_save")
    assert base_case != case_id(workload_id, eager_id, memory_budget_bytes=1000, strategy="peakaware")
    assert base_case != case_id(workload_id, inductor_id, memory_budget_bytes=1000, strategy="all_save")


def test_replicate_and_attempt_identity_are_separate():
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
    workload_id = _fingerprint(task)
    execution_id = execution_config_fingerprint(_execution())
    stable_case_id = case_id(workload_id, execution_id, memory_budget_bytes=1000, strategy="peakaware")
    base_replicate = replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=1000,
        seed=7,
        replicate_index=0,
    )

    assert stable_case_id == case_id(workload_id, execution_id, memory_budget_bytes=1000, strategy="peakaware")
    assert base_replicate != replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=1000,
        seed=8,
        replicate_index=0,
    )
    assert base_replicate != replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=1000,
        seed=7,
        replicate_index=1,
    )
    assert base_replicate == replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=1000,
        seed=7,
        replicate_index=0,
    )
    assert attempt_fingerprint(base_replicate, attempt_id=0) != attempt_fingerprint(
        base_replicate,
        attempt_id=1,
    )
    assert base_replicate != replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=900,
        seed=7,
        replicate_index=0,
    )
    all_save_case = case_id(workload_id, execution_id, memory_budget_bytes=1000, strategy="all_save")
    peakaware_case = case_id(workload_id, execution_id, memory_budget_bytes=1000, strategy="peakaware")
    assert all_save_case != peakaware_case
    assert base_replicate == replicate_fingerprint(
        workload_id,
        execution_id,
        memory_budget_bytes=1000,
        seed=7,
        replicate_index=0,
    )


@pytest.mark.parametrize("extra_key", ("dtype", "model", "optimizer", "budget", "strategy", "seed", "attempt"))
def test_execution_spec_constructor_rejects_non_schema_fields(extra_key):
    with pytest.raises(TypeError):
        ExecutionSpec(
            schema_version="1.0",
            backend="aot_eager",
            device="cpu",
            compiler_protocol="aot-autograd-v1",
            precision_protocol="fp32-v1",
            measurement_protocol="publication-v1",
            **{extra_key: "forbidden"},
        )


def test_execution_fingerprint_rejects_untyped_mapping():
    with pytest.raises(TypeError, match="requires an ExecutionSpec"):
        execution_config_fingerprint({"backend": "aot_eager"})


def test_execution_fingerprint_rejects_subclasses_with_hidden_fields():
    class PollutedExecutionSpec(ExecutionSpec):
        method = "peakaware"

    polluted = PollutedExecutionSpec(
        schema_version="1.0",
        backend="aot_eager",
        device="cuda",
        compiler_protocol="aot-lowered-v1",
        precision_protocol="fp32-v1",
        measurement_protocol="publication-v1",
    )

    with pytest.raises(TypeError, match="exact type"):
        execution_config_fingerprint(polluted)


def test_canonical_json_and_fingerprint_are_stable_across_order_and_roundtrip():
    first = {"z": [3, {"b": True, "a": None}], "a": 1}
    second = {"a": 1, "z": [3, {"a": None, "b": True}]}

    encoded = canonical_json(first)
    assert encoded == canonical_json(second)
    assert encoded == canonical_json(json.loads(encoded))
    assert execution_config_fingerprint(_execution()) == execution_config_fingerprint(_execution())


def test_manifest_rejects_declared_model_configuration_mismatch():
    task = build_resnet50_task(image_size=32, num_classes=10)
    assert task.workload is not None
    incorrect_model_config = FrozenConfig({**dict(task.workload.model_config), "num_classes": 11})
    incorrect_task = replace(task, workload=replace(task.workload, model_config=incorrect_model_config))

    with pytest.raises(ValueError, match="resnet num_classes mismatch"):
        build_manifest_entry(incorrect_task, microbatch_size=1, seed=0)


def test_manifest_rejects_compute_dtype_that_does_not_match_actual_model():
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
    assert task.workload is not None
    false_bf16_task = replace(task, workload=replace(task.workload, compute_dtype="torch.bfloat16"))

    with pytest.raises(ValueError, match="compute dtype mismatch"):
        build_manifest_entry(false_bf16_task, microbatch_size=1, seed=0)


@pytest.mark.parametrize(
    ("task", "forbidden_name", "error"),
    (
        (
            build_bert_base_task(
                sequence_length=8,
                vocab_size=101,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
            ),
            "BERT-Base",
            "bert display_name mismatch",
        ),
        (
            build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4),
            "GPT-2",
            "gpt2 display_name mismatch",
        ),
    ),
)
def test_manifest_rejects_standard_name_for_tiny_like_configuration(task, forbidden_name, error):
    assert task.workload is not None
    incorrect_task = replace(task, workload=replace(task.workload, display_name=forbidden_name))

    with pytest.raises(ValueError, match=error):
        build_manifest_entry(incorrect_task, microbatch_size=1, seed=0)


def test_manifest_rejects_incomplete_optimizer_or_incorrect_loss_reduction():
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
    assert task.workload is not None
    incomplete_optimizer = replace(
        task,
        workload=replace(task.workload, optimizer_config=FrozenConfig({"name": "AdamW", "lr": 0.0001})),
    )
    incorrect_reduction = replace(
        task,
        workload=replace(
            task.workload,
            loss_config=FrozenConfig({"name": "logits_squared_mean", "reduction": "sum"}),
        ),
    )

    with pytest.raises(ValueError, match="optimizer defaults mismatch"):
        build_manifest_entry(incomplete_optimizer, microbatch_size=1, seed=0)
    with pytest.raises(ValueError, match="loss reduction mismatch"):
        build_manifest_entry(incorrect_reduction, microbatch_size=1, seed=0)


def test_workload_configuration_is_recursively_immutable():
    spec = build_resnet50_task().workload
    assert spec is not None

    with pytest.raises(TypeError):
        spec.model_config["num_classes"] = 11
    assert isinstance(spec.model_config["block_counts"], tuple)


@pytest.mark.parametrize(
    ("task", "display_name", "parameter_count", "shape", "dtype"),
    (
        (build_resnet50_task(), "ResNet-50", 23_528_522, [2, 3, 224, 224], "torch.float32"),
        (build_vit_b16_task(), "ViT-B/16", 85_806_346, [2, 3, 224, 224], "torch.float32"),
        (build_bert_base_task(), "BERT-like-2L-64H", 2_090_690, [2, 32], "torch.int64"),
        (build_gpt2_task(), "GPT2-like-2L-64H", 6_535_040, [2, 32], "torch.int64"),
    ),
)
def test_main_workload_manifest_matches_actual_model(
    task,
    display_name: str,
    parameter_count: int,
    shape: list[int],
    dtype: str,
):
    entry = build_manifest_entry(task, microbatch_size=2, seed=0)

    assert entry["workload"]["display_name"] == display_name
    assert entry["manifest_entry_id"] == entry["workload_fingerprint"]
    assert entry["workload"]["compute_dtype"] == "torch.float32"
    assert entry["workload"]["parameter_dtype"] == "torch.float32"
    assert entry["parameter_count"] == parameter_count
    assert entry["trainable_parameter_count"] == parameter_count
    assert entry["batch_inputs"] == [{"path": "args[0]", "shape": shape, "dtype": dtype}]
    assert entry["parameter_count_method"] == "sum_numel_over_unique_parameter_objects"
    assert entry["actual_optimizer_defaults"] == entry["workload"]["optimizer"]
    assert "distribution" in entry["workload"]["input"]


def test_t1_separates_token_input_dtype_from_bert_and_gpt_compute_dtype():
    entries = [
        build_manifest_entry(
            build_bert_base_task(
                sequence_length=8,
                vocab_size=101,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
            ),
            microbatch_size=3,
            seed=11,
        ),
        build_manifest_entry(
            build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4),
            microbatch_size=3,
            seed=11,
        ),
    ]

    table = render_t1_markdown({"schema_version": "1.0", "workloads": entries})

    assert "BERT-like-1L-16H" in table
    assert "GPT2-like-1L-16H" in table
    assert "`3x8`" in table
    assert "AdamW (lr=0.0001)" in table
    assert all(str(entry["parameter_count"]) in table for entry in entries)
    assert "seed" not in table.lower()
    assert "Input dtype" in table
    assert "Compute dtype" in table
    assert "`torch.int64`" in table
    assert "`torch.float32`" in table
    assert "| 8 | N/A | 101 |" in table
    bert_row = next(line for line in table.splitlines() if line.startswith("| BERT-like"))
    gpt_row = next(line for line in table.splitlines() if line.startswith("| GPT2-like"))
    assert "`torch.int64` | `torch.float32` | `torch.float32`" in bert_row
    assert "`torch.int64` | `torch.float32` | `torch.float32`" in gpt_row


def test_manifest_seed_is_reproducibility_metadata_not_workload_identity():
    task = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)

    first = build_manifest_entry(task, microbatch_size=2, seed=7)
    second = build_manifest_entry(task, microbatch_size=2, seed=8)

    assert first["seed"] == 7
    assert second["seed"] == 8
    assert first["workload_fingerprint"] == second["workload_fingerprint"]


def test_same_registry_key_can_have_multiple_sequence_entries_but_not_duplicates():
    short = build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4)
    long = build_gpt2_task(sequence_length=16, vocab_size=101, n_embd=16, n_layer=1, n_head=4)

    manifest = build_workload_manifest(
        [short, long],
        microbatch_size=2,
        seed=0,
        compiler_mode="aot_eager",
        execution_config=_execution(),
    )

    assert [entry["workload"]["registry_key"] for entry in manifest["workloads"]] == ["gpt2", "gpt2"]
    assert len({entry["manifest_entry_id"] for entry in manifest["workloads"]}) == 2
    environment = manifest["environment"]
    assert environment["execution"]["compiler_mode"] == "aot_eager"
    assert environment["execution"]["parameters"]["backend"] == "aot_eager"
    for value_key, reason_key in (
        ("git_commit", "git_commit_reason"),
        ("git_dirty", "git_dirty_reason"),
        ("cuda_version", "cuda_version_reason"),
        ("gpu_devices", "gpu_devices_reason"),
        ("driver_version", "driver_version_reason"),
    ):
        if environment[value_key] is None or environment[value_key] == []:
            assert environment[reason_key]
    for device in environment["gpu_devices"]:
        if device["uuid"] is None:
            assert device["uuid_reason"]
    with pytest.raises(ValueError, match="duplicate fingerprints"):
        build_workload_manifest([short, short], microbatch_size=2, seed=0)


def test_record_reference_validator_rejects_unknown_name_and_config_mismatches():
    entry = build_manifest_entry(
        build_gpt2_task(sequence_length=8, vocab_size=101, n_embd=16, n_layer=1, n_head=4),
        microbatch_size=2,
        seed=0,
    )
    manifest = {"schema_version": "1.0", "workloads": [entry]}
    record = {
        "manifest_entry_id": entry["manifest_entry_id"],
        "workload_fingerprint": entry["workload_fingerprint"],
        "display_name": entry["workload"]["display_name"],
        "workload_config": entry["workload"],
    }

    assert validate_record_manifest_reference(manifest, record) is entry
    assert validate_record_manifest_reference(
        manifest,
        {
            "manifest_entry_id": entry["manifest_entry_id"],
            "workload_fingerprint": entry["workload_fingerprint"],
            "display_name": entry["workload"]["display_name"],
            "config": entry["workload"],
        },
    ) is entry
    with pytest.raises(ValueError, match="unknown workload fingerprint"):
        validate_record_manifest_reference(manifest, {**record, "workload_fingerprint": "unknown"})
    with pytest.raises(ValueError, match="record manifest_entry_id mismatch"):
        validate_record_manifest_reference(manifest, {**record, "manifest_entry_id": "wrong"})
    with pytest.raises(ValueError, match="requires manifest_entry_id"):
        missing_entry_id = {key: value for key, value in record.items() if key != "manifest_entry_id"}
        validate_record_manifest_reference(manifest, missing_entry_id)
    with pytest.raises(ValueError, match="record display_name mismatch"):
        validate_record_manifest_reference(manifest, {**record, "display_name": "GPT-2"})
    with pytest.raises(ValueError, match="record workload_config mismatch"):
        validate_record_manifest_reference(
            manifest,
            {**record, "workload_config": {**entry["workload"], "compute_dtype": "torch.bfloat16"}},
        )


def test_export_workload_manifest_cli_writes_deterministic_json_and_t1(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    first_json = tmp_path / "first.json"
    first_t1 = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_t1 = tmp_path / "second.md"
    common = [
        sys.executable,
        str(repo_root / "scripts" / "export_workload_manifest.py"),
        "--tasks",
        "tiny_residual_w8",
        "--microbatch-size",
        "2",
        "--seed",
        "5",
        "--compiler-mode",
        "aot_eager",
    ]

    subprocess.run(common + ["--output", str(first_json), "--t1-output", str(first_t1)], cwd=repo_root, check=True)
    subprocess.run(common + ["--output", str(second_json), "--t1-output", str(second_t1)], cwd=repo_root, check=True)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_t1.read_bytes() == second_t1.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["workloads"][0]["workload"]["registry_key"] == "tiny_residual_w8"
    assert payload["environment"]["execution"]["compiler_mode"] == "aot_eager"
    assert payload["environment"]["git_commit"]
    assert payload["environment"]["python_version"]
    assert payload["environment"]["torch_version"]
    assert "TinyResidual-W8" in first_t1.read_text(encoding="utf-8")
    assert "## Environment" in first_t1.read_text(encoding="utf-8")
