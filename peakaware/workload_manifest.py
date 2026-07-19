from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from peakaware.contracts import ExecutionSpec, TrainingTaskSpec, WorkloadSpec


MANIFEST_SCHEMA_VERSION = "1.0"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, WorkloadSpec):
        return workload_spec_to_dict(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not support non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    separators = (",", ":") if indent is None else None
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        indent=indent,
        separators=separators,
        sort_keys=True,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def workload_spec_to_dict(spec: WorkloadSpec) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "registry_key": spec.registry_key,
        "display_name": spec.display_name,
        "model_family": spec.model_family,
        "implementation": spec.implementation,
        "model": _canonical_value(spec.model_config),
        "input": _canonical_value(spec.input_config),
        "optimizer": _canonical_value(spec.optimizer_config),
        "loss": _canonical_value(spec.loss_config),
        "compute_dtype": spec.compute_dtype,
        "parameter_dtype": spec.parameter_dtype,
    }


def workload_fingerprint(
    spec: WorkloadSpec,
    *,
    microbatch_size: int,
) -> str:
    if not isinstance(microbatch_size, int) or isinstance(microbatch_size, bool) or microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    return _fingerprint(
        {
            "identity_kind": "workload",
            "workload": workload_spec_to_dict(spec),
            "microbatch_size": microbatch_size,
        }
    )


def execution_spec_to_dict(spec: ExecutionSpec) -> dict[str, str]:
    return {
        "schema_version": spec.schema_version,
        "backend": spec.backend,
        "device": spec.device,
        "compiler_protocol": spec.compiler_protocol,
        "precision_protocol": spec.precision_protocol,
        "measurement_protocol": spec.measurement_protocol,
    }


def execution_config_fingerprint(config: ExecutionSpec) -> str:
    if type(config) is not ExecutionSpec:
        raise TypeError("execution_config_fingerprint requires an ExecutionSpec of exact type")
    return _fingerprint({"identity_kind": "execution", "config": execution_spec_to_dict(config)})


def case_id(
    workload_fingerprint: str,
    execution_config_fingerprint: str,
    *,
    memory_budget_bytes: int,
    strategy: str,
) -> str:
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if not strategy:
        raise ValueError("strategy must not be empty")
    return _fingerprint(
        {
            "identity_kind": "case",
            "workload_fingerprint": workload_fingerprint,
            "execution_config_fingerprint": execution_config_fingerprint,
            "memory_budget_bytes": memory_budget_bytes,
            "strategy": strategy,
        }
    )


def replicate_fingerprint(
    workload_fingerprint: str,
    execution_config_fingerprint: str,
    *,
    memory_budget_bytes: int,
    seed: int,
    replicate_index: int,
) -> str:
    if not isinstance(memory_budget_bytes, int) or isinstance(memory_budget_bytes, bool) or memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(replicate_index, int) or isinstance(replicate_index, bool) or replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    return _fingerprint(
        {
            "identity_kind": "replicate",
            "workload_fingerprint": workload_fingerprint,
            "execution_config_fingerprint": execution_config_fingerprint,
            "memory_budget_bytes": memory_budget_bytes,
            "seed": seed,
            "replicate_index": replicate_index,
        }
    )


def attempt_fingerprint(replicate_fingerprint: str, *, attempt_id: str | int) -> str:
    if not isinstance(attempt_id, (str, int)) or isinstance(attempt_id, bool):
        raise TypeError("attempt_id must be a string or integer")
    if isinstance(attempt_id, str) and not attempt_id:
        raise ValueError("attempt_id must not be empty")
    if isinstance(attempt_id, int) and attempt_id < 0:
        raise ValueError("numeric attempt_id must be non-negative")
    return _fingerprint(
        {
            "identity_kind": "attempt",
            "replicate_fingerprint": replicate_fingerprint,
            "attempt_id": attempt_id,
        }
    )


def _qualified_class_name(model: nn.Module) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}.values()
    parameter_count = sum(parameter.numel() for parameter in unique_parameters)
    trainable_parameter_count = sum(parameter.numel() for parameter in unique_parameters if parameter.requires_grad)
    return parameter_count, trainable_parameter_count


def _tensor_record(path: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "path": path,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _batch_records(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, value in enumerate(args):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch args[{index}] is not a tensor")
        records.append(_tensor_record(f"args[{index}]", value))
    for key in sorted(kwargs):
        value = kwargs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch kwargs[{key!r}] is not a tensor")
        records.append(_tensor_record(f"kwargs.{key}", value))
    return records


def _require_equal(label: str, declared: Any, actual: Any) -> None:
    if declared != actual:
        raise ValueError(f"{label} mismatch: declared {declared!r}, actual {actual!r}")


def _validate_dtype_name(label: str, dtype_name: str) -> None:
    if not dtype_name.startswith("torch."):
        raise ValueError(f"invalid {label} {dtype_name!r}")
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"invalid {label} {dtype_name!r}")


def _validate_model(spec: WorkloadSpec, model: nn.Module) -> None:
    config = spec.model_config
    implementation = spec.implementation
    if implementation == "torchvision.models.resnet50":
        _require_equal("resnet architecture", config["architecture"], type(model).__name__)
        _require_equal(
            "resnet block counts",
            tuple(config["block_counts"]),
            tuple(len(getattr(model, f"layer{i}")) for i in range(1, 5)),
        )
        _require_equal("resnet block", config["block"], type(model.layer1[0]).__name__)
        _require_equal("resnet num_classes", config["num_classes"], model.fc.out_features)
        _require_equal("resnet groups", config["groups"], model.layer1[0].conv2.groups)
        _require_equal("resnet width_per_group", config["width_per_group"], model.layer1[0].conv2.out_channels)
        _require_equal("resnet norm_layer", config["norm_layer"], "torch.nn.BatchNorm2d")
        _require_equal("resnet display_name", spec.display_name, "ResNet-50")
    elif implementation == "torchvision.models.vit_b_16":
        _require_equal("vit architecture", config["architecture"], type(model).__name__)
        _require_equal("vit image_size", config["image_size"], model.image_size)
        _require_equal("vit patch_size", config["patch_size"], model.patch_size)
        _require_equal("vit hidden_dim", config["hidden_dim"], model.hidden_dim)
        _require_equal("vit num_layers", config["num_layers"], len(model.encoder.layers))
        _require_equal("vit num_heads", config["num_heads"], model.encoder.layers[0].self_attention.num_heads)
        _require_equal("vit mlp_dim", config["mlp_dim"], model.encoder.layers[0].mlp[0].out_features)
        _require_equal("vit num_classes", config["num_classes"], model.heads.head.out_features)
        _require_equal("vit dropout", config["dropout"], model.encoder.dropout.p)
        _require_equal(
            "vit attention_dropout",
            config["attention_dropout"],
            model.encoder.layers[0].self_attention.dropout,
        )
        _require_equal(
            "vit representation_size",
            config["representation_size"],
            getattr(model, "representation_size", None),
        )
        _require_equal("vit display_name", spec.display_name, "ViT-B/16")
    elif implementation == "transformers.BertForSequenceClassification":
        for key in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "num_labels",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "max_position_embeddings",
            "type_vocab_size",
            "layer_norm_eps",
            "classifier_dropout",
            "pad_token_id",
        ):
            _require_equal(f"bert {key}", config[key], getattr(model.config, key))
        expected_name = f"BERT-like-{model.config.num_hidden_layers}L-{model.config.hidden_size}H"
        _require_equal("bert display_name", spec.display_name, expected_name)
    elif implementation == "peakaware.models.registry.GPT2Like":
        _require_equal("gpt2 vocab_size", config["vocab_size"], model.token_embedding.num_embeddings)
        _require_equal("gpt2 width", config["width"], model.token_embedding.embedding_dim)
        _require_equal("gpt2 sequence_length", config["sequence_length"], model.position_embedding.num_embeddings)
        _require_equal("gpt2 num_layers", config["num_layers"], len(model.blocks))
        _require_equal("gpt2 num_heads", config["num_heads"], model.blocks[0].num_heads)
        _require_equal(
            "gpt2 mlp_ratio",
            config["mlp_ratio"],
            model.blocks[0].mlp[0].out_features // model.token_embedding.embedding_dim,
        )
        _require_equal("gpt2 lm_head_bias", config["lm_head_bias"], model.lm_head.bias is not None)
        _require_equal(
            "gpt2 tie_word_embeddings",
            config["tie_word_embeddings"],
            model.token_embedding.weight is model.lm_head.weight,
        )
        expected_name = f"GPT2-like-{len(model.blocks)}L-{model.token_embedding.embedding_dim}H"
        _require_equal("gpt2 display_name", spec.display_name, expected_name)
    elif implementation.endswith(".TinyResidual"):
        _require_equal("tiny residual width", config["width"], model.a.in_features)
        _require_equal("tiny residual display_name", spec.display_name, f"TinyResidual-W{model.a.in_features}")
    elif implementation.endswith(".TinyMLP"):
        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        _require_equal("tiny mlp depth", config["depth"], len(linear_layers) - 1)
        _require_equal("tiny mlp width", config["width"], linear_layers[0].in_features)
        _require_equal(
            "tiny mlp display_name",
            spec.display_name,
            f"TinyMLP-W{linear_layers[0].in_features}-D{len(linear_layers) - 1}",
        )
    elif implementation.endswith(".TinyAttentionBlock"):
        _require_equal("tiny attention width", config["width"], model.q.in_features)
        sequence_length = spec.input_config["shape_without_batch"][0]
        _require_equal(
            "tiny attention display_name",
            spec.display_name,
            f"TinyAttention-W{model.q.in_features}-S{sequence_length}",
        )
    else:
        raise ValueError(f"no workload validator for implementation {implementation!r}")


def _validate_task(
    task: TrainingTaskSpec,
    model: nn.Module,
    batch: Sequence[dict[str, Any]],
    microbatch_size: int,
) -> dict[str, Any]:
    if task.workload is None:
        raise ValueError(f"task {task.name!r} has no workload spec")
    spec = task.workload
    _require_equal("registry key", spec.registry_key, task.name)
    if len(batch) != 1:
        raise ValueError("publication workloads must declare exactly one tensor input")
    expected_shape = [microbatch_size, *spec.input_config["shape_without_batch"]]
    _require_equal("batch input shape", expected_shape, batch[0]["shape"])
    _require_equal("batch input dtype", spec.input_config["dtype"], batch[0]["dtype"])
    _validate_input_distribution(spec)
    _validate_model(spec, model)

    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if len(parameter_dtypes) != 1:
        raise ValueError(f"model has mixed parameter dtypes: {parameter_dtypes}")
    _require_equal("parameter dtype", spec.parameter_dtype, parameter_dtypes[0])
    _validate_dtype_name("compute dtype", spec.compute_dtype)
    _validate_dtype_name("parameter dtype", spec.parameter_dtype)
    _require_equal("compute dtype", spec.compute_dtype, parameter_dtypes[0])

    optimizer = task.build_optimizer(model)
    actual_optimizer = {"name": type(optimizer).__name__, **_canonical_value(optimizer.defaults)}
    declared_optimizer = _canonical_value(spec.optimizer_config)
    _require_equal("optimizer defaults", declared_optimizer, actual_optimizer)
    expected_loss_functions = {
        "squared_mean": "squared_mean_loss",
        "logits_squared_mean": "logits_squared_mean_loss",
    }
    declared_loss = spec.loss_config["name"]
    if declared_loss not in expected_loss_functions:
        raise ValueError(f"unknown declared loss {declared_loss!r}")
    _require_equal("loss function", expected_loss_functions[declared_loss], task.loss_fn.__name__)
    _require_equal("loss reduction", spec.loss_config["reduction"], "mean")
    return actual_optimizer


def _validate_input_distribution(spec: WorkloadSpec) -> None:
    distribution = spec.input_config.get("distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("input distribution must be a structured mapping")
    if spec.input_config["kind"] == "token_ids":
        expected = {"name": "randint", "low": 0, "high_exclusive": spec.input_config["vocab_size"]}
    else:
        expected = {"name": "normal", "mean": 0.0, "std": 1.0}
    _require_equal("input distribution", _canonical_value(distribution), expected)


def build_manifest_entry(
    task: TrainingTaskSpec,
    *,
    microbatch_size: int,
    seed: int,
) -> dict[str, Any]:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if task.workload is None:
        raise ValueError(f"task {task.name!r} has no workload spec")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = task.build_model()
        args, kwargs = task.build_batch(microbatch_size)
    batch = _batch_records(args, kwargs)
    actual_optimizer_defaults = _validate_task(task, model, batch, microbatch_size)
    parameter_count, trainable_parameter_count = _parameter_counts(model)
    return {
        "manifest_entry_id": workload_fingerprint(
            task.workload,
            microbatch_size=microbatch_size,
        ),
        "workload_fingerprint": workload_fingerprint(
            task.workload,
            microbatch_size=microbatch_size,
        ),
        "workload": workload_spec_to_dict(task.workload),
        "microbatch_size": microbatch_size,
        "seed": seed,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_count_method": "sum_numel_over_unique_parameter_objects",
        "model_class": _qualified_class_name(model),
        "batch_inputs": batch,
        "actual_optimizer_defaults": actual_optimizer_defaults,
    }


def build_workload_manifest(
    tasks: Sequence[TrainingTaskSpec],
    *,
    microbatch_size: int,
    seed: int,
    compiler_mode: str | None = None,
    execution_config: ExecutionSpec | None = None,
) -> dict[str, Any]:
    entries = [
        build_manifest_entry(task, microbatch_size=microbatch_size, seed=seed)
        for task in tasks
    ]
    entries.sort(key=lambda entry: (entry["workload"]["registry_key"], entry["manifest_entry_id"]))
    fingerprints = [entry["workload_fingerprint"] for entry in entries]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("workload manifest contains duplicate fingerprints")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "environment": collect_environment_snapshot(
            compiler_mode=compiler_mode,
            execution_config=execution_config,
        ),
        "workloads": entries,
    }


def validate_record_manifest_reference(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_entry_id = record.get("manifest_entry_id")
    workload_fingerprint = record.get("workload_fingerprint")
    display_name = record.get("display_name")
    workload_config = record.get("workload_config", record.get("workload", record.get("config")))
    if not manifest_entry_id or not workload_fingerprint or display_name is None or workload_config is None:
        raise ValueError(
            "record reference requires manifest_entry_id, workload_fingerprint, display_name, and workload_config"
        )

    entries = manifest.get("workloads", [])
    for entry in entries:
        _require_equal(
            "manifest entry identity",
            entry.get("manifest_entry_id"),
            entry.get("workload_fingerprint"),
        )
    matches = [entry for entry in entries if entry.get("workload_fingerprint") == workload_fingerprint]
    if not matches:
        raise ValueError(f"unknown workload fingerprint: {workload_fingerprint}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous workload fingerprint: {workload_fingerprint}")
    entry = matches[0]
    _require_equal("record manifest_entry_id", manifest_entry_id, entry["manifest_entry_id"])
    _require_equal("record display_name", display_name, entry["workload"]["display_name"])
    _require_equal(
        "record workload_config",
        _canonical_value(workload_config),
        _canonical_value(entry["workload"]),
    )
    return entry


validate_record_reference = validate_record_manifest_reference


def _run_command(args: Sequence[str], cwd: Path | None = None) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"
    return result.stdout.strip(), None


def collect_environment_snapshot(
    *,
    compiler_mode: str | None,
    execution_config: ExecutionSpec | None = None,
) -> dict[str, Any]:
    if execution_config is not None and type(execution_config) is not ExecutionSpec:
        raise TypeError("execution_config must be an ExecutionSpec of exact type")
    parameters = execution_spec_to_dict(execution_config) if execution_config is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    git_commit, git_commit_reason = _run_command(("git", "rev-parse", "HEAD"), repo_root)
    git_status, git_dirty_reason = _run_command(("git", "status", "--porcelain"), repo_root)
    git_dirty = None if git_status is None else bool(git_status)

    cuda_version = torch.version.cuda
    cuda_version_reason = None if cuda_version is not None else "PyTorch was built without CUDA support"
    cuda_available = torch.cuda.is_available()
    gpu_devices: list[dict[str, Any]] = []
    gpu_devices_reason: str | None = None
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            raw_uuid = getattr(properties, "uuid", None)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": str(raw_uuid) if raw_uuid is not None else None,
                    "uuid_reason": None if raw_uuid is not None else "CUDA device properties do not expose UUID",
                }
            )
    else:
        gpu_devices_reason = "torch.cuda.is_available() is false"

    nvidia_smi, nvidia_smi_reason = _run_command(
        ("nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader,nounits")
    )
    driver_version: str | None = None
    driver_version_reason = nvidia_smi_reason
    if nvidia_smi:
        rows = [row.split(", ") for row in nvidia_smi.splitlines() if row.strip()]
        if rows and all(len(row) == 4 for row in rows):
            driver_versions = sorted({row[3] for row in rows})
            driver_version = ",".join(driver_versions)
            driver_version_reason = None
            by_index = {int(row[0]): row for row in rows}
            if not gpu_devices:
                gpu_devices = [
                    {
                        "index": int(row[0]),
                        "name": row[1],
                        "uuid": row[2],
                        "uuid_reason": None,
                    }
                    for row in rows
                ]
                gpu_devices_reason = None
            for device in gpu_devices:
                row = by_index.get(device["index"])
                if row is not None:
                    device["name"] = row[1]
                    device["uuid"] = row[2]
                    device["uuid_reason"] = None
        else:
            driver_version_reason = "nvidia-smi returned an unexpected row format"
    elif nvidia_smi_reason is None:
        driver_version_reason = "nvidia-smi returned no GPU rows"
    if not gpu_devices and gpu_devices_reason is None:
        gpu_devices_reason = "no GPU devices were reported"

    return {
        "git_commit": git_commit,
        "git_commit_reason": git_commit_reason,
        "git_dirty": git_dirty,
        "git_dirty_reason": git_dirty_reason,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cuda_version_reason": cuda_version_reason,
        "gpu_devices": gpu_devices,
        "gpu_devices_reason": gpu_devices_reason,
        "driver_version": driver_version,
        "driver_version_reason": driver_version_reason,
        "execution": {
            "compiler_mode": compiler_mode,
            "compiler_mode_reason": None if compiler_mode is not None else "compiler mode was not supplied",
            "parameters": parameters,
            "parameters_reason": None if parameters else "execution parameters were not supplied",
        },
    }


def render_t1_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "| Workload | Entry ID | Registry key | Parameters | Trainable | Layers/blocks | Hidden | Heads | "
        "Sequence | Image size | Vocab | Input | Microbatch | Input dtype | Compute dtype | Parameter dtype | "
        "Optimizer | Loss |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---|---:|---|---|---|---|---|",
    ]
    for entry in manifest["workloads"]:
        workload = entry["workload"]
        model = workload["model"]
        batch_input = entry["batch_inputs"][0]
        optimizer = workload["optimizer"]
        loss = workload["loss"]
        input_config = workload["input"]
        layers = model.get("num_hidden_layers", model.get("num_layers", model.get("block_counts", "N/A")))
        if isinstance(layers, list):
            layers = "/".join(str(value) for value in layers)
        hidden = model.get("hidden_size", model.get("hidden_dim", model.get("width", "N/A")))
        heads = model.get("num_attention_heads", model.get("num_heads", "N/A"))
        shape_without_batch = input_config["shape_without_batch"]
        sequence = shape_without_batch[0] if input_config["kind"] in {"token_ids", "dense_sequence"} else "N/A"
        image_size = (
            f"{shape_without_batch[-2]}x{shape_without_batch[-1]}"
            if input_config["kind"] == "image"
            else "N/A"
        )
        vocab = input_config.get("vocab_size", "N/A")
        lines.append(
            "| {display} | `{entry_id}` | `{key}` | {parameters} | {trainable} | {layers} | {hidden} | "
            "{heads} | {sequence} | {image_size} | {vocab} | `{shape}` | {microbatch} | `{input_dtype}` | "
            "`{compute_dtype}` | `{parameter_dtype}` | {optimizer} (lr={lr}) | {loss} ({reduction}) |".format(
                display=workload["display_name"],
                entry_id=entry["manifest_entry_id"],
                key=workload["registry_key"],
                parameters=entry["parameter_count"],
                trainable=entry["trainable_parameter_count"],
                layers=layers,
                hidden=hidden,
                heads=heads,
                sequence=sequence,
                image_size=image_size,
                vocab=vocab,
                shape="x".join(str(size) for size in batch_input["shape"]),
                microbatch=entry["microbatch_size"],
                input_dtype=batch_input["dtype"],
                compute_dtype=workload["compute_dtype"],
                parameter_dtype=workload["parameter_dtype"],
                optimizer=optimizer["name"],
                lr=optimizer["lr"],
                loss=loss["name"],
                reduction=loss["reduction"],
            )
        )
    environment = manifest.get("environment")
    lines.extend(("", "## Environment", "", "| Field | Value | Missing reason |", "|---|---|---|"))
    if environment is None:
        lines.append("| environment | `null` | environment snapshot missing |")
    else:
        environment_rows = (
            ("Git commit", environment["git_commit"], environment["git_commit_reason"]),
            ("Git dirty", environment["git_dirty"], environment["git_dirty_reason"]),
            ("Python", environment["python_version"], None),
            ("PyTorch", environment["torch_version"], None),
            ("CUDA", environment["cuda_version"], environment["cuda_version_reason"]),
            ("GPU devices", environment["gpu_devices"], environment["gpu_devices_reason"]),
            ("Driver", environment["driver_version"], environment["driver_version_reason"]),
            (
                "Compiler mode",
                environment["execution"]["compiler_mode"],
                environment["execution"]["compiler_mode_reason"],
            ),
            (
                "Execution parameters",
                environment["execution"]["parameters"],
                environment["execution"]["parameters_reason"],
            ),
        )
        for field, value, reason in environment_rows:
            rendered_value = "null" if value is None else canonical_json(value).replace("|", "\\|")
            rendered_reason = "" if reason is None else str(reason).replace("|", "\\|")
            lines.append(f"| {field} | `{rendered_value}` | {rendered_reason} |")
    return "\n".join(lines) + "\n"


def write_workload_manifest(
    manifest: Mapping[str, Any],
    json_path: str | Path,
    t1_path: str | Path,
) -> None:
    json_output = Path(json_path)
    t1_output = Path(t1_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    t1_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(canonical_json(manifest, indent=2) + "\n", encoding="utf-8")
    t1_output.write_text(render_t1_markdown(manifest), encoding="utf-8")
