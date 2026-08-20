import math

import pytest
import torch

from peakaware.cache.executable import (
    load_executable_cache,
    load_executable_measurement_cache,
    select_cached_executable,
    store_executable_cache,
    store_executable_measurement_cache,
)
from peakaware.cache.keys import build_compiled_artifact_key, build_plan_evaluation_key
from peakaware.cache.store import CacheEntry, invalidate_downstream, load_cache_entry, store_cache_entry
from peakaware.contracts import MeasuredExecutable
from peakaware.cost.attention import AttentionHardwareSpec, ScaledDotProductAttentionCostProvider
from peakaware.cost.base import (
    MetadataViewCostProvider,
    OpCost,
    OpSignature,
    RooflineFallbackProvider,
    StaticCostProvider,
    StructuralZeroCostProvider,
)
from peakaware.cost.collector import (
    collect_microbenchmark,
    collect_model_trace,
    measure_cuda_events,
    summarize_samples,
)
from peakaware.cost.calibration import build_residual_calibration_report
from peakaware.cost.composite import CompositeCostProvider, build_composite_provider
from peakaware.cost.legacy_adapter import LegacyCostmodelAdapter
from peakaware.cost.profile_db import ExactProfileProvider, InterpolatedProfileProvider, ProfileDB, ProfileRecord
from peakaware.errors import PatchRestoreError, PluginConflictError
from peakaware.plugins import ServiceKind, build_default_registry
from peakaware.plugins.patching import PatchSession, PatchSpec
from peakaware.plugins.registry import PluginRegistry


def test_registry_resolves_highest_priority_and_freezes():
    registry = PluginRegistry()
    registry.register_service("cost_provider", "static_low", object(), priority=0)
    high = object()
    registry.register_service("cost_provider", "static_high", high, priority=10)
    registry.register_hook("after_search", lambda x: x, priority=1)

    snapshot = registry.freeze()

    assert snapshot.resolve("cost_provider") is high
    assert snapshot.services_for("cost_provider")[0].service is high
    assert len(snapshot.hooks_for("after_search")) == 1
    with pytest.raises(PluginConflictError):
        registry.register_service("cost_provider", "late", object())


def test_registry_rejects_duplicate_services_but_allows_patch_chains():
    registry = PluginRegistry()
    registry.register_service("cost_provider", "same", object())
    with pytest.raises(PluginConflictError):
        registry.register_service("cost_provider", "same", object())

    def wrapper(next_fn, *args, **kwargs):
        return next_fn(*args, **kwargs)

    registry.register_patch(PatchSpec("a", "math", "sqrt", wrapper))
    registry.register_patch(PatchSpec("b", "math", "sqrt", wrapper))
    with pytest.raises(PluginConflictError):
        registry.register_patch(PatchSpec("a", "math", "sqrt", wrapper))


def test_patch_session_restores_identity_and_composes_wrappers():
    original = math.sqrt
    calls = []

    def first(next_fn, *args, **kwargs):
        calls.append("first")
        return next_fn(*args, **kwargs) + 1

    def second(next_fn, *args, **kwargs):
        calls.append("second")
        return next_fn(*args, **kwargs) * 2

    specs = (
        PatchSpec("first", "math", "sqrt", first, priority=10),
        PatchSpec("second", "math", "sqrt", second, priority=0),
    )

    with PatchSession(specs):
        assert math.sqrt(9) == 7
        assert calls == ["first", "second"]

    assert math.sqrt is original


def test_patch_session_rolls_back_when_enter_fails_midway():
    original_sqrt = math.sqrt

    def wrapper(next_fn, *args, **kwargs):
        return next_fn(*args, **kwargs) + 1

    specs = (
        PatchSpec("valid", "math", "sqrt", wrapper),
        PatchSpec("missing", "math", "definitely_missing_peakaware_attr", wrapper),
    )

    with pytest.raises(AttributeError):
        with PatchSession(specs):
            pass

    assert math.sqrt is original_sqrt
    assert math.sqrt(9) == 3


@pytest.mark.parametrize(
    ("target_attribute", "expected"),
    (
        ("add", 2.0),
        ("mul", 1.0),
    ),
)
def test_patch_session_restores_real_torch_function_after_wrapper_exception(target_attribute, expected):
    original = getattr(torch, target_attribute)

    def raising_wrapper(next_fn, *args, **kwargs):
        del next_fn, args, kwargs
        raise RuntimeError("injected torch patch failure")

    spec = PatchSpec(f"faulty_torch_{target_attribute}", "torch", target_attribute, raising_wrapper)

    with pytest.raises(RuntimeError, match="injected torch patch failure"):
        with PatchSession((spec,)):
            getattr(torch, target_attribute)(torch.ones(1), torch.ones(1))

    restored = getattr(torch, target_attribute)
    assert restored is original
    assert torch.equal(restored(torch.ones(1), torch.ones(1)), torch.full((1,), expected))


def test_patch_session_validates_every_wrapper_signature_before_patching():
    original_sqrt = math.sqrt

    def wrapper(next_fn, *args, **kwargs):
        return next_fn(*args, **kwargs)

    specs = (
        PatchSpec("compatible", "math", "sqrt", wrapper, expected_signature="(x, /)"),
        PatchSpec("incompatible", "math", "sqrt", wrapper, expected_signature="(value)"),
    )

    with pytest.raises(PatchRestoreError, match="signature mismatch"):
        with PatchSession(specs):
            pass

    assert math.sqrt is original_sqrt


def test_cache_keys_are_stable_and_separate_layers():
    plan_key = build_plan_evaluation_key(
        analysis_key="a",
        optimizer_mode="adamw",
        cost_database_version="v1",
        search_policy_version="m1",
        budget_bucket=123,
    )
    same_plan_key = build_plan_evaluation_key(
        analysis_key="a",
        optimizer_mode="adamw",
        cost_database_version="v1",
        search_policy_version="m1",
        budget_bucket=123,
    )
    artifact_key = build_compiled_artifact_key(
        lowered_plan_fingerprint="p",
        state_signature="s",
        input_guards=(("shape", "(2, 3)"),),
        device_capability="cpu",
        torch_version="2.13",
        compiler_version="none",
        partition_plugin_version="core",
    )

    assert plan_key == same_plan_key
    assert plan_key != artifact_key


def test_cache_entry_round_trip_validates_provenance(tmp_path):
    entry = CacheEntry(key="k1", artifact={"value": 3}, provenance={"torch": "2.13", "schema": 1})

    store_cache_entry(tmp_path, "analysis", entry)

    assert load_cache_entry(tmp_path, "analysis", "k1", {"torch": "2.13"}).artifact == {"value": 3}
    assert load_cache_entry(tmp_path, "analysis", "k1", {"torch": "2.12"}) is None


def test_cache_entry_store_failure_leaves_no_partial_files(tmp_path):
    entry = CacheEntry(key="bad", artifact=lambda value: value, provenance={"torch": "2.13"})

    with pytest.raises(Exception):
        store_cache_entry(tmp_path, "analysis", entry)

    layer_dir = tmp_path / "analysis"
    assert not (layer_dir / "bad.pkl").exists()
    assert not (layer_dir / "bad.json").exists()
    assert not list(layer_dir.glob("*.tmp"))


def test_cache_invalidate_downstream_removes_lower_layers(tmp_path):
    store_cache_entry(tmp_path, "capture", CacheEntry("c", "capture", {}))
    store_cache_entry(tmp_path, "analysis", CacheEntry("a", "analysis", {}))
    store_cache_entry(tmp_path, "executable", CacheEntry("e", "executable", {}))

    removed = invalidate_downstream(tmp_path, "capture")

    assert len(removed) == 4
    assert (tmp_path / "capture" / "c.pkl").exists()
    assert not (tmp_path / "analysis" / "a.pkl").exists()
    assert not (tmp_path / "executable" / "e.pkl").exists()


def test_executable_cache_round_trip_and_selection(tmp_path):
    slow = MeasuredExecutable("slow", abs, 100, 20.0, True)
    fast = MeasuredExecutable("fast", abs, 120, 10.0, True)
    too_large = MeasuredExecutable("large", abs, 1000, 1.0, True)

    store_executable_cache(tmp_path, "fast-key", fast, {"compiler": "none"})
    loaded = load_executable_cache(tmp_path, "fast-key", {"compiler": "none"})
    selected = select_cached_executable((slow, fast, too_large), memory_budget_bytes=200)
    time_selected = select_cached_executable(
        (slow, fast, too_large),
        memory_budget_bytes=200,
        selection_objective="min_time_then_peak",
    )

    assert loaded is not None
    assert loaded.plan_id == "fast"
    assert selected is slow
    assert time_selected is fast


def test_executable_measurement_cache_rebinds_callable(tmp_path):
    executable = MeasuredExecutable("plan", abs, 10, 2.0, True, {"step_us": 2.0})

    store_executable_measurement_cache(tmp_path, "plan-key", executable, {"compiler": "none"})
    loaded = load_executable_measurement_cache(tmp_path, "plan-key", round, {"compiler": "none"})

    assert loaded is not None
    assert loaded.forward_backward is round
    assert loaded.measured_peak_bytes == 10


def test_profile_db_exact_lookup_round_trip(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    signature = OpSignature("add", "aten.add", 12, 12, "float32")
    record = ProfileRecord(
        signature_hash="ignored",
        sample_count=12,
        p50_us=3.5,
        p90_us=5.0,
        mean_us=4.0,
        workspace_bytes=64,
    )

    assert db.lookup_exact(signature) is None
    db.upsert_profile(signature, record)
    cost = db.lookup_exact(signature)

    assert cost is not None
    assert cost.estimated_us == 3.5
    assert cost.memory_bytes == 64
    assert cost.confidence == 1.0
    assert cost.hardware_version != "unknown"
    assert cost.software_version.startswith("torch:")


def test_profile_db_signature_reuses_profiles_across_fx_node_names(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    profiled = OpSignature(
        "mm_17",
        "aten.mm.default",
        96,
        48,
        "torch.float32",
        input_shapes=((4, 3), (3, 4)),
        output_shapes=((4, 4),),
        input_dtypes=("torch.float32", "torch.float32"),
        output_dtypes=("torch.float32",),
    )
    repeated_node = OpSignature(
        "mm_203",
        "aten.mm.default",
        96,
        48,
        "torch.float32",
        input_shapes=profiled.input_shapes,
        output_shapes=profiled.output_shapes,
        input_dtypes=profiled.input_dtypes,
        output_dtypes=profiled.output_dtypes,
    )
    db.upsert_profile(profiled, ProfileRecord("ignored", 20, 7.0, 8.0, 7.5, 0))

    cost = db.lookup_exact(repeated_node)

    assert cost is not None
    assert cost.estimated_us == 7.0


def test_profile_db_exact_signature_distinguishes_shapes_and_dtypes(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    source = OpSignature(
        "mm",
        "aten.mm.default",
        96,
        48,
        "torch.float32",
        input_shapes=((4, 3), (3, 4)),
        output_shapes=((4, 4),),
        input_dtypes=("torch.float32", "torch.float32"),
        output_dtypes=("torch.float32",),
    )
    different_shape = OpSignature(
        "mm_1",
        "aten.mm.default",
        192,
        96,
        "torch.float32",
        input_shapes=((8, 3), (3, 4)),
        output_shapes=((8, 4),),
        input_dtypes=source.input_dtypes,
        output_dtypes=source.output_dtypes,
    )
    different_dtype = OpSignature(
        "mm_2",
        "aten.mm.default",
        48,
        24,
        "torch.float16",
        input_shapes=source.input_shapes,
        output_shapes=source.output_shapes,
        input_dtypes=("torch.float16", "torch.float16"),
        output_dtypes=("torch.float16",),
    )
    db.upsert_profile(source, ProfileRecord("ignored", 20, 7.0, 8.0, 7.5, 0))

    assert db.lookup_exact(different_shape) is None
    assert db.lookup_exact(different_dtype) is None


def test_profile_db_isolates_rows_by_hardware_and_software_environment(tmp_path):
    path = tmp_path / "profiles.sqlite"
    signature = OpSignature("add", "aten.add.Tensor", 16, 8, "torch.float32")
    record = ProfileRecord("ignored", 20, 3.0, 4.0, 3.5, 0)
    first = ProfileDB(path, hardware_version="gpu:a", software_version="torch:a")
    first.upsert_profile(signature, record)

    other_hardware = ProfileDB(path, hardware_version="gpu:b", software_version="torch:a")
    other_software = ProfileDB(path, hardware_version="gpu:a", software_version="torch:b")

    assert first.lookup_exact(signature) is not None
    assert other_hardware.lookup_exact(signature) is None
    assert other_software.lookup_exact(signature) is None
    other_hardware.upsert_profile(signature, ProfileRecord("ignored", 20, 9.0, 10.0, 9.5, 0))
    other_hardware.invalidate_by_environment()
    assert first.lookup_exact(signature) is not None
    assert other_hardware.lookup_exact(signature) is None


def test_profile_db_migrates_legacy_rows_as_environment_unscoped(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE op_profile (
                signature_hash TEXT PRIMARY KEY,
                target TEXT NOT NULL DEFAULT '',
                dtype TEXT NOT NULL DEFAULT '',
                total_bytes INTEGER NOT NULL DEFAULT 0,
                sample_count INTEGER NOT NULL,
                p50_us REAL NOT NULL,
                p90_us REAL NOT NULL,
                mean_us REAL NOT NULL,
                workspace_bytes INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO op_profile VALUES ('old', 'aten.add.Tensor', 'float32', 16, 20, 1, 2, 1.5, 0)"
        )

    db = ProfileDB(path, hardware_version="gpu:a", software_version="torch:a")

    assert db.lookup_exact(OpSignature("add", "aten.add.Tensor", 8, 8, "float32")) is None
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(op_profile)")}
        environments = conn.execute(
            "SELECT hardware_version, software_version FROM op_profile"
        ).fetchall()
    assert {"hardware_version", "software_version", "input_shapes_json"} <= columns
    assert environments == [("legacy:unscoped", "legacy:unscoped")]


def test_profile_collector_summarizes_samples_and_writes_db(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    signature = OpSignature("mul", "aten.mul", 8, 8, "float32")

    record = summarize_samples((3.0, 1.0, 2.0, 4.0), workspace_bytes=16)
    result = collect_microbenchmark(
        signature,
        lambda x: x * 2,
        (torch.ones(2),),
        warmup=0,
        repeats=2,
        db=db,
    )
    cost = db.lookup_exact(signature)

    assert record.sample_count == 4
    assert record.p50_us == 2.5
    assert record.p90_us == 4.0
    assert result.record.sample_count == 2
    assert result.record.source == "microbenchmark"
    assert cost is not None
    assert cost.estimated_us == result.record.p50_us


def test_cuda_event_measurement_falls_back_to_wall_time_on_cpu():
    elapsed_us, workspace = measure_cuda_events(lambda x: x.relu(), (torch.tensor([-1.0, 1.0]),))

    assert elapsed_us >= 0.0
    assert workspace >= 0


def test_collect_model_trace_reports_profiler_events():
    events = collect_model_trace(lambda x: (x + 1).relu(), (torch.ones(2),))

    assert events
    assert events == tuple(sorted(events, key=lambda event: (-event.cpu_time_total_us, event.name)))
    assert any("relu" in event.name.lower() or "add" in event.name.lower() for event in events)


def test_profile_db_nearest_interpolates_same_target_and_dtype(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    source = OpSignature("add", "aten.add", 100, 100, "float32")
    target = OpSignature("add", "aten.add", 200, 200, "float32")
    other_dtype = OpSignature("add", "aten.add", 200, 200, "float16")
    db.upsert_profile(source, ProfileRecord("ignored", 12, 10.0, 12.0, 11.0, 20))

    interpolated = db.lookup_nearest(target)

    assert interpolated is not None
    assert interpolated.source == "profile_db_interpolated"
    assert interpolated.estimated_us == 20.0
    assert interpolated.hardware_version != "unknown"
    assert interpolated.software_version.startswith("torch:")
    assert db.lookup_nearest(other_dtype) is None


def test_profile_db_matmul_interpolation_uses_flops_not_total_bytes(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    source = OpSignature(
        "mm",
        "aten.mm.default",
        96,
        64,
        "torch.float32",
        input_shapes=((4, 3), (3, 4)),
        output_shapes=((4, 4),),
    )
    target = OpSignature(
        "mm_1",
        "aten.mm.default",
        160,
        128,
        "torch.float32",
        input_shapes=((8, 3), (3, 4)),
        output_shapes=((8, 4),),
    )
    db.upsert_profile(source, ProfileRecord("ignored", 20, 10.0, 12.0, 11.0, 0))

    interpolated = db.lookup_nearest(target)

    assert interpolated is not None
    assert interpolated.estimated_us == 20.0


def test_profile_db_refuses_unsafe_generic_convolution_interpolation(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    source = OpSignature(
        "convolution",
        "aten.convolution.default",
        1024,
        2048,
        "torch.float32",
        input_shapes=((1, 16, 8, 8), (32, 16, 3, 3)),
        output_shapes=((1, 32, 6, 6),),
    )
    target = OpSignature(
        "convolution_1",
        "aten.convolution.default",
        4096,
        8192,
        "torch.float32",
        input_shapes=((1, 16, 16, 16), (32, 16, 3, 3)),
        output_shapes=((1, 32, 14, 14),),
    )
    db.upsert_profile(source, ProfileRecord("ignored", 20, 10.0, 12.0, 11.0, 0))

    assert db.lookup_nearest(target) is None


def test_composite_cost_provider_uses_exact_then_interpolation_then_fallback(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    exact = OpSignature(
        "mm",
        "aten.mm",
        128,
        64,
        "float32",
        input_shapes=((4, 4), (4, 4)),
        output_shapes=((4, 4),),
    )
    nearby = OpSignature(
        "mm_1",
        "aten.mm",
        256,
        128,
        "float32",
        input_shapes=((8, 4), (4, 4)),
        output_shapes=((8, 4),),
    )
    missing = OpSignature("relu", "aten.relu", 16, 16, "float32")
    db.upsert_profile(exact, ProfileRecord("ignored", 20, 4.0, 5.0, 4.5, 32))
    provider = CompositeCostProvider(
        (
            ExactProfileProvider(db),
            InterpolatedProfileProvider(db),
            RooflineFallbackProvider(bandwidth_bytes_per_us=16, launch_overhead_us=1.0),
        )
    )

    exact_cost = provider.estimate_with_provenance(exact)
    nearby_cost = provider.estimate_with_provenance(nearby)
    missing_cost = provider.estimate_with_provenance(missing)

    assert exact_cost is not None and exact_cost.cost.source == "profile_db_exact"
    assert nearby_cost is not None and nearby_cost.cost.source == "profile_db_interpolated"
    assert missing_cost is not None and missing_cost.cost.source == "roofline_fallback"
    assert missing_cost.cost.estimated_us > 1.0
    assert exact_cost.cost.hardware_version != "unknown"
    assert nearby_cost.cost.software_version.startswith("torch:")
    assert missing_cost.cost.hardware_version != "unknown"


def test_builtin_cost_providers_attach_hardware_and_software_provenance():
    signature = OpSignature("add", "aten.add", 8, 8, "float32")
    static = StaticCostProvider().estimate(signature)
    legacy = LegacyCostmodelAdapter().estimate(signature)
    roofline = RooflineFallbackProvider().estimate(signature)

    assert static.hardware_version == "generic"
    assert static.software_version.startswith("torch:")
    assert legacy is not None
    assert legacy.hardware_version in {
        static.hardware_version,
        "zhanlu:A3,A3",
        "zhanlu:RTX_A6000,RTX_A6000",
    }
    assert legacy.software_version == static.software_version
    assert roofline.hardware_version != "unknown"
    assert roofline.software_version.startswith("torch:")


def _test_attention_hardware():
    return AttentionHardwareSpec(
        hardware_id="RTX_A6000",
        compute_tflops={"float16": 154.8, "bfloat16": 154.8, "float32": 38.7},
        vector_tflops={"float16": 19.3, "bfloat16": 19.3, "float32": 19.3},
        hbm_tbps=0.768,
        fused_launch_us=1.0,
    )


def test_sdpa_provider_models_fused_forward_and_backward_without_global_workspace():
    provider = ScaledDotProductAttentionCostProvider(_test_attention_hardware())
    qkv = ((1, 12, 128, 64),) * 3
    forward = OpSignature(
        "_scaled_dot_product_efficient_attention",
        "aten._scaled_dot_product_efficient_attention.default",
        3 * 1 * 12 * 128 * 64 * 4,
        0,
        "torch.float32",
        input_shapes=qkv,
        input_dtypes=("torch.float32",) * 3,
    )
    backward = OpSignature(
        "_scaled_dot_product_efficient_attention_backward",
        "aten._scaled_dot_product_efficient_attention_backward.default",
        6 * 1 * 12 * 128 * 64 * 4,
        0,
        "torch.float32",
        input_shapes=((1, 12, 128, 64),) * 5 + ((1, 12, 128), (), ()),
        input_dtypes=("torch.float32",) * 8,
    )

    forward_cost = provider.estimate(forward)
    backward_cost = provider.estimate(backward)

    assert forward_cost is not None
    assert backward_cost is not None
    assert forward_cost.estimated_us > 1.0
    assert backward_cost.estimated_us > forward_cost.estimated_us
    assert forward_cost.memory_bytes == 0
    assert backward_cost.memory_bytes == 0
    assert forward_cost.source == "analytical:sdpa_fused"
    assert forward_cost.hardware_version == "zhanlu:RTX_A6000"


def test_sdpa_provider_rejects_unfused_attention_and_invalid_shapes():
    provider = ScaledDotProductAttentionCostProvider(_test_attention_hardware())

    assert provider.estimate(OpSignature("bmm", "aten.bmm.default", 1, 1)) is None
    assert provider.estimate(
        OpSignature(
            "sdpa",
            "aten._scaled_dot_product_efficient_attention.default",
            1,
            1,
            input_shapes=((1, 12, 128),) * 3,
        )
    ) is None


def test_legacy_costmodel_adapter_uses_analytical_model_when_supported():
    signature = OpSignature("add", "aten.add", 1024, 1024, "float32")
    cost = LegacyCostmodelAdapter().estimate(signature)

    assert cost is not None
    assert cost.estimated_us > 0
    assert cost.source == "legacy_adapter:static_fallback" or cost.source.startswith(
        "legacy_adapter:zhanlu_analytical"
    )


def test_residual_calibration_report_improves_biased_predictions():
    records = [
        {
            "status": "ok",
            "task_name": "task_a",
            "measured_plan_results": [
                {"estimated_peak_bytes": 100, "measured_peak_bytes": 150},
                {"estimated_peak_bytes": 200, "measured_peak_bytes": 250},
            ],
        },
        {
            "status": "ok",
            "task_name": "task_b",
            "measured_plan_results": [
                {"estimated_peak_bytes": 1000, "measured_peak_bytes": 900},
                {"estimated_peak_bytes": 1100, "measured_peak_bytes": 1000},
            ],
        },
    ]

    report = build_residual_calibration_report(records)

    assert report["rule_count"] == 2
    assert report["evaluation"]["mean_calibrated_ape"] < report["evaluation"]["mean_raw_ape"]
    assert report["evaluation"]["covered_row_count"] == 4


def test_residual_calibration_supports_holdout_split():
    records = [
        {
            "status": "ok",
            "task_name": "task_a",
            "matrix_pass_index": 0,
            "measured_plan_results": [{"estimated_peak_bytes": 100, "measured_peak_bytes": 150}],
        },
        {
            "status": "ok",
            "task_name": "task_a",
            "matrix_pass_index": 1,
            "measured_plan_results": [{"estimated_peak_bytes": 200, "measured_peak_bytes": 250}],
        },
    ]

    report = build_residual_calibration_report(records, holdout_field="matrix_pass_index", holdout_value="1")

    assert report["split"]["train_record_count"] == 1
    assert report["split"]["eval_record_count"] == 1
    assert report["evaluation"]["mean_calibrated_ape"] == 0.0


def test_build_composite_provider_adds_roofline_tail():
    class EmptyProvider:
        source = "empty"

        def supports(self, signature):
            return True

        def estimate(self, signature):
            return None

    provider = build_composite_provider((EmptyProvider(),))
    cost = provider.estimate(OpSignature("x", "aten.x", 1, 1))

    assert isinstance(cost, OpCost)
    assert cost.source == "roofline_fallback"


@pytest.mark.parametrize(
    ("op_name", "target"),
    (
        ("primals_1", "primals_1"),
        ("tangents_2", "tangents_2"),
        ("output", "output"),
        ("arg0", "arg0"),
        ("arg_1", "arg_1"),
        ("placeholder_3", "placeholder_3"),
        ("lifted_tensor_0", "lifted_tensor_0"),
    ),
)
def test_structural_zero_cost_provider_models_graph_interface_nodes(op_name, target):
    provider = StructuralZeroCostProvider()
    signature = OpSignature(op_name, target, 1024, 2048, "float32")

    cost = provider.estimate(signature)

    assert cost is not None
    assert cost.estimated_us == 0.0
    assert cost.memory_bytes == 0
    assert cost.source == "structural_zero"
    assert cost.confidence == 1.0


@pytest.mark.parametrize(
    ("op_name", "target"),
    (
        ("argmax", "aten.argmax.default"),
        ("argmin_1", "aten.argmin.default"),
        ("argument", "custom.argument"),
    ),
)
def test_structural_zero_cost_provider_does_not_hide_real_arg_ops(op_name, target):
    provider = StructuralZeroCostProvider()
    signature = OpSignature(op_name, target, 1024, 2048, "float32")

    assert not provider.supports(signature)
    assert provider.estimate(signature) is None


@pytest.mark.parametrize(
    "target",
    (
        "<built-in function getitem>",
        "aten._unsafe_view.default",
        "aten.detach.default",
        "aten.expand.default",
        "aten.permute.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.dim",
        "aten.t.default",
        "aten.transpose.int",
        "aten.unsqueeze.default",
        "aten.view.default",
    ),
)
def test_metadata_view_provider_models_alias_only_cuda_nodes_as_zero(target):
    provider = MetadataViewCostProvider()
    cost = provider.estimate(OpSignature("node", target, 4096, 4096, "torch.float32"))

    assert cost is not None
    assert cost.estimated_us == 0.0
    assert cost.memory_bytes == 0
    assert cost.source == "metadata_view_zero"
    assert cost.confidence == 1.0


@pytest.mark.parametrize(
    "target",
    (
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.reshape.default",
        "aten.select_backward.default",
        "aten.slice_backward.default",
    ),
)
def test_metadata_view_provider_does_not_hide_copy_or_backward_kernels(target):
    provider = MetadataViewCostProvider()
    signature = OpSignature("node", target, 4096, 4096, "torch.float32")

    assert not provider.supports(signature)
    assert provider.estimate(signature) is None


def test_default_builtin_registry_exposes_core_m2_services(tmp_path):
    snapshot = build_default_registry(profile_db_path=tmp_path / "profiles.sqlite")

    assert snapshot.resolve(ServiceKind.CAPTURE_BACKEND, "aot_or_fx") is not None
    assert snapshot.resolve(ServiceKind.PLAN_DIAGNOSTIC, "diagnose_plan") is not None
    assert snapshot.resolve(ServiceKind.CANDIDATE_POLICY, "min_cut_seed") == "torch_default_partition_seed"
    correction = snapshot.resolve(ServiceKind.RUNTIME_VALIDATOR, "inductor_correction")({})
    cost_names = tuple(record.name for record in snapshot.services_for(ServiceKind.COST_PROVIDER))

    assert correction.status == "unavailable"
    assert "Top-K measurement" in correction.reason
    assert cost_names[:6] == (
        "structural_zero",
        "metadata_view_zero",
        "profile_db_exact",
        "profile_db_interpolated",
        "sdpa_fused_analytical",
        "legacy_costmodel",
    )
    assert snapshot.resolve("profile_db", "default") is not None


def test_default_registry_cost_providers_feed_composite_provider(tmp_path):
    profile_path = tmp_path / "profiles.sqlite"
    snapshot = build_default_registry(profile_db_path=profile_path)
    db = snapshot.resolve("profile_db", "default")
    signature = OpSignature("mm", "aten.mm", 8, 8, "float32")
    db.upsert_profile(signature, ProfileRecord("ignored", 10, 2.0, 3.0, 2.5, 16))
    provider = build_composite_provider(tuple(record.service for record in snapshot.services_for(ServiceKind.COST_PROVIDER)))

    cost = provider.estimate(signature)

    assert cost is not None
    assert cost.source == "profile_db_exact"
    assert cost.estimated_us == 2.0


def test_default_registry_composite_prefers_structural_zero_for_interface_nodes(tmp_path):
    snapshot = build_default_registry(profile_db_path=tmp_path / "profiles.sqlite")
    provider = build_composite_provider(
        tuple(record.service for record in snapshot.services_for(ServiceKind.COST_PROVIDER))
    )

    for name in ("primals_1", "tangents_1", "output"):
        cost = provider.estimate(OpSignature(name, name, 4096, 4096, "float32"))
        assert cost is not None
        assert cost.source == "structural_zero"
        assert cost.estimated_us == 0.0
        assert cost.memory_bytes == 0
