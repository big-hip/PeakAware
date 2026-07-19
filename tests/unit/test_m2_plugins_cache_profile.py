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
from peakaware.cost.base import OpCost, OpSignature, RooflineFallbackProvider, StaticCostProvider
from peakaware.cost.collector import (
    collect_microbenchmark,
    collect_model_trace,
    measure_cuda_events,
    summarize_samples,
)
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

    assert loaded is not None
    assert loaded.plan_id == "fast"
    assert selected is fast


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


def test_composite_cost_provider_uses_exact_then_interpolation_then_fallback(tmp_path):
    db = ProfileDB(tmp_path / "profiles.sqlite")
    exact = OpSignature("mm", "aten.mm", 128, 64, "float32")
    nearby = OpSignature("mm", "aten.mm", 256, 128, "float32")
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
    assert legacy.hardware_version == static.hardware_version
    assert legacy.software_version == static.software_version
    assert roofline.hardware_version != "unknown"
    assert roofline.software_version.startswith("torch:")


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


def test_default_builtin_registry_exposes_core_m2_services(tmp_path):
    snapshot = build_default_registry(profile_db_path=tmp_path / "profiles.sqlite")

    assert snapshot.resolve(ServiceKind.CAPTURE_BACKEND, "aot_or_fx") is not None
    assert snapshot.resolve(ServiceKind.PLAN_DIAGNOSTIC, "diagnose_plan") is not None
    assert snapshot.resolve(ServiceKind.CANDIDATE_POLICY, "min_cut_seed") == "torch_default_partition_seed"
    correction = snapshot.resolve(ServiceKind.RUNTIME_VALIDATOR, "inductor_correction")({})
    cost_names = tuple(record.name for record in snapshot.services_for(ServiceKind.COST_PROVIDER))

    assert correction.status == "unavailable"
    assert "Top-K measurement" in correction.reason
    assert cost_names[:3] == ("profile_db_exact", "profile_db_interpolated", "static_fallback")
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
