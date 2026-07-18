import math

import pytest

from peakaware.cache.executable import load_executable_cache, select_cached_executable, store_executable_cache
from peakaware.cache.keys import build_compiled_artifact_key, build_plan_evaluation_key
from peakaware.cache.store import CacheEntry, invalidate_downstream, load_cache_entry, store_cache_entry
from peakaware.contracts import MeasuredExecutable
from peakaware.cost.base import OpSignature
from peakaware.cost.profile_db import ProfileDB, ProfileRecord
from peakaware.errors import PluginConflictError
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
