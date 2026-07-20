from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from peakaware.publication.budget_calibration import (
    AllSaveReference,
    CalibrationManifest,
    LegacyBudgetConversionError,
    MinCutPolicy,
    PolicyMeasurement,
    compile_representatives,
    convert_physical_budget_to_activation_memory_budget,
    convert_physical_budget_to_ratio,
    derive_physical_budgets,
    physical_budget_to_ratio,
    select_policy_for_budget,
)


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


IDENTITY = {"source_key": _fp("source"), "backend": "inductor", "environment": _fp("env")}


def _reference() -> AllSaveReference:
    return AllSaveReference(
        **IDENTITY,
        replicate_seeds=(1, 2, 3, 4, 5),
        replicate_process_ids=(101, 102, 103, 104, 105),
        per_replicate_peak_bytes=((98, 100), (120, 110), (101, 99), (95, 90), (105, 103)),
    )


def _policy(
    ratio: float,
    *,
    fw: str = "fw",
    bw: str = "bw",
    residual: str = "residual",
) -> MinCutPolicy:
    return MinCutPolicy.create(
        ratio=ratio,
        fw_graph_identity=fw,
        bw_graph_identity=bw,
        residual_identity=residual,
        **IDENTITY,
    )


def _tuning(
    policy: MinCutPolicy,
    seed: int,
    peaks: tuple[int, ...],
    median_time: float,
    *,
    process_id: int | None = None,
    status: str = "ok",
) -> PolicyMeasurement:
    return PolicyMeasurement(
        policy_id=policy.policy_id,
        cohort="tuning",
        seed=seed,
        process_id=process_id or 1000 + seed,
        budget_ratio=None,
        budget_bytes=None,
        per_run_peak_bytes=peaks,
        median_event_step_seconds=None if status == "oom" else median_time,
        status=status,
        **IDENTITY,
    )


def _evaluation(
    policy: MinCutPolicy,
    seed: int,
    ratio: float,
    budget: int,
    peak: int,
    *,
    process_id: int | None = None,
) -> PolicyMeasurement:
    return PolicyMeasurement(
        policy_id=policy.policy_id,
        cohort="evaluation",
        seed=seed,
        process_id=process_id or 2000 + seed,
        budget_ratio=ratio,
        budget_bytes=budget,
        per_run_peak_bytes=(peak,),
        median_event_step_seconds=1.0,
        status="ok" if peak <= budget else "budget_violation",
        **IDENTITY,
    )


def test_all_save_repeat_max_median_floor_and_immutability():
    reference = _reference()
    assert reference.replicate_max_peak_bytes == (100, 120, 101, 95, 105)
    assert reference.P_ref == 101
    assert derive_physical_budgets(reference, (1.0, 0.9, 0.0)) == (101, 90, 0)
    assert reference.to_canonical_json() == reference.to_canonical_json()
    with pytest.raises(FrozenInstanceError):
        reference.backend = "aot_eager"  # type: ignore[misc]


def test_all_save_requires_five_unique_processes_and_seeds():
    with pytest.raises(ValueError, match="at least 5"):
        replace(
            _reference(),
            replicate_seeds=(1, 2, 3, 4),
            replicate_process_ids=(101, 102, 103, 104),
            per_replicate_peak_bytes=((1,),) * 4,
        )
    with pytest.raises(ValueError, match="seeds must be unique"):
        replace(_reference(), replicate_seeds=(1, 2, 3, 4, 4))
    with pytest.raises(ValueError, match="process IDs must be unique"):
        replace(_reference(), replicate_process_ids=(101, 102, 103, 104, 104))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_key", "not-a-fingerprint", "64-character hex"),
        ("environment", "ABCDEF" * 10 + "ABCD", "64-character hex"),
        ("backend", "eager", "aot_eager"),
    ),
)
def test_identity_protocol_is_strict(field: str, value: str, message: str):
    with pytest.raises(ValueError, match=message):
        replace(_reference(), **{field: value})


def test_policy_id_is_canonical_and_compile_alias_requires_all_identities():
    alias_a = _policy(0.2)
    alias_b = _policy(0.8)
    different_fw = _policy(0.3, fw="other-fw")
    different_bw = _policy(0.4, bw="other-bw")
    different_residual = _policy(0.5, residual="other-residual")
    different_source = MinCutPolicy.create(
        ratio=0.6,
        fw_graph_identity="fw",
        bw_graph_identity="bw",
        residual_identity="residual",
        source_key=_fp("other-source"),
        backend="inductor",
        environment=IDENTITY["environment"],
    )
    different_backend = MinCutPolicy.create(
        ratio=0.7,
        fw_graph_identity="fw",
        bw_graph_identity="bw",
        residual_identity="residual",
        source_key=IDENTITY["source_key"],
        backend="aot_eager",
        environment=IDENTITY["environment"],
    )
    different_environment = MinCutPolicy.create(
        ratio=0.9,
        fw_graph_identity="fw",
        bw_graph_identity="bw",
        residual_identity="residual",
        source_key=IDENTITY["source_key"],
        backend="inductor",
        environment=_fp("other-environment"),
    )

    representatives = compile_representatives(
        (
            alias_b,
            different_residual,
            different_bw,
            different_source,
            different_backend,
            different_environment,
            alias_a,
            different_fw,
        )
    )
    assert len(representatives) == 7
    assert {alias_a.policy_id, alias_b.policy_id} & {
        policy.policy_id for policy in representatives
    }
    with pytest.raises(ValueError, match="canonical policy content hash"):
        replace(alias_a, policy_id=_fp("invented"))
    with pytest.raises(ValueError, match="canonical policy content hash"):
        replace(alias_a, bw_graph_identity="tampered")


def test_discrete_selection_non_monotonic_peak_and_stable_ties():
    alpha = _policy(0.2, fw="alpha")
    beta = _policy(0.5, fw="beta")
    gamma = _policy(0.8, fw="gamma")
    tuning = (
        _tuning(alpha, 10, (70, 72), 3.0),
        _tuning(alpha, 11, (71, 73), 1.0),
        _tuning(alpha, 12, (72, 74), 2.0),
        _tuning(beta, 10, (91, 95), 0.5, process_id=2010),
        _tuning(beta, 11, (89, 94), 0.5, process_id=2011),
        _tuning(gamma, 10, (60, 61), 2.0, process_id=3010),
        _tuning(gamma, 11, (62, 63), 2.0, process_id=3011),
    )
    selected = select_policy_for_budget(
        ratio=0.8, budget_bytes=80, policies=(alpha, beta, gamma), measurements=tuning
    )
    assert selected.status == "selected"
    assert selected.policy_id == gamma.policy_id
    assert selected.tuning_max_peak_bytes == 63


def test_selection_final_tie_uses_canonical_policy_id():
    first = _policy(0.2, fw="first")
    second = _policy(0.8, fw="second")
    tuning = (
        _tuning(first, 10, (60,), 1.0),
        _tuning(second, 10, (60,), 1.0, process_id=3010),
    )
    selection = select_policy_for_budget(
        ratio=0.8, budget_bytes=80, policies=(first, second), measurements=tuning
    )
    assert selection.policy_id == min(first.policy_id, second.policy_id)


def test_all_oom_or_over_budget_freezes_infeasible_with_tuning_evidence():
    oom = _policy(0.2, fw="oom")
    large = _policy(0.8, fw="large")
    tuning = (
        _tuning(oom, 10, (), 0.0, status="oom"),
        _tuning(large, 11, (101,), 1.0),
    )
    selection = select_policy_for_budget(
        ratio=0.8, budget_bytes=80, policies=(oom, large), measurements=tuning
    )
    assert selection.status == "infeasible"
    assert selection.policy_id is None
    assert len(selection.tuning_evidence_sha256) == 64


def test_tuning_process_count_and_duplicate_slot_identity_are_rejected():
    policy = _policy(0.5)
    four = tuple(_tuning(policy, seed, (50,), 1.0) for seed in (10, 11, 12, 13))
    with pytest.raises(ValueError, match=r"1\.\.3"):
        select_policy_for_budget(ratio=0.5, budget_bytes=80, policies=(policy,), measurements=four)
    duplicate_seed = (_tuning(policy, 10, (50,), 1.0), _tuning(policy, 10, (51,), 1.1))
    with pytest.raises(ValueError, match="duplicate seed"):
        select_policy_for_budget(
            ratio=0.5, budget_bytes=80, policies=(policy,), measurements=duplicate_seed
        )
    paired = _policy(0.8, fw="paired")
    duplicate_process = (
        _tuning(policy, 10, (50,), 1.0, process_id=5000),
        _tuning(paired, 10, (50,), 1.0, process_id=5000),
    )
    with pytest.raises(ValueError, match="duplicate process identity"):
        select_policy_for_budget(
            ratio=0.5,
            budget_bytes=80,
            policies=(policy, paired),
            measurements=duplicate_process,
        )


def test_measurement_budget_slot_status_and_oom_protocol():
    policy = _policy(0.5)
    with pytest.raises(ValueError, match="complete budget slot"):
        replace(_tuning(policy, 10, (50,), 1.0), cohort="evaluation")
    with pytest.raises(ValueError, match="does not match"):
        replace(_evaluation(policy, 20, 0.8, 80, 81), status="ok")
    violation = _evaluation(policy, 20, 0.8, 80, 81)
    assert violation.recomputed_status == "budget_violation"
    with pytest.raises(ValueError, match="cannot contain"):
        replace(_tuning(policy, 10, (50,), 1.0), status="oom")


def test_measurement_source_identity_must_match_registered_policy():
    policy = _policy(0.5)
    mismatch = replace(_tuning(policy, 10, (50,), 1.0), source_key=_fp("other-source"))
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        select_policy_for_budget(
            ratio=0.5, budget_bytes=80, policies=(policy,), measurements=(mismatch,)
        )


def _selected_manifest(evaluation_count: int = 5) -> CalibrationManifest:
    reference = replace(
        _reference(), per_replicate_peak_bytes=((100,), (100,), (100,), (100,), (100,))
    )
    fast = _policy(0.5, fw="fast")
    slow = _policy(0.8, fw="slow")
    tuning = (
        _tuning(fast, 10, (70,), 1.0),
        _tuning(slow, 10, (60,), 2.0, process_id=3010),  # Paired seed across policies.
    )
    selection = select_policy_for_budget(
        ratio=0.8, budget_bytes=80, policies=(fast, slow), measurements=tuning
    )
    evaluation = tuple(
        _evaluation(fast, seed, 0.8, 80, 75, process_id=4000 + seed)
        for seed in range(20, 20 + evaluation_count)
    )
    return CalibrationManifest(
        reference=reference,
        ratios=(0.8,),
        policies=(fast, slow),
        tuning_measurements=tuning,
        evaluation_measurements=evaluation,
        selections=(selection,),
    )


def test_manifest_requires_five_held_out_processes_per_feasible_budget():
    manifest = _selected_manifest()
    assert manifest.selections[0].status == "selected"
    with pytest.raises(ValueError, match="at least 5 evaluation processes"):
        _selected_manifest(evaluation_count=4)


def test_manifest_distinguishes_same_policy_across_budget_slots():
    reference = replace(
        _reference(), per_replicate_peak_bytes=((100,), (100,), (100,), (100,), (100,))
    )
    policy = _policy(0.5)
    tuning = (_tuning(policy, 10, (40,), 1.0),)
    ratios = (0.5, 0.8)
    selections = tuple(
        select_policy_for_budget(
            ratio=ratio,
            budget_bytes=budget,
            policies=(policy,),
            measurements=tuning,
        )
        for ratio, budget in zip(ratios, (50, 80))
    )
    evaluation = tuple(
        _evaluation(policy, seed, ratio, budget, 45, process_id=5000 + index * 100 + seed)
        for index, (ratio, budget, seeds) in enumerate(
            ((0.5, 50, range(20, 25)), (0.8, 80, range(30, 35)))
        )
        for seed in seeds
    )
    manifest = CalibrationManifest(
        reference=reference,
        ratios=ratios,
        policies=(policy,),
        tuning_measurements=tuning,
        evaluation_measurements=evaluation,
        selections=selections,
    )
    assert {row.budget_slot for row in manifest.evaluation_measurements} == {(0.5, 50), (0.8, 80)}


def test_manifest_seed_sets_are_pairwise_disjoint_but_tuning_pairing_is_allowed():
    manifest = _selected_manifest()
    assert {row.seed for row in manifest.tuning_measurements} == {10}
    with pytest.raises(ValueError, match="pairwise disjoint"):
        replace(
            manifest,
            evaluation_measurements=tuple(
                replace(row, seed=10 if index == 0 else row.seed)
                for index, row in enumerate(manifest.evaluation_measurements)
            ),
        )
    with pytest.raises(ValueError, match="pairwise disjoint"):
        replace(
            manifest,
            reference=replace(manifest.reference, replicate_seeds=(1, 2, 3, 4, 10)),
        )


def test_infeasible_budget_requires_no_evaluation_but_retains_rows():
    reference = replace(
        _reference(), per_replicate_peak_bytes=((100,), (100,), (100,), (100,), (100,))
    )
    policy = _policy(0.5)
    tuning = (_tuning(policy, 10, (90,), 1.0),)
    selection = select_policy_for_budget(
        ratio=0.8, budget_bytes=80, policies=(policy,), measurements=tuning
    )
    manifest = CalibrationManifest(
        reference=reference,
        ratios=(0.8,),
        policies=(policy,),
        tuning_measurements=tuning,
        evaluation_measurements=(),
        selections=(selection,),
    )
    assert manifest.selections[0].status == "infeasible"
    assert manifest.tuning_measurements == tuning


def test_evaluation_cannot_reselect_or_target_unfrozen_policy():
    manifest = _selected_manifest()
    other = manifest.policies[1]
    invalid = tuple(
        _evaluation(other, seed, 0.8, 80, 60, process_id=6000 + seed)
        for seed in range(20, 25)
    )
    with pytest.raises(ValueError, match="does not match a frozen"):
        replace(manifest, evaluation_measurements=invalid)


def test_physical_budget_has_no_inverse_ratio_conversion():
    converters = (
        physical_budget_to_ratio,
        convert_physical_budget_to_ratio,
        convert_physical_budget_to_activation_memory_budget,
    )
    for converter in converters:
        with pytest.raises(LegacyBudgetConversionError, match="not converted"):
            converter(physical_budget_bytes=80, fixed_physical_bytes=20, maximum_saved_bytes=100)
