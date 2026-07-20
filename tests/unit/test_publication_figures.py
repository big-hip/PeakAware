import csv
import json
import xml.etree.ElementTree as ET

from peakaware.experiments import (
    ExperimentRecord,
    experiment_records_from_dicts,
    experiment_records_to_dicts,
    write_experiment_json,
)
from peakaware.publication.figures import (
    build_publication_figures,
    validate_publication_figure_artifacts,
)


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        variant_name="diagnostic_hints_on",
        config_fingerprint={"compile_backend": "aot_eager", "top_k": 3},
        task_name="synthetic",
        microbatch_size=1,
        budget_bytes=90,
        status="ok",
        selected_plan_id="peakaware",
        selected_plan_key="selected-key",
        graph_key="graph",
        selected_saved_value_ids=(1,),
        selected_effective_saved_value_ids=(1,),
        selected_estimated_peak_bytes=82,
        baseline_plan_id="all_save",
        baseline_estimated_peak_bytes=100,
        selected_estimated_peak_reduction_bytes=18,
        measured_peak_bytes=80,
        measured_peak_reserved_bytes=88,
        measured_budget_headroom_bytes=10,
        all_save_measured_peak_bytes=100,
        all_save_measured_step_us=20.0,
        selected_measured_peak_reduction_vs_all_save_bytes=20,
        selected_step_time_delta_vs_all_save_us=8.0,
        selected_samples_per_second_speedup_vs_all_save=20.0 / 12.0,
        measured_step_us=12.0,
        measurement_repeats=20,
        measurement_warmup_steps=5,
        samples_per_second=83.3,
        feasibility_status="FEASIBLE",
        baseline_peak_phase="fw",
        selected_peak_phase="bw",
        measured_peak_phase="bw",
        selected_peak_phase_match=True,
        measured_fw_us=3.0,
        measured_bw_us=6.0,
        measured_optimizer_us=3.0,
        measured_fw_peak_bytes=70,
        measured_bw_peak_bytes=80,
        measured_optimizer_peak_bytes=60,
        diagnostic_primary_cause="REMATERIALIZATION_WAVE",
        diagnostic_normalized_saved_reduction_bytes=20,
        diagnostic_realization_gap_bytes=0,
        diagnostic_total_expectation_gap_bytes=18,
        diagnostic_counterfactuals=(
            {
                "level": "D0",
                "candidate_peak_bytes": 100,
                "candidate_peak_phase": "fw",
                "status": "available",
                "confidence": 0.7,
            },
            {
                "level": "D3",
                "candidate_peak_bytes": 82,
                "candidate_peak_phase": "bw",
                "status": "available",
                "confidence": 0.8,
            },
            {
                "level": "D5",
                "candidate_peak_bytes": 80,
                "candidate_peak_phase": "bw",
                "status": "available",
                "confidence": 0.9,
            },
        ),
        measured_candidate_count=3,
        measured_plan_results=(
            {
                "plan_id": "all_save",
                "estimated_peak_bytes": 100,
                "measured_peak_bytes": 100,
                "measured_step_us": 20.0,
                "measured_peak_phase": "fw",
                "correctness_passed": True,
                "calibrated_estimated_peak_bytes": 100,
                "phase_metrics": {
                    "fw_peak_bytes": 100,
                    "bw_peak_bytes": 90,
                    "optimizer_peak_bytes": 70,
                },
            },
            {
                "plan_id": "torch_min_cut",
                "estimated_peak_bytes": 88,
                "measured_peak_bytes": 85,
                "measured_step_us": 14.0,
                "measured_peak_phase": "bw",
                "correctness_passed": True,
                "calibrated_estimated_peak_bytes": 88,
                "phase_metrics": {
                    "fw_peak_bytes": 65,
                    "bw_peak_bytes": 85,
                    "optimizer_peak_bytes": 70,
                },
            },
            {
                "plan_id": "peakaware",
                "estimated_peak_bytes": 82,
                "measured_peak_bytes": 80,
                "measured_step_us": 12.0,
                "measured_peak_phase": "bw",
                "correctness_passed": True,
                "calibrated_estimated_peak_bytes": 82,
                "phase_metrics": {
                    "fw_peak_bytes": 60,
                    "bw_peak_bytes": 80,
                    "optimizer_peak_bytes": 70,
                },
            },
        ),
        selected_prediction_error_bytes=2,
        selected_prediction_relative_error=0.025,
        selected_calibrated_prediction_error_bytes=2,
        selected_calibrated_prediction_relative_error=0.025,
        selected_feasibility_prediction_match=True,
        simulation_accuracy_candidate_count=3,
        simulation_accuracy_mean_absolute_error_bytes=2.0,
        simulation_accuracy_max_absolute_error_bytes=3,
        simulation_accuracy_mean_absolute_relative_error=0.025,
        simulation_accuracy_within_10_percent_rate=1.0,
        cache_total_hits=1,
        cache_total_misses=1,
        cache_hit_rate=0.5,
        cache_layer_hits={"analysis": 1},
        cache_layer_misses={"executable": 1},
        optimization_total_us=100.0,
        optimization_capture_us=10.0,
        optimization_ir_build_us=20.0,
        optimization_analysis_us=30.0,
        optimization_executor_build_us=10.0,
        optimization_candidate_validation_measurement_us=30.0,
        optimization_amortization_steps=50.0,
        actual_joint_capture_count=1,
        candidate_count=3,
        fallback_plan_ids=(),
        selected_activation_checkpoint=False,
        selected_aot_partition_runtime=True,
        activation_checkpoint_candidate_count=0,
        aot_partition_runtime_candidate_count=2,
        diagnostic_hints_enabled=True,
        diagnostic_hint_count=1,
        diagnostic_hint_kinds=("SAVE_PEAK_STORAGE",),
        diagnostic_hint_candidate_match_count=1,
        diagnostic_hint_order_changed=True,
        diagnostic_hint_order_delta_count=1,
        repaired_candidate_count=1,
        repair_success_count=1,
        feasible_before_repair_count=0,
        feasible_after_repair_count=1,
    )


def test_build_publication_figures_writes_auditable_artifact_directories(tmp_path):
    records_path = tmp_path / "records.json"
    write_experiment_json((_record(),), records_path)

    artifacts = build_publication_figures(
        (_record(),),
        tmp_path / "figures",
        status="provisional",
        ev_ids=("EV-TEST",),
        source_paths=(records_path,),
        command="generate",
    )

    assert {artifact.figure_id for artifact in artifacts} == {
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
    }
    f2 = tmp_path / "figures" / "F2_pareto"
    assert (f2 / "figure.svg").exists()
    assert (f2 / "source.csv").exists()
    assert (f2 / "source.schema.json").exists()
    assert (f2 / "plot_config.json").exists()
    assert (f2 / "caption.md").exists()
    provenance = json.loads((f2 / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "provisional"
    assert provenance["ev_ids"] == ["EV-TEST"]
    assert provenance["source_checksums"][str(records_path)]
    ET.parse(f2 / "figure.svg")

    with (f2 / "source.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["strategy"] for row in rows} == {"all_save", "torch_min_cut", "peakaware"}
    assert any(row["normalized_peak"] == "0.8" for row in rows)
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F3").row_count == 3
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F4").row_count == 2
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F6").row_count == 3
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F7").row_count == 9
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F8").row_count == 3
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F9").row_count == 5
    assert next(artifact for artifact in artifacts if artifact.figure_id == "F10").row_count == 1


def test_records_round_trip_still_builds_figures(tmp_path):
    payload = experiment_records_to_dicts((_record(),))
    records = experiment_records_from_dicts(payload)

    artifacts = build_publication_figures(records, tmp_path / "figures")

    assert all(artifact.row_count > 0 for artifact in artifacts)


def test_figures_label_capture_and_compile_backend(tmp_path):
    record = _record()
    payload = experiment_records_to_dicts((record,))[0]
    payload["config_fingerprint"] = {**payload["config_fingerprint"], "capture_backend": "aot"}
    records = experiment_records_from_dicts([payload])

    build_publication_figures(records, tmp_path / "figures")

    with (tmp_path / "figures" / "F2_pareto" / "source.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["backend"] for row in rows} == {"aot/aot_eager"}


def test_frozen_figures_require_evidence_and_checksum_sources(tmp_path):
    records = (_record(),)

    try:
        build_publication_figures(records, tmp_path / "figures", status="frozen")
    except ValueError as exc:
        assert "EV-*" in str(exc)
    else:
        raise AssertionError("frozen figure generation accepted missing evidence ids")

    try:
        build_publication_figures(
            records,
            tmp_path / "figures",
            status="frozen",
            ev_ids=("EV-TEST",),
            source_paths=(tmp_path / "missing.json",),
        )
    except ValueError as exc:
        assert "source files" in str(exc)
    else:
        raise AssertionError("frozen figure generation accepted a missing source file")


def test_validate_publication_figure_artifacts_reports_ok_and_failures(tmp_path):
    records_path = tmp_path / "records.json"
    write_experiment_json((_record(),), records_path)
    figure_root = tmp_path / "figures"
    build_publication_figures(
        (_record(),),
        figure_root,
        status="draft",
        source_paths=(records_path,),
    )

    payload = validate_publication_figure_artifacts(figure_root)

    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert {item["status"] for item in payload["figures"]} == {"draft"}
    assert all(item["row_count"] > 0 for item in payload["figures"])

    (figure_root / "F2_pareto" / "source.csv").write_text("broken\n", encoding="utf-8")
    broken = validate_publication_figure_artifacts(figure_root)

    assert broken["ok"] is False
    assert any(
        "source.schema.json row_count mismatch" in error
        or "source_csv_sha256 mismatch" in error
        for item in broken["figures"]
        for error in item["errors"]
    )


def test_validate_publication_figure_artifacts_enforces_frozen_status(tmp_path):
    figure_root = tmp_path / "figures"
    build_publication_figures((_record(),), figure_root, status="draft")

    payload = validate_publication_figure_artifacts(figure_root, require_frozen=True)

    assert payload["ok"] is False
    assert any(
        "expected frozen figure status" in error
        for item in payload["figures"]
        for error in item["errors"]
    )
