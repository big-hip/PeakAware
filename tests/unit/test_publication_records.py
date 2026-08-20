import json

from peakaware.experiments import write_experiment_json
from peakaware.experiments import _measured_plan_results
from peakaware.experiments import summarize_selected_regret
from peakaware.publication.records import validate_publication_records
from tests.unit.test_publication_figures import _record


def test_validate_publication_records_accepts_valid_rows(tmp_path):
    path = tmp_path / "records.json"
    write_experiment_json((_record(),), path)

    payload = validate_publication_records(path)

    assert payload["ok"] is True
    assert payload["record_count"] == 1
    assert payload["status_counts"] == {"ok": 1}


def test_validate_publication_records_rejects_missing_file(tmp_path):
    payload = validate_publication_records(tmp_path / "missing.json")

    assert payload["ok"] is False
    assert "missing records file" in payload["errors"][0]


def test_validate_publication_records_can_require_runtime_identity(tmp_path):
    record = _record()
    path = tmp_path / "records.json"
    write_experiment_json((record,), path)

    payload = validate_publication_records(path, require_runtime_identity=True)

    assert payload["ok"] is True

    broken = {**json.loads(path.read_text(encoding="utf-8"))[0], "selected_aot_partition_runtime": False}
    path.write_text(json.dumps([broken]), encoding="utf-8")
    failed = validate_publication_records(path, require_runtime_identity=True)

    assert failed["ok"] is False
    assert any("runtime identity" in error for error in failed["errors"])


def test_measured_plan_results_preserve_strategy_provenance():
    rows = _measured_plan_results(
        {
            "plans": (
                {
                    "plan_id": "torch_min_cut",
                    "estimated_peak_bytes": 80,
                    "estimated_step_us": 10.0,
                    "strategy_provenance": {"source": "pytorch_min_cut_proxy"},
                    "simulated_memory_timeline": (
                        {"phase": "start", "position": 0.0, "bytes": 0},
                        {"phase": "overall_peak", "position": 5.0, "bytes": 80},
                    ),
                },
            ),
            "measured_candidates": (
                {
                    "plan_id": "torch_min_cut",
                    "peak_bytes": 82,
                    "step_us": 11.0,
                    "peak_phase": "bw",
                    "phase_metrics": {},
                    "actual_memory_timeline": (
                        {"phase": "start", "position": 0.0, "bytes": 0},
                        {"phase": "overall_peak", "position": 5.0, "bytes": 82},
                    ),
                    "prediction_error": {},
                    "correctness_passed": True,
                },
            ),
        }
    )

    assert rows[0]["strategy_provenance"] == {"source": "pytorch_min_cut_proxy"}
    assert rows[0]["memory_timeline_fit"]["point_count"] == 2


def test_measured_plan_results_flags_proxy_memory_realization_gap():
    rows = _measured_plan_results(
        {
            "plans": (
                {
                    "plan_id": "all_save",
                    "estimated_peak_bytes": 400 << 20,
                    "estimated_step_us": 10.0,
                    "simulated_memory_timeline": (),
                },
                {
                    "plan_id": "block_checkpoint",
                    "estimated_peak_bytes": 250 << 20,
                    "estimated_step_us": 12.0,
                    "simulated_memory_timeline": (),
                },
            ),
            "measured_candidates": (
                {
                    "plan_id": "all_save",
                    "peak_bytes": 410 << 20,
                    "step_us": 11.0,
                    "peak_phase": "bw",
                    "phase_metrics": {},
                    "actual_memory_timeline": (),
                    "prediction_error": {},
                    "correctness_passed": True,
                },
                {
                    "plan_id": "block_checkpoint",
                    "peak_bytes": 409 << 20,
                    "step_us": 12.0,
                    "peak_phase": "bw",
                    "phase_metrics": {},
                    "actual_memory_timeline": (),
                    "prediction_error": {},
                    "correctness_passed": True,
                },
            ),
        }
    )

    by_id = {row["plan_id"]: row for row in rows}

    assert by_id["all_save"]["memory_realization_gap_detected"] is None
    assert by_id["block_checkpoint"]["memory_realization_gap_detected"] is True
    assert by_id["block_checkpoint"]["memory_realization_gap_bytes"] == 149 << 20


def test_selected_regret_uses_best_measured_feasible_candidate():
    record = _record()

    payload = summarize_selected_regret((record,))
    row = payload["rows"][0]

    assert payload["row_count"] == 1
    assert row["selected_plan_id"] == "peakaware"
    assert row["best_feasible_plan_id"] == "peakaware"
    assert row["selected_regret_us"] == 0.0
    assert payload["selected_best_feasible_rate"] == 1.0
