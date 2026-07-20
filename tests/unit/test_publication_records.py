import json

from peakaware.experiments import write_experiment_json
from peakaware.experiments import _measured_plan_results
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
                },
            ),
            "measured_candidates": (
                {
                    "plan_id": "torch_min_cut",
                    "peak_bytes": 82,
                    "step_us": 11.0,
                    "peak_phase": "bw",
                    "phase_metrics": {},
                    "prediction_error": {},
                    "correctness_passed": True,
                },
            ),
        }
    )

    assert rows[0]["strategy_provenance"] == {"source": "pytorch_min_cut_proxy"}
