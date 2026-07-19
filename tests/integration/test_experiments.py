import csv
import json
import subprocess
import sys

from peakaware.config import PeakAwareConfig
from peakaware.experiments import run_experiment_matrix, write_experiment_csv, write_experiment_json


def test_experiment_matrix_writes_json_and_csv(tmp_path):
    records = run_experiment_matrix(
        task_names=("tiny_residual_w8",),
        microbatch_sizes=(1,),
        budget_bytes=(1 << 28,),
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0, top_k=1),
    )
    json_path = tmp_path / "records.json"
    csv_path = tmp_path / "records.csv"

    write_experiment_json(records, json_path)
    write_experiment_csv(records, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].selected_plan_key is not None
    assert records[0].graph_key is not None
    assert records[0].selected_saved_value_ids
    assert set(records[0].selected_saved_value_ids).issubset(records[0].selected_effective_saved_value_ids)
    assert records[0].selected_estimated_peak_bytes is not None
    assert records[0].selected_prediction_error_bytes is not None
    assert records[0].candidate_count >= records[0].measured_candidate_count
    assert records[0].cache_total_hits == 0
    assert payload[0]["selected_plan_id"] is not None
    assert payload[0]["selected_plan_key"] == records[0].selected_plan_key
    assert payload[0]["graph_key"] == records[0].graph_key
    assert payload[0]["selected_saved_value_ids"] == list(records[0].selected_saved_value_ids)
    assert payload[0]["selected_effective_saved_value_ids"] == list(records[0].selected_effective_saved_value_ids)
    assert rows[0]["task_name"] == "tiny_residual_w8"
    assert rows[0]["graph_key"] == records[0].graph_key
    assert rows[0]["selected_estimated_peak_bytes"]
    assert "selected_prediction_error_bytes" in rows[0]


def test_run_experiments_script_writes_requested_artifacts(tmp_path):
    json_path = tmp_path / "records.json"
    csv_path = tmp_path / "records.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--tasks",
            "tiny_mlp_w8_d3",
            "--budget-mib",
            "256",
            "--microbatches",
            "1",
            "--top-k",
            "1",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert stdout_payload[0]["status"] == "ok"
    assert stdout_payload[0]["selected_plan_key"]
    assert stdout_payload[0]["graph_key"]
    assert stdout_payload[0]["selected_saved_value_ids"]
    assert stdout_payload[0]["selected_effective_saved_value_ids"]
    assert stdout_payload[0]["selected_prediction_error_bytes"] is not None
    assert stdout_payload[0]["cache_total_hits"] == 0
    assert file_payload[0]["task_name"] == "tiny_mlp_w8_d3"
    assert csv_path.read_text(encoding="utf-8").startswith("task_name,microbatch_size,budget_bytes")
