import json
import subprocess
import sys

from peakaware.sac_baseline import run_sac_baseline, run_sac_baseline_matrix


def test_sac_baseline_reports_correctness_and_metrics_on_cpu():
    payload = run_sac_baseline(
        task_name="tiny_mlp_w8_d3",
        microbatch_size=1,
        device="cpu",
        warmup_steps=0,
        repeats=1,
    )

    assert payload["status"] == "ok"
    assert payload["baseline_id"] == "pytorch_sac_prefer_recompute"
    assert payload["correctness_passed"] is True
    assert payload["performance_result_usable"] is True
    assert payload["correctness_note"] is None
    assert "eager_metrics" in payload
    assert "sac_metrics" in payload


def test_run_sac_baseline_script_writes_json(tmp_path):
    output_path = tmp_path / "sac_baseline.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_sac_baseline.py",
            "--task",
            "tiny_mlp_w8_d3",
            "--microbatch",
            "1",
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert stdout_payload["correctness_passed"] is True


def test_sac_baseline_matrix_reports_usable_rows_on_cpu():
    payload = run_sac_baseline_matrix(
        task_names=("tiny_mlp_w8_d3",),
        microbatch_sizes=(1, 2),
        device="cpu",
        warmup_steps=0,
        repeats=1,
    )

    assert payload["baseline_id"] == "pytorch_sac_prefer_recompute"
    assert payload["row_count"] == 2
    assert payload["ok_count"] == 2
    assert payload["usable_count"] == 2
    assert payload["unusable_count"] == 0
    assert len(payload["rows"]) == 2


def test_run_sac_baseline_matrix_script_writes_json(tmp_path):
    output_path = tmp_path / "sac_matrix.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_sac_baseline_matrix.py",
            "--tasks",
            "tiny_mlp_w8_d3",
            "--microbatches",
            "1,2",
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert stdout_payload["row_count"] == 2
