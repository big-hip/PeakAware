import json
import subprocess
import sys

from peakaware.hint_benchmark import run_synthetic_hint_effectiveness_benchmark


def test_synthetic_hint_effectiveness_benchmark_reports_order_change():
    payload = run_synthetic_hint_effectiveness_benchmark()
    row = payload["rows"][0]

    assert payload["case_count"] == 1
    assert payload["changed_search_order_case_count"] == 1
    assert payload["verdict"] == "changed_search_order"
    assert row["candidate_match_delta"] == 2
    assert row["order_delta_count_delta"] > 0
    assert row["enabled"]["diagnostic_hint_count"] > 0
    assert row["enabled"]["diagnostic_hint_order_changed"] is True
    assert row["disabled"]["diagnostic_hint_order_changed"] is False


def test_hint_effectiveness_benchmark_script_writes_json(tmp_path):
    output_path = tmp_path / "nested" / "hint_effectiveness.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_hint_effectiveness_benchmark.py",
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
    assert stdout_payload["verdict"] == "changed_search_order"
