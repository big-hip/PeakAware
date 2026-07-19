import json
import subprocess
import sys


def test_root_cause_evaluation_script_scores_prediction_rows(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.json"
    output_path = tmp_path / "nested" / "evaluation.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "plan_id": "case_a",
                    "primary_cause": "REMATERIALIZATION_WAVE",
                    "root_causes": ["REMATERIALIZATION_WAVE", "COST_MODEL_MISRANK"],
                }
            ]
        ),
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(
            [
                {
                    "plan_id": "case_a",
                    "primary_cause": "REMATERIALIZATION_WAVE",
                    "root_causes": ["REMATERIALIZATION_WAVE"],
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_root_causes.py",
            "--predictions",
            str(predictions_path),
            "--labels",
            str(labels_path),
            "--output-json",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload["case_count"] == 1
    assert stdout_payload["primary_accuracy"] == 1.0
    assert stdout_payload["micro_precision"] == 0.5
    assert stdout_payload["micro_recall"] == 1.0
    assert file_payload == stdout_payload
