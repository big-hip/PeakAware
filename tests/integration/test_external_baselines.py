import json
import subprocess
import sys

from peakaware.external_baselines import summarize_external_baseline_capabilities


def test_external_baseline_capabilities_report_current_torch_apis():
    payload = summarize_external_baseline_capabilities()
    baselines = payload["baselines"]

    assert payload["environment"]["torch_version"]
    assert set(baselines) == {
        "aot_min_cut_proxy",
        "inductor_memory_budget",
        "selective_activation_checkpointing",
    }
    assert baselines["aot_min_cut_proxy"]["status"] == "proxy"
    assert baselines["selective_activation_checkpointing"]["status"] in {"available", "unavailable"}
    assert baselines["inductor_memory_budget"]["status"] in {"available", "unavailable"}
    if baselines["inductor_memory_budget"]["status"] == "unavailable":
        assert baselines["inductor_memory_budget"]["unavailable_reason"]


def test_inspect_external_baselines_script_writes_json(tmp_path):
    output_path = tmp_path / "external_baselines.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_external_baselines.py",
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
    assert "selective_activation_checkpointing" in stdout_payload["baselines"]
