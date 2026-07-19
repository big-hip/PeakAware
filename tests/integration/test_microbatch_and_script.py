import json
import subprocess
import sys

from peakaware.config import PeakAwareConfig
from peakaware.microbatch import optimize_microbatches
from peakaware.models import TrainingTaskRegistry


def test_microbatch_optimizer_selects_candidate():
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")

    result = optimize_microbatches(
        task,
        (1, 2),
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0),
    )

    assert result.candidates
    assert result.selected in result.candidates
    assert result.selected.useful_samples_per_second > 0


def test_microbatch_optimizer_supports_isolated_default_tasks():
    task = TrainingTaskRegistry.with_defaults().get("tiny_mlp_w8_d3")

    result = optimize_microbatches(
        task,
        (1,),
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=1,
            isolate_candidate_measurement=True,
            candidate_worker_timeout_s=30.0,
        ),
    )

    assert result.selected.microbatch_size == 1
    assert result.selected.result.executable.correctness_passed


def test_run_mvp_script_outputs_json():
    completed = subprocess.run(
        [sys.executable, "scripts/run_mvp.py", "--budget-mib", "256"],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "single"
    assert payload["dry_run_passed"] is True
    assert payload["report"]["diagnostic"]["counterfactuals"]


def test_run_mvp_script_writes_report_json(tmp_path):
    report_path = tmp_path / "mvp.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_mvp.py",
            "--budget-mib",
            "256",
            "--microbatches",
            "1,2",
            "--report-json",
            str(report_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_payload["mode"] == "microbatch"
    assert file_payload["selected_report"]["diagnostic"]["counterfactuals"]
