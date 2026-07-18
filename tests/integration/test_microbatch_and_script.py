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
