from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_main_experiment_artifact_accepts_event_only_timeline(tmp_path: Path):
    records_json = tmp_path / "records.json"
    output_root = tmp_path / "artifact"
    records_json.write_text(json.dumps([_ready_record()]) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_main_experiment_artifact.py",
            "--records-json",
            str(records_json),
            "--output-root",
            str(output_root),
            "--task-name",
            "toy",
            "--variant-name",
            "diagnostic_hints_on",
            "--require-gpu-util",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_root / "main_experiment_artifact_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((output_root / "derived" / "main_experiment_status.json").read_text(encoding="utf-8"))
    simulation = json.loads((output_root / "derived" / "simulation_accuracy.json").read_text(encoding="utf-8"))
    timeline_manifest = json.loads(
        (
            output_root
            / "figures"
            / "F_actual_vs_simulated_memory_timeline"
            / "timeline_artifact_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["main_question_status"] == "ready_for_final_main_experiment_claim"
    assert manifest["timeline_ready"] is True
    assert manifest["gpu_ready"] is True
    assert manifest["simulation_accuracy"]["candidate_count"] == 1
    assert status["blockers"] == []
    assert simulation["candidate_count"] == 1
    assert simulation["memory"]["raw"]["count"] == 1
    assert simulation["time"]["raw_costmodel"]["count"] == 1
    assert timeline_manifest["mode"] == "continuous_event_timeline"
    assert "actual_vs_simulated_event_gpu_figure.svg" in timeline_manifest["generated_files"]


def _ready_record() -> dict:
    return {
        "status": "ok",
        "task_name": "toy",
        "variant_name": "diagnostic_hints_on",
        "microbatch_size": 1,
        "budget_bytes": 4096,
        "selected_plan_id": "peakaware_plan",
        "measured_peak_bytes": 3000,
        "measured_step_us": 30.0,
        "measured_fw_us": 10.0,
        "measured_bw_us": 15.0,
        "measured_optimizer_us": 5.0,
        "measurement_repeats": 20,
        "measurement_warmup_steps": 5,
        "selected_actual_overall_sampled_memory_trace": [
            {"time_us": 0.0, "phase": "fw", "allocated_bytes": 1000, "reserved_bytes": 1200, "allocator_peak_bytes": 1000},
            {"time_us": 10.0, "phase": "fw", "allocated_bytes": 2200, "reserved_bytes": 2400, "allocator_peak_bytes": 2200},
            {"time_us": 25.0, "phase": "bw", "allocated_bytes": 3000, "reserved_bytes": 3200, "allocator_peak_bytes": 3000},
            {"time_us": 30.0, "phase": "optimizer", "allocated_bytes": 1800, "reserved_bytes": 3200, "allocator_peak_bytes": 3000},
        ],
        "selected_lowered_fx_l2_simulated_memory_event_trace": [
            {"time_us": 0.0, "phase": "fw", "event": "phase_start", "bytes": 900, "payload_bytes": 900},
            {"time_us": 10.0, "phase": "fw", "event": "op", "bytes": 2100, "payload_bytes": 2100},
            {"time_us": 25.0, "phase": "bw", "event": "op", "bytes": 2900, "payload_bytes": 2900},
            {"time_us": 30.0, "phase": "optimizer", "event": "op", "bytes": 1700, "payload_bytes": 1700},
        ],
        "selected_gpu_util_trace": [
            {"time_us": 0.0, "gpu_util_percent": 20, "memory_util_percent": 30, "source": "test"},
            {"time_us": 10.0, "gpu_util_percent": 80, "memory_util_percent": 45, "source": "test"},
            {"time_us": 30.0, "gpu_util_percent": 35, "memory_util_percent": 40, "source": "test"},
        ],
        "measured_plan_results": [
            {
                "plan_id": "peakaware_plan",
                "measured_peak_bytes": 3000,
                "measured_step_us": 30.0,
                "strategy_provenance": {"source": "pytorch_min_cut"},
            }
        ],
    }
