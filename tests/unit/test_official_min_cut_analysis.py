from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from peakaware.publication.min_cut_analysis import (
    OFFICIAL_MIN_CUT_API,
    OFFICIAL_MIN_CUT_SOLVER,
    analyze_official_min_cut,
)


ROOT = Path(__file__).resolve().parents[2]


def _record(task: str, ratio: float, peak: int, step: float, residuals: int) -> dict:
    return {
        "task_name": task,
        "display_name": task.upper(),
        "method": "pytorch_min_cut",
        "method_config": {"activation_memory_budget": ratio},
        "status": "ok",
        "backend": "aot_eager",
        "microbatch_size": 1,
        "workload_fingerprint": f"workload-{task}",
        "environment": {
            "gpu_name": "test-gpu",
            "gpu_uuid": "test-gpu-uuid",
            "torch_version": "2.test",
        },
        "seed": 1234,
        "execution_fingerprint": "execution-test",
        "correctness_report": {"passed": True},
        "runtime_identity": {
            "method_id": "pytorch_aot_min_cut",
            "api": OFFICIAL_MIN_CUT_API,
            "is_real": True,
            "model_sha256": f"model-{task}",
            "fw_residual_names": [f"v{index}" for index in range(residuals)],
            "provenance": {
                "solver": OFFICIAL_MIN_CUT_SOLVER,
                "partitioner_cost_model": "inductor",
                "memory_budget_ratio": ratio,
            },
        },
        "measurement_aggregate": {
            "publication_qualified": True,
            "overall_peak_bytes": peak,
            "overall_event_us": step,
            "overall_wall_us": step + 1.0,
            "measurement_repeats": 20,
            "measurement_warmup_steps": 5,
        },
    }


def test_analysis_computes_baseline_deltas_pareto_and_monotonicity():
    records = [
        _record("toy", 0.0, 80, 20.0, 1),
        _record("toy", 0.5, 90, 15.0, 2),
        _record("toy", 1.0, 100, 10.0, 3),
    ]

    payload = analyze_official_min_cut(records)

    assert payload["summary"]["identity_verified_count"] == 3
    assert payload["summary"]["qualified_record_count"] == 3
    assert payload["summary"]["residual_monotonic_pass_count"] == 2
    assert payload["summary"]["physical_peak_monotonic_pass_count"] == 2
    assert payload["summary"]["step_time_monotonic_pass_count"] == 2
    assert payload["summary"]["paired_protocol_ready"] is True
    assert payload["summary"]["comparison_readiness"] == "official_baseline_paired_canary"
    ratio_zero = payload["rows"][0]
    assert ratio_zero["peak_reduction_vs_baseline"] == pytest.approx(0.2)
    assert ratio_zero["time_overhead_vs_baseline"] == pytest.approx(1.0)
    assert [row["pareto_dominated"] for row in payload["rows"]] == [False, False, False]


def test_analysis_rejects_proxy_identity_from_qualified_count():
    record = _record("toy", 1.0, 100, 10.0, 3)
    record["runtime_identity"]["api"] = "peakaware.proxy"

    payload = analyze_official_min_cut([record])

    assert payload["summary"]["identity_verified_count"] == 0
    assert payload["summary"]["qualified_record_count"] == 0
    assert payload["rows"] == []
    assert "runtime_api_mismatch" in payload["record_audit"][0]["identity_failure_reasons"]


def test_analysis_marks_cross_device_or_cross_seed_sweep_unpaired():
    records = [
        _record("toy", 0.0, 80, 20.0, 1),
        _record("toy", 1.0, 100, 10.0, 3),
    ]
    records[1]["seed"] = 5678
    records[1]["environment"]["gpu_uuid"] = "other-gpu"

    payload = analyze_official_min_cut(records)

    assert payload["summary"]["paired_protocol_ready"] is False
    assert payload["summary"]["comparison_readiness"] == "official_baseline_unpaired_canary"
    assert payload["task_summaries"][0]["seed_consistent"] is False
    assert payload["task_summaries"][0]["gpu_uuid_consistent"] is False


def test_analysis_accepts_matching_multi_replicate_seed_and_model_sets():
    records = []
    for ratio in (0.0, 1.0):
        for replicate_index, seed in enumerate((1234, 5678)):
            record = _record("toy", ratio, 80 + int(ratio * 20), 20.0 - ratio * 10, 2)
            record["seed"] = seed
            record["runtime_identity"]["model_sha256"] = f"model-toy-seed-{seed}"
            record["replicate_index"] = replicate_index
            records.append(record)

    payload = analyze_official_min_cut(records)

    assert payload["summary"]["minimum_replicates_per_point"] == 2
    assert payload["summary"]["paired_protocol_ready"] is True
    assert payload["task_summaries"][0]["seeds"] == [1234, 5678]
    assert payload["task_summaries"][0]["model_initialization_consistent"] is True


def test_cli_writes_json_csv_and_report(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(_record("toy", ratio, 100 + int(ratio * 10), 20 - ratio * 10, 2))
        for ratio in (0.0, 0.5, 1.0))
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_official_min_cut_baseline.py"),
            str(records_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads((output_dir / "official_min_cut_analysis.json").read_text(encoding="utf-8"))
    assert payload["summary"]["qualified_record_count"] == 3
    assert (output_dir / "official_min_cut_pareto.csv").is_file()
    report = (output_dir / "OFFICIAL_MIN_CUT_BASELINE_REPORT.md").read_text(encoding="utf-8")
    assert "正式外部 planner" in report
    assert "不能据此宣称 PeakAware 整体优于官方 min-cut" in report
