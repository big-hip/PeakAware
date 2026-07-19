import csv
import json
import subprocess
import sys

import torch
from torch import nn

from peakaware.config import PeakAwareConfig
from peakaware.contracts import TrainingTaskSpec
from peakaware.experiments import (
    ExperimentRecord,
    run_experiment_matrix,
    summarize_experiment_records,
    write_experiment_csv,
    write_experiment_json,
    write_experiment_summary_json,
)
from peakaware.models import TrainingTaskRegistry


def _build_linear_model() -> nn.Module:
    return nn.Linear(1, 1)


def _build_linear_batch(batch_size: int):
    return (torch.randn(batch_size, 1),), {}


def _linear_loss(out: torch.Tensor) -> torch.Tensor:
    return out.sum()


def _build_linear_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.1)


def _build_exact_small_task() -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name="exact_linear",
        build_model=_build_linear_model,
        build_batch=_build_linear_batch,
        loss_fn=_linear_loss,
        build_optimizer=_build_linear_optimizer,
    )


def _minimal_record(
    *,
    status: str,
    budget_bytes: int,
    measured_peak_bytes: int | None,
    samples_per_second: float | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        variant_name="diagnostic_hints_on" if status == "ok" else "failed",
        task_name="synthetic",
        microbatch_size=1,
        budget_bytes=budget_bytes,
        status=status,
        selected_plan_id="p" if status == "ok" else None,
        selected_plan_key="k" if status == "ok" else None,
        graph_key="g" if status == "ok" else None,
        selected_saved_value_ids=(1,) if status == "ok" else (),
        selected_effective_saved_value_ids=(1,) if status == "ok" else (),
        selected_estimated_peak_bytes=measured_peak_bytes,
        baseline_plan_id="all_save" if status == "ok" else None,
        baseline_estimated_peak_bytes=None if measured_peak_bytes is None else measured_peak_bytes + 20,
        selected_estimated_peak_reduction_bytes=20 if status == "ok" else None,
        measured_peak_bytes=measured_peak_bytes,
        measured_peak_reserved_bytes=0 if measured_peak_bytes is not None else None,
        measured_budget_headroom_bytes=None if measured_peak_bytes is None else budget_bytes - measured_peak_bytes,
        all_save_measured_peak_bytes=None if measured_peak_bytes is None else measured_peak_bytes + 10,
        all_save_measured_step_us=12.0 if status == "ok" else None,
        selected_measured_peak_reduction_vs_all_save_bytes=10 if status == "ok" else None,
        selected_step_time_delta_vs_all_save_us=2.0 if status == "ok" else None,
        selected_samples_per_second_speedup_vs_all_save=1.2 if status == "ok" else None,
        measured_step_us=10.0 if status == "ok" else None,
        measurement_repeats=1 if status == "ok" else None,
        measurement_warmup_steps=0 if status == "ok" else None,
        samples_per_second=samples_per_second,
        feasibility_status="FEASIBLE" if status == "ok" else None,
        baseline_peak_phase="fw" if status == "ok" else None,
        selected_peak_phase="bw" if status == "ok" else None,
        measured_peak_phase="bw" if status == "ok" else None,
        selected_peak_phase_match=True if status == "ok" else None,
        diagnostic_primary_cause="REMATERIALIZATION_WAVE" if status == "ok" else None,
        diagnostic_normalized_saved_reduction_bytes=32 if status == "ok" else None,
        diagnostic_realization_gap_bytes=12 if status == "ok" else None,
        diagnostic_total_expectation_gap_bytes=None,
        measured_candidate_count=1 if status == "ok" else 0,
        selected_prediction_error_bytes=4 if status == "ok" else None,
        selected_prediction_relative_error=0.05 if status == "ok" else None,
        simulation_accuracy_candidate_count=2 if status == "ok" else 0,
        simulation_accuracy_mean_absolute_error_bytes=6.0 if status == "ok" else None,
        simulation_accuracy_max_absolute_error_bytes=8 if status == "ok" else None,
        simulation_accuracy_mean_absolute_relative_error=0.075 if status == "ok" else None,
        simulation_accuracy_within_10_percent_rate=0.5 if status == "ok" else None,
        cache_total_hits=1,
        cache_total_misses=1,
        cache_hit_rate=0.5,
        candidate_count=1 if status == "ok" else 0,
        fallback_plan_ids=(),
        diagnostic_hints_enabled=True if status == "ok" else None,
        diagnostic_hint_count=2 if status == "ok" else 0,
        diagnostic_hint_kinds=("SAVE_PEAK_STORAGE",) if status == "ok" else (),
        repaired_candidate_count=1 if status == "ok" else 0,
        repair_success_count=1 if status == "ok" else 0,
        feasible_before_repair_count=0 if status == "ok" else 0,
        feasible_after_repair_count=1 if status == "ok" else 0,
    )


def test_experiment_summary_counts_budget_violations_and_failures():
    summary = summarize_experiment_records(
        (
            _minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80, samples_per_second=10.0),
            _minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=120, samples_per_second=20.0),
            _minimal_record(status="failed", budget_bytes=100, measured_peak_bytes=None),
        )
    )

    assert summary.total_records == 3
    assert summary.ok_records == 2
    assert summary.failed_records == 1
    assert summary.success_rate == 2 / 3
    assert "torch_version" in summary.environment_fingerprint
    assert "python_version" in summary.environment_fingerprint
    assert summary.variant_counts == {"diagnostic_hints_on": 2, "failed": 1}
    assert summary.budget_violation_count == 1
    assert summary.budget_violation_rate == 0.5
    assert summary.mean_samples_per_second == 15.0
    assert summary.mean_measured_budget_headroom_bytes == 0.0
    assert summary.mean_measured_peak_reduction_vs_all_save_bytes == 10.0
    assert summary.mean_selected_step_time_delta_vs_all_save_us == 2.0
    assert summary.mean_selected_samples_per_second_speedup_vs_all_save == 1.2
    assert summary.mean_estimated_peak_reduction_bytes == 20.0
    assert summary.selected_prediction_count == 2
    assert summary.mean_selected_prediction_absolute_error_bytes == 4.0
    assert summary.mean_selected_prediction_absolute_relative_error == 0.05
    assert summary.simulation_accuracy_candidate_count == 4
    assert summary.mean_simulation_accuracy_absolute_error_bytes == 6.0
    assert summary.max_simulation_accuracy_absolute_error_bytes == 8
    assert summary.mean_simulation_accuracy_within_10_percent_rate == 0.5
    assert summary.phase_classification_count == 2
    assert summary.phase_classification_accuracy == 1.0
    assert summary.root_cause_counts == {"REMATERIALIZATION_WAVE": 2}
    assert summary.selected_peak_phase_counts == {"bw": 2}
    assert summary.measured_peak_phase_counts == {"bw": 2}
    assert summary.diagnostic_hints_enabled_count == 2
    assert summary.diagnostic_hint_count == 4
    assert summary.diagnostic_hint_kind_counts == {"SAVE_PEAK_STORAGE": 2}
    assert summary.repaired_candidate_count == 2
    assert summary.repair_success_count == 2
    assert summary.repair_success_rate == 1.0
    assert summary.feasible_after_repair_count == 2
    assert summary.mean_diagnostic_normalized_saved_reduction_bytes == 32.0
    assert summary.mean_diagnostic_realization_gap_bytes == 12.0
    assert summary.aggregate_cache_hit_rate == 0.5


def test_experiment_matrix_writes_json_and_csv(tmp_path):
    records = run_experiment_matrix(
        task_names=("tiny_residual_w8",),
        microbatch_sizes=(1,),
        budget_bytes=(1 << 28,),
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0, top_k=1),
    )
    json_path = tmp_path / "records.json"
    csv_path = tmp_path / "records.csv"
    summary_path = tmp_path / "summary.json"

    write_experiment_json(records, json_path)
    write_experiment_csv(records, csv_path)
    summary = summarize_experiment_records(records)
    write_experiment_summary_json(summary, summary_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].variant_name == "default"
    assert records[0].selected_plan_key is not None
    assert records[0].graph_key is not None
    assert records[0].selected_saved_value_ids
    assert set(records[0].selected_saved_value_ids).issubset(records[0].selected_effective_saved_value_ids)
    assert records[0].selected_estimated_peak_bytes is not None
    assert records[0].baseline_plan_id == "all_save"
    assert records[0].baseline_estimated_peak_bytes is not None
    assert records[0].selected_estimated_peak_reduction_bytes is not None
    assert records[0].measured_budget_headroom_bytes is not None
    assert records[0].all_save_measured_peak_bytes is not None
    assert records[0].selected_measured_peak_reduction_vs_all_save_bytes is not None
    assert records[0].selected_samples_per_second_speedup_vs_all_save is not None
    assert records[0].selected_peak_phase is not None
    assert hasattr(records[0], "measured_peak_phase")
    assert hasattr(records[0], "selected_peak_phase_match")
    assert records[0].measurement_repeats == 1
    assert records[0].measurement_warmup_steps == 0
    assert records[0].simulation_accuracy_candidate_count >= 1
    assert records[0].diagnostic_hints_enabled is True
    assert records[0].diagnostic_hint_count >= 0
    assert records[0].selected_prediction_error_bytes is not None
    assert records[0].candidate_count >= records[0].measured_candidate_count
    assert records[0].cache_total_hits == 0
    assert payload[0]["selected_plan_id"] is not None
    assert payload[0]["variant_name"] == "default"
    assert payload[0]["selected_plan_key"] == records[0].selected_plan_key
    assert payload[0]["graph_key"] == records[0].graph_key
    assert payload[0]["selected_saved_value_ids"] == list(records[0].selected_saved_value_ids)
    assert payload[0]["selected_effective_saved_value_ids"] == list(records[0].selected_effective_saved_value_ids)
    assert rows[0]["task_name"] == "tiny_residual_w8"
    assert rows[0]["graph_key"] == records[0].graph_key
    assert rows[0]["selected_estimated_peak_bytes"]
    assert rows[0]["selected_estimated_peak_reduction_bytes"]
    assert rows[0]["selected_measured_peak_reduction_vs_all_save_bytes"]
    assert "selected_prediction_error_bytes" in rows[0]
    assert summary.total_records == 1
    assert summary.ok_records == 1
    assert summary.budget_violation_count == 0
    assert summary.max_feasible_microbatch == 1
    assert summary_payload["success_rate"] == 1.0
    assert "torch_version" in summary_payload["environment_fingerprint"]
    assert "cuda_available" in summary_payload["environment_fingerprint"]
    assert summary_payload["variant_counts"] == {"default": 1}
    assert summary_payload["mean_measured_peak_reduction_vs_all_save_bytes"] is not None
    assert "phase_classification_accuracy" in summary_payload
    assert "measured_peak_phase_counts" in summary_payload
    assert summary_payload["selected_prediction_count"] == 1
    assert summary_payload["simulation_accuracy_candidate_count"] >= 1
    assert summary_payload["root_cause_counts"]
    assert "diagnostic_hint_count" in summary_payload
    assert "repair_success_rate" in summary_payload


def test_experiment_matrix_can_include_exact_small_graph_baseline():
    registry = TrainingTaskRegistry()
    registry.register(_build_exact_small_task())
    records = run_experiment_matrix(
        task_names=("exact_linear",),
        microbatch_sizes=(1,),
        budget_bytes=(1 << 28,),
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0, top_k=1),
        registry=registry,
        include_exact_baseline=True,
    )

    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].exact_plan_key is not None
    assert records[0].exact_estimated_peak_bytes is not None
    assert records[0].selected_exact_peak_gap_bytes is not None
    assert records[0].selected_exact_peak_gap_bytes >= 0
    assert records[0].exact_error_type is None


def test_experiment_matrix_records_exact_baseline_fail_closed():
    records = run_experiment_matrix(
        task_names=("tiny_residual_w8",),
        microbatch_sizes=(1,),
        budget_bytes=(1 << 28,),
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0, top_k=1),
        include_exact_baseline=True,
        exact_max_candidate_count=0,
    )

    assert records[0].status == "ok"
    assert records[0].exact_plan_key is None
    assert records[0].exact_error_type == "PlanValidationError"


def test_run_experiments_script_writes_requested_artifacts(tmp_path):
    json_path = tmp_path / "records.json"
    csv_path = tmp_path / "records.csv"
    summary_path = tmp_path / "summary.json"
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
            "--diagnostic-hints",
            "both",
            "--selection-objective",
            "min_peak_then_time",
            "--measurement-repeats",
            "2",
            "--cache-root",
            str(tmp_path / "cache"),
            "--exact-small-graph",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--output-summary-json",
            str(summary_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(stdout_payload) == 2
    assert {record["variant_name"] for record in stdout_payload} == {
        "diagnostic_hints_on",
        "diagnostic_hints_off",
    }
    assert all(record["status"] == "ok" for record in stdout_payload)
    assert stdout_payload[0]["status"] == "ok"
    assert stdout_payload[0]["selected_plan_key"]
    assert stdout_payload[0]["exact_plan_key"] is None
    assert stdout_payload[0]["exact_error_type"] == "PlanValidationError"
    assert stdout_payload[0]["graph_key"]
    assert stdout_payload[0]["selected_saved_value_ids"]
    assert stdout_payload[0]["selected_effective_saved_value_ids"]
    assert stdout_payload[0]["selected_prediction_error_bytes"] is not None
    assert stdout_payload[0]["selected_estimated_peak_reduction_bytes"] is not None
    assert stdout_payload[0]["selected_measured_peak_reduction_vs_all_save_bytes"] is not None
    assert "selected_peak_phase_match" in stdout_payload[0]
    assert all(record["measurement_repeats"] == 2 for record in stdout_payload)
    assert stdout_payload[0]["simulation_accuracy_candidate_count"] >= 1
    assert {record["diagnostic_hints_enabled"] for record in stdout_payload} == {True, False}
    assert stdout_payload[0]["cache_total_hits"] == 0
    assert file_payload[0]["task_name"] == "tiny_mlp_w8_d3"
    assert file_payload[0]["exact_error_type"] == "PlanValidationError"
    assert summary_payload["total_records"] == 2
    assert summary_payload["ok_records"] == 2
    assert "python_version" in summary_payload["environment_fingerprint"]
    assert summary_payload["variant_counts"] == {"diagnostic_hints_off": 1, "diagnostic_hints_on": 1}
    assert summary_payload["selected_prediction_count"] == 2
    assert summary_payload["mean_selected_samples_per_second_speedup_vs_all_save"] is not None
    assert "phase_classification_count" in summary_payload
    assert summary_payload["mean_simulation_accuracy_absolute_error_bytes"] is not None
    assert "diagnostic_hint_kind_counts" in summary_payload
    assert summary_payload["exact_failure_count"] == 2
    assert csv_path.read_text(encoding="utf-8").startswith("variant_name,task_name,microbatch_size,budget_bytes")
