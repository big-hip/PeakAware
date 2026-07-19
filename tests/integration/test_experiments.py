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
    summarize_baseline_comparisons,
    summarize_hint_ablation,
    summarize_layered_simulation_accuracy,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
    write_experiment_csv,
    write_experiment_baseline_comparison_json,
    write_experiment_hint_ablation_json,
    write_experiment_json,
    write_experiment_layered_accuracy_json,
    write_experiment_summary_json,
    write_experiment_variant_summary_json,
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
    variant_name: str | None = None,
    diagnostic_hints_enabled: bool | None = None,
) -> ExperimentRecord:
    variant_name = variant_name or ("diagnostic_hints_on" if status == "ok" else "failed")
    diagnostic_hints_enabled = status == "ok" if diagnostic_hints_enabled is None else diagnostic_hints_enabled
    diagnostic_counterfactuals = ()
    if status == "ok" and measured_peak_bytes is not None:
        diagnostic_counterfactuals = (
            {
                "level": "D0",
                "status": "available",
                "candidate_peak_bytes": measured_peak_bytes + 20,
                "candidate_peak_phase": "fw",
                "peak_gain_bytes": 0,
                "confidence": 0.7,
                "unavailable_reason": None,
            },
            {
                "level": "D3",
                "status": "available",
                "candidate_peak_bytes": measured_peak_bytes + 4,
                "candidate_peak_phase": "bw",
                "peak_gain_bytes": 0,
                "confidence": 0.8,
                "unavailable_reason": None,
            },
            {
                "level": "D5",
                "status": "available",
                "candidate_peak_bytes": measured_peak_bytes,
                "candidate_peak_phase": "bw",
                "peak_gain_bytes": 0,
                "confidence": 0.9,
                "unavailable_reason": None,
            },
        )
    return ExperimentRecord(
        variant_name=variant_name,
        config_fingerprint={
            "top_k": 1,
            "device": "cpu",
            "selection_objective": "min_peak_then_time",
            "enable_diagnostic_hints": diagnostic_hints_enabled,
        },
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
        diagnostic_counterfactuals=diagnostic_counterfactuals,
        measured_candidate_count=3 if status == "ok" else 0,
        measured_plan_results=()
        if status != "ok" or measured_peak_bytes is None
        else (
            {
                "plan_id": "all_save",
                "estimated_peak_bytes": 80,
                "measured_peak_bytes": measured_peak_bytes + 10,
                "measured_step_us": 12.0,
                "measured_feasible": measured_peak_bytes + 10 <= budget_bytes,
                "prediction_error_bytes": 8,
                "correctness_passed": True,
            },
            {
                "plan_id": "torch_min_cut",
                "estimated_peak_bytes": 80,
                "measured_peak_bytes": measured_peak_bytes,
                "measured_step_us": 10.0,
                "measured_feasible": measured_peak_bytes <= budget_bytes,
                "prediction_error_bytes": 4,
                "correctness_passed": True,
            },
            {
                "plan_id": "block_checkpoint",
                "estimated_peak_bytes": measured_peak_bytes + 5,
                "measured_peak_bytes": measured_peak_bytes + 5,
                "measured_step_us": 11.0,
                "measured_feasible": measured_peak_bytes + 5 <= budget_bytes,
                "prediction_error_bytes": None,
                "correctness_passed": True,
            },
        ),
        selected_prediction_error_bytes=4 if status == "ok" else None,
        selected_prediction_relative_error=0.05 if status == "ok" else None,
        selected_feasibility_prediction_match=True if status == "ok" else None,
        simulation_accuracy_candidate_count=2 if status == "ok" else 0,
        simulation_accuracy_mean_absolute_error_bytes=6.0 if status == "ok" else None,
        simulation_accuracy_max_absolute_error_bytes=8 if status == "ok" else None,
        simulation_accuracy_mean_absolute_relative_error=0.075 if status == "ok" else None,
        simulation_accuracy_within_10_percent_rate=0.5 if status == "ok" else None,
        cache_total_hits=1,
        cache_total_misses=1,
        cache_hit_rate=0.5,
        cache_layer_hits={"analysis": 1} if status == "ok" else {},
        cache_layer_misses={"capture": 1} if status == "ok" else {},
        optimization_total_us=100.0 if status == "ok" else None,
        optimization_capture_us=10.0 if status == "ok" else None,
        optimization_ir_build_us=20.0 if status == "ok" else None,
        optimization_analysis_us=30.0 if status == "ok" else None,
        optimization_executor_build_us=5.0 if status == "ok" else None,
        optimization_candidate_validation_measurement_us=35.0 if status == "ok" else None,
        optimization_amortization_steps=50.0 if status == "ok" else None,
        actual_joint_capture_count=1 if status == "ok" else 0,
        candidate_count=1 if status == "ok" else 0,
        fallback_plan_ids=(),
        diagnostic_hints_enabled=diagnostic_hints_enabled if status == "ok" else None,
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
    assert summary.p50_selected_prediction_absolute_error_bytes == 4.0
    assert summary.p90_selected_prediction_absolute_error_bytes == 4.0
    assert summary.mean_selected_prediction_absolute_relative_error == 0.05
    assert summary.p50_selected_prediction_absolute_relative_error == 0.05
    assert summary.p90_selected_prediction_absolute_relative_error == 0.05
    assert summary.simulation_accuracy_candidate_count == 4
    assert summary.mean_simulation_accuracy_absolute_error_bytes == 6.0
    assert summary.p50_simulation_accuracy_absolute_error_bytes == 8.0
    assert summary.p90_simulation_accuracy_absolute_error_bytes == 8.0
    assert summary.max_simulation_accuracy_absolute_error_bytes == 8
    assert summary.mean_simulation_accuracy_absolute_relative_error == 0.075
    assert summary.p50_simulation_accuracy_absolute_relative_error == 0.1
    assert summary.p90_simulation_accuracy_absolute_relative_error == 0.1
    assert summary.mean_simulation_accuracy_within_10_percent_rate == 0.5
    assert summary.phase_classification_count == 2
    assert summary.phase_classification_accuracy == 1.0
    assert summary.feasible_classification_count == 2
    assert summary.feasible_classification_accuracy == 1.0
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
    assert summary.cache_layer_hits == {"analysis": 2}
    assert summary.cache_layer_misses == {"capture": 2}
    assert summary.cache_layer_hit_rates == {"analysis": 1.0, "capture": 0.0}
    assert summary.mean_optimization_total_us == 100.0
    assert summary.mean_optimization_capture_us == 10.0
    assert summary.mean_optimization_ir_build_us == 20.0
    assert summary.mean_optimization_analysis_us == 30.0
    assert summary.mean_optimization_executor_build_us == 5.0
    assert summary.mean_optimization_candidate_validation_measurement_us == 35.0
    assert summary.mean_optimization_amortization_steps == 50.0
    assert summary.total_actual_joint_capture_count == 2
    variant_summaries = summarize_experiment_records_by_variant(
        (
            _minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80, samples_per_second=10.0),
            _minimal_record(status="failed", budget_bytes=100, measured_peak_bytes=None),
        )
    )
    assert set(variant_summaries) == {"diagnostic_hints_on", "failed"}
    assert variant_summaries["diagnostic_hints_on"].ok_records == 1


def test_hint_ablation_summary_pairs_on_off_variants():
    records = (
        _minimal_record(
            status="ok",
            budget_bytes=100,
            measured_peak_bytes=80,
            samples_per_second=12.0,
            variant_name="diagnostic_hints_on",
            diagnostic_hints_enabled=True,
        ),
        _minimal_record(
            status="ok",
            budget_bytes=100,
            measured_peak_bytes=120,
            samples_per_second=10.0,
            variant_name="diagnostic_hints_off",
            diagnostic_hints_enabled=False,
        ),
        _minimal_record(
            status="ok",
            budget_bytes=100,
            measured_peak_bytes=80,
            samples_per_second=11.0,
            variant_name="unpaired",
            diagnostic_hints_enabled=True,
        ),
    )

    summary = summarize_hint_ablation(records)

    assert summary["pair_count"] == 1
    assert summary["both_ok_count"] == 1
    assert summary["success_rate_delta"] == 0.0
    assert summary["mean_samples_per_second_delta"] == 2.0
    assert summary["rows"][0]["budget_violation_delta"] == -1
    assert summary["rows"][0]["conclusion"] == "improved_budget"
    assert summary["conclusion_counts"] == {"improved_budget": 1}


def test_baseline_comparison_summary_reports_selected_deltas():
    records = (
        _minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80, samples_per_second=10.0),
        _minimal_record(status="failed", budget_bytes=100, measured_peak_bytes=None),
    )

    summary = summarize_baseline_comparisons(records)

    assert summary["row_count"] == 3
    assert set(summary["baseline_groups"]) == {"all_save", "block_checkpoint", "torch_min_cut"}
    assert summary["baseline_groups"]["all_save"]["measured_count"] == 1
    assert summary["baseline_groups"]["all_save"]["mean_peak_reduction_vs_plan_bytes"] == 10.0
    assert summary["baseline_groups"]["all_save"]["mean_step_time_delta_vs_plan_us"] == 2.0
    assert summary["baseline_groups"]["all_save"]["selected_peak_win_count"] == 1
    assert summary["baseline_groups"]["torch_min_cut"]["mean_samples_per_second_speedup_vs_plan"] == 1.0


def test_baseline_comparison_summary_can_include_usable_sac_rows():
    records = (_minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80),)
    sac_payload = {
        "rows": [
            {
                "task_name": "synthetic",
                "microbatch_size": 1,
                "device": "cpu",
                "status": "ok",
                "baseline_id": "pytorch_sac_prefer_recompute",
                "performance_result_usable": True,
                "correctness_passed": True,
                "sac_overall_peak_bytes": 95,
                "sac_step_us": 15.0,
            },
            {
                "task_name": "synthetic",
                "microbatch_size": 1,
                "device": "cpu",
                "status": "ok",
                "performance_result_usable": False,
                "sac_overall_peak_bytes": 70,
                "sac_step_us": 8.0,
            },
        ]
    }

    summary = summarize_baseline_comparisons(records, sac_baseline=sac_payload)

    assert summary["external_sac_rows_seen"] == 2
    assert summary["external_sac_rows_usable"] == 1
    assert summary["external_sac_keys_matched"] == 1
    assert summary["baseline_groups"]["pytorch_sac"]["measured_count"] == 1
    sac_row = [row for row in summary["rows"] if row["baseline_group"] == "pytorch_sac"][0]
    assert sac_row["peak_reduction_vs_plan_bytes"] == 15
    assert sac_row["step_time_delta_vs_plan_us"] == 5.0


def test_layered_simulation_accuracy_summarizes_diagnostic_counterfactuals():
    records = (
        _minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80),
        _minimal_record(status="failed", budget_bytes=100, measured_peak_bytes=None),
    )

    summary = summarize_layered_simulation_accuracy(records)

    assert summary["row_count"] == 3
    assert set(summary["level_summaries"]) == {"D0", "D3", "D5"}
    assert summary["level_summaries"]["D0"]["row_count"] == 1
    assert summary["level_summaries"]["D5"]["mean_absolute_error_bytes"] == 0.0
    assert summary["level_summaries"]["D5"]["max_absolute_error_bytes"] == 0
    assert summary["level_summaries"]["D5"]["within_10_percent_rate"] == 1.0
    assert summary["level_summaries"]["D5"]["phase_classification_accuracy"] == 1.0


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
    variant_summary_path = tmp_path / "variant_summary.json"

    write_experiment_json(records, json_path)
    write_experiment_csv(records, csv_path)
    summary = summarize_experiment_records(records)
    write_experiment_summary_json(summary, summary_path)
    write_experiment_variant_summary_json(summarize_experiment_records_by_variant(records), variant_summary_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    variant_summary_payload = json.loads(variant_summary_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].variant_name == "default"
    assert records[0].config_fingerprint["device"] == "cpu"
    assert records[0].config_fingerprint["selection_objective"] == "min_peak_then_time"
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
    assert records[0].measured_plan_results
    assert records[0].diagnostic_counterfactuals
    assert records[0].diagnostic_hints_enabled is True
    assert records[0].diagnostic_hint_count >= 0
    assert records[0].selected_prediction_error_bytes is not None
    assert records[0].candidate_count >= records[0].measured_candidate_count
    assert records[0].cache_total_hits == 0
    assert isinstance(records[0].cache_layer_hits, dict)
    assert isinstance(records[0].cache_layer_misses, dict)
    assert payload[0]["selected_plan_id"] is not None
    assert payload[0]["variant_name"] == "default"
    assert payload[0]["config_fingerprint"]["top_k"] == 1
    assert payload[0]["selected_plan_key"] == records[0].selected_plan_key
    assert payload[0]["graph_key"] == records[0].graph_key
    assert payload[0]["selected_saved_value_ids"] == list(records[0].selected_saved_value_ids)
    assert payload[0]["selected_effective_saved_value_ids"] == list(records[0].selected_effective_saved_value_ids)
    assert payload[0]["diagnostic_counterfactuals"]
    assert rows[0]["task_name"] == "tiny_residual_w8"
    assert rows[0]["graph_key"] == records[0].graph_key
    assert rows[0]["selected_estimated_peak_bytes"]
    assert rows[0]["selected_estimated_peak_reduction_bytes"]
    assert rows[0]["selected_measured_peak_reduction_vs_all_save_bytes"]
    assert rows[0]["optimization_total_us"]
    assert rows[0]["actual_joint_capture_count"] == "1"
    assert "selected_prediction_error_bytes" in rows[0]
    assert summary.total_records == 1
    assert summary.ok_records == 1
    assert summary.budget_violation_count == 0
    assert summary.max_feasible_microbatch == 1
    assert summary_payload["success_rate"] == 1.0
    assert "torch_version" in summary_payload["environment_fingerprint"]
    assert "cuda_available" in summary_payload["environment_fingerprint"]
    assert summary_payload["variant_counts"] == {"default": 1}
    assert variant_summary_payload["default"]["total_records"] == 1
    assert summary_payload["mean_measured_peak_reduction_vs_all_save_bytes"] is not None
    assert "phase_classification_accuracy" in summary_payload
    assert "feasible_classification_accuracy" in summary_payload
    assert "measured_peak_phase_counts" in summary_payload
    assert summary_payload["selected_prediction_count"] == 1
    assert summary_payload["simulation_accuracy_candidate_count"] >= 1
    assert "p50_simulation_accuracy_absolute_error_bytes" in summary_payload
    assert "p90_simulation_accuracy_absolute_error_bytes" in summary_payload
    assert summary_payload["root_cause_counts"]
    assert "diagnostic_hint_count" in summary_payload
    assert "repair_success_rate" in summary_payload
    assert "mean_optimization_total_us" in summary_payload
    assert "cache_layer_hit_rates" in summary_payload
    assert summary_payload["total_actual_joint_capture_count"] == 1


def test_experiment_writers_create_parent_directories(tmp_path):
    records = (_minimal_record(status="ok", budget_bytes=100, measured_peak_bytes=80),)
    summary = summarize_experiment_records(records)
    variant_summaries = summarize_experiment_records_by_variant(records)
    base = tmp_path / "nested" / "artifacts"

    write_experiment_json(records, base / "records.json")
    write_experiment_csv(records, base / "records.csv")
    write_experiment_summary_json(summary, base / "summary.json")
    write_experiment_variant_summary_json(variant_summaries, base / "variant_summary.json")
    write_experiment_hint_ablation_json(records, base / "hint_ablation.json")
    write_experiment_baseline_comparison_json(records, base / "baseline_comparison.json")
    write_experiment_layered_accuracy_json(records, base / "layered_accuracy.json")

    assert (base / "records.json").exists()
    assert (base / "records.csv").exists()
    assert (base / "summary.json").exists()
    assert (base / "variant_summary.json").exists()
    assert (base / "hint_ablation.json").exists()
    assert (base / "baseline_comparison.json").exists()
    assert (base / "layered_accuracy.json").exists()


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
    variant_summary_path = tmp_path / "variant_summary.json"
    hint_ablation_path = tmp_path / "hint_ablation.json"
    baseline_comparison_path = tmp_path / "baseline_comparison.json"
    layered_accuracy_path = tmp_path / "layered_accuracy.json"
    sac_baseline_path = tmp_path / "sac_baseline.json"
    sac_baseline_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_name": "tiny_mlp_w8_d3",
                        "microbatch_size": 1,
                        "device": "cpu",
                        "status": "ok",
                        "baseline_id": "pytorch_sac_prefer_recompute",
                        "performance_result_usable": True,
                        "correctness_passed": True,
                        "sac_overall_peak_bytes": 1 << 20,
                        "sac_step_us": 100.0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
            "--device",
            "cpu",
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
            "--output-variant-summary-json",
            str(variant_summary_path),
            "--output-hint-ablation-json",
            str(hint_ablation_path),
            "--output-baseline-comparison-json",
            str(baseline_comparison_path),
            "--sac-baseline-json",
            str(sac_baseline_path),
            "--output-layered-accuracy-json",
            str(layered_accuracy_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    variant_summary_payload = json.loads(variant_summary_path.read_text(encoding="utf-8"))
    hint_ablation_payload = json.loads(hint_ablation_path.read_text(encoding="utf-8"))
    baseline_comparison_payload = json.loads(baseline_comparison_path.read_text(encoding="utf-8"))
    layered_accuracy_payload = json.loads(layered_accuracy_path.read_text(encoding="utf-8"))

    assert len(stdout_payload) == 2
    assert {record["variant_name"] for record in stdout_payload} == {
        "diagnostic_hints_on",
        "diagnostic_hints_off",
    }
    assert all(record["status"] == "ok" for record in stdout_payload)
    assert stdout_payload[0]["status"] == "ok"
    assert stdout_payload[0]["selected_plan_key"]
    assert stdout_payload[0]["config_fingerprint"]["device"] == "cpu"
    assert stdout_payload[0]["config_fingerprint"]["measurement_repeats"] == 2
    assert stdout_payload[0]["exact_plan_key"] is None
    assert stdout_payload[0]["exact_error_type"] == "PlanValidationError"
    assert stdout_payload[0]["graph_key"]
    assert stdout_payload[0]["selected_saved_value_ids"]
    assert stdout_payload[0]["selected_effective_saved_value_ids"]
    assert stdout_payload[0]["measured_plan_results"]
    assert stdout_payload[0]["diagnostic_counterfactuals"]
    assert stdout_payload[0]["selected_prediction_error_bytes"] is not None
    assert stdout_payload[0]["selected_feasibility_prediction_match"] is not None
    assert stdout_payload[0]["selected_estimated_peak_reduction_bytes"] is not None
    assert stdout_payload[0]["selected_measured_peak_reduction_vs_all_save_bytes"] is not None
    assert stdout_payload[0]["optimization_total_us"] is not None
    assert stdout_payload[0]["actual_joint_capture_count"] == 1
    assert "selected_peak_phase_match" in stdout_payload[0]
    assert all(record["measurement_repeats"] == 2 for record in stdout_payload)
    assert stdout_payload[0]["simulation_accuracy_candidate_count"] >= 1
    assert {record["diagnostic_hints_enabled"] for record in stdout_payload} == {True, False}
    assert stdout_payload[0]["cache_total_hits"] == 0
    assert "cache_layer_hits" in stdout_payload[0]
    assert "cache_layer_misses" in stdout_payload[0]
    assert file_payload[0]["task_name"] == "tiny_mlp_w8_d3"
    assert file_payload[0]["exact_error_type"] == "PlanValidationError"
    assert summary_payload["total_records"] == 2
    assert summary_payload["ok_records"] == 2
    assert "python_version" in summary_payload["environment_fingerprint"]
    assert summary_payload["variant_counts"] == {"diagnostic_hints_off": 1, "diagnostic_hints_on": 1}
    assert set(variant_summary_payload) == {"diagnostic_hints_off", "diagnostic_hints_on"}
    assert variant_summary_payload["diagnostic_hints_on"]["ok_records"] == 1
    assert variant_summary_payload["diagnostic_hints_off"]["ok_records"] == 1
    assert hint_ablation_payload["pair_count"] == 1
    assert "all_save" in baseline_comparison_payload["baseline_groups"]
    assert "pytorch_sac" in baseline_comparison_payload["baseline_groups"]
    assert baseline_comparison_payload["external_sac_rows_usable"] == 1
    assert baseline_comparison_payload["external_sac_keys_matched"] == 1
    assert baseline_comparison_payload["row_count"] >= 2
    assert "D5" in layered_accuracy_payload["level_summaries"]
    assert layered_accuracy_payload["row_count"] >= 1
    assert hint_ablation_payload["rows"][0]["conclusion"] in {
        "improved_budget",
        "improved_search",
        "improved_success",
        "improved_throughput",
        "neutral",
        "regressed_budget",
        "regressed_search",
        "regressed_success",
        "regressed_throughput",
    }
    assert summary_payload["selected_prediction_count"] == 2
    assert summary_payload["mean_selected_samples_per_second_speedup_vs_all_save"] is not None
    assert "phase_classification_count" in summary_payload
    assert "feasible_classification_count" in summary_payload
    assert summary_payload["mean_simulation_accuracy_absolute_error_bytes"] is not None
    assert summary_payload["p50_simulation_accuracy_absolute_error_bytes"] is not None
    assert summary_payload["p90_simulation_accuracy_absolute_error_bytes"] is not None
    assert "diagnostic_hint_kind_counts" in summary_payload
    assert summary_payload["exact_failure_count"] == 2
    assert "mean_optimization_amortization_steps" in summary_payload
    assert set(summary_payload["cache_layer_hit_rates"]).issubset({"analysis", "executable"})
    assert summary_payload["total_actual_joint_capture_count"] == 2
    assert csv_path.read_text(encoding="utf-8").startswith(
        "variant_name,config_fingerprint,task_name,microbatch_size,budget_bytes"
    )
