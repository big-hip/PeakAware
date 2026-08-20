from __future__ import annotations

import pytest
import torch

from peakaware import PeakAwareConfig, optimize_training
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import summarize_result
from peakaware.search_efficiency import (
    analyze_search_efficiency,
    compare_actual_measurement_runs,
    replay_record,
)


def _candidate(
    plan_id: str,
    *,
    estimated_peak: int,
    estimated_step: float,
    measured_peak: int,
    measured_step: float,
    fw_peak: int,
    elapsed_us: float,
    status: str = "ok",
) -> dict:
    return {
        "plan_id": plan_id,
        "estimated_peak_bytes": estimated_peak,
        "estimated_step_us": estimated_step,
        "estimated_feasible": estimated_peak <= 100,
        "measured_peak_bytes": measured_peak,
        "measured_step_us": measured_step,
        "correctness_passed": True,
        "status": status,
        "phase_metrics": {
            "fw_peak_bytes": fw_peak,
            "overall_peak_bytes": measured_peak,
            "candidate_validation_elapsed_us": elapsed_us,
            "measurement_repeats": 5,
            "measurement_warmup_steps": 1,
        },
    }


def _record() -> dict:
    return {
        "status": "ok",
        "task_name": "toy",
        "variant_name": "diagnostic_hints_on",
        "microbatch_size": 2,
        "budget_bytes": 100,
        "matrix_pass_index": 0,
        "optimization_total_us": 1_500.0,
        "optimization_candidate_validation_measurement_us": 1_000.0,
        "optimization_capture_us": 200.0,
        "optimization_ir_build_us": 50.0,
        "optimization_executor_build_us": 10.0,
        "optimization_analysis_us": 240.0,
        "measured_plan_results": [
            _candidate(
                "all_save",
                estimated_peak=90,
                estimated_step=10,
                measured_peak=90,
                measured_step=15,
                fw_peak=80,
                elapsed_us=100,
            ),
            _candidate(
                "fast_oracle",
                estimated_peak=95,
                estimated_step=30,
                measured_peak=85,
                measured_step=10,
                fw_peak=60,
                elapsed_us=200,
            ),
            _candidate(
                "fw_only",
                estimated_peak=92,
                estimated_step=20,
                measured_peak=90,
                measured_step=12,
                fw_peak=50,
                elapsed_us=300,
            ),
            _candidate(
                "oom_candidate",
                estimated_peak=120,
                estimated_step=5,
                measured_peak=130,
                measured_step=50,
                fw_peak=40,
                elapsed_us=400,
                status="cuda_oom",
            ),
        ],
    }


def test_replay_record_computes_hits_regret_cost_and_exclusions():
    row = replay_record(_record(), top_k=2)

    assert row is not None
    assert row["ranked_plan_ids"] == ["all_save", "fw_only", "fast_oracle", "oom_candidate"]
    assert row["topk_plan_ids"] == ["all_save", "fw_only"]
    assert row["full_candidate_count"] == 4
    assert row["peakaware_measured_candidate_count"] == 2
    assert row["best_time_hit"] is False
    assert row["selected_plan_id"] == "fw_only"
    assert row["time_regret_ratio"] == pytest.approx(0.2)
    assert row["peak_regret_ratio"] == pytest.approx(5 / 85)
    assert row["min_peak_gap_ratio"] == pytest.approx(5 / 85)
    assert row["full_measurement_gpu_hours"] == pytest.approx(1_000 / 3_600_000_000)
    assert row["peakaware_measurement_gpu_hours"] == pytest.approx(400 / 3_600_000_000)
    assert row["full_total_search_gpu_hours"] == pytest.approx(1_500 / 3_600_000_000)
    assert row["simulation_only_total_search_gpu_hours"] == pytest.approx(500 / 3_600_000_000)
    assert row["counterfactual_topk_total_search_gpu_hours"] == pytest.approx(900 / 3_600_000_000)
    assert row["simulation_only_end_to_end_speedup"] == pytest.approx(3.0)
    assert row["zero_validation_plan_selection_speedup"] == pytest.approx(3.0)
    assert row["counterfactual_topk_end_to_end_speedup"] == pytest.approx(1_500 / 900)
    assert row["counterfactual_topk_validation_speedup"] == pytest.approx(2.5)
    assert row["simulation_only_selected_plan_id"] == "all_save"
    assert row["simulation_only_actual_feasible"] is True
    assert row["simulation_only_time_regret_ratio"] == pytest.approx(0.5)
    assert row["random_topk_best_time_hit_probability"] == pytest.approx(0.5)
    assert row["random_topk_any_feasible_probability"] == pytest.approx(1.0)
    assert row["random_topk_expected_time_regret_ratio"] == pytest.approx(0.15)
    assert row["random_topk_measurement_gpu_hours"] == pytest.approx(500 / 3_600_000_000)
    assert row["random_topk_total_search_gpu_hours"] == pytest.approx(1_000 / 3_600_000_000)
    assert row["random_topk_end_to_end_speedup"] == pytest.approx(1.5)
    assert row["excluded_actual_oom_count"] == 1
    assert row["excluded_actual_budget_violation_count"] == 0
    assert row["excluded_fw_only_no_global_gain_count"] == 0


def test_analyze_search_efficiency_reports_topk_sweep_and_fw_false_gain():
    payload = analyze_search_efficiency(
        [_record()],
        selected_top_k=1,
        top_k_values=(1, 2, 3),
        variants=frozenset({"diagnostic_hints_on"}),
    )

    assert payload["aggregate"]["candidate_measurement_reduction_rate"] == pytest.approx(0.75)
    assert payload["aggregate"]["excluded_fw_only_no_global_gain_count"] == 1
    assert payload["aggregate"]["excluded_actual_oom_count"] == 1
    assert payload["aggregate"]["simulation_only_end_to_end_speedup"] == pytest.approx(3.0)
    assert payload["aggregate"]["zero_validation_plan_selection_speedup"] == pytest.approx(3.0)
    assert payload["aggregate"]["counterfactual_topk_end_to_end_speedup"] == pytest.approx(
        1_500 / 600
    )
    assert payload["aggregate"]["simulation_only_oracle_hit_rate"] == 0.0
    assert payload["aggregate"]["random_topk_best_time_hit_rate"] == pytest.approx(0.25)
    assert payload["aggregate"]["random_topk_feasible_rate"] == pytest.approx(0.75)
    assert payload["aggregate"]["random_topk_mean_time_regret_ratio"] == pytest.approx(0.7 / 3)
    assert payload["aggregate"]["random_topk_end_to_end_speedup"] == pytest.approx(2.0)
    assert payload["aggregate"]["mean_min_peak_gap_ratio"] == pytest.approx(5 / 85)
    assert payload["aggregate"]["p50_min_peak_gap_ratio"] == pytest.approx(5 / 85)
    assert payload["aggregate"]["p90_min_peak_gap_ratio"] == pytest.approx(5 / 85)
    grouped_zero = payload["grouped_zero_validation_selection"]
    assert grouped_zero["group_count"] == 1
    assert grouped_zero["actual_feasible_rate"] == pytest.approx(1.0)
    assert grouped_zero["oracle_hit_rate"] == pytest.approx(0.0)
    assert grouped_zero["mean_time_regret_ratio"] == pytest.approx(0.5)
    assert payload["by_k"]["3"]["best_time_hit_rate"] == 1.0
    assert payload["by_task"]["toy"]["record_count"] == 1


def test_random_topk_reports_zero_probability_when_no_candidate_is_feasible():
    record = _record()
    record["budget_bytes"] = 1

    row = replay_record(record, top_k=2)

    assert row is not None
    assert row["oracle_fastest_feasible_plan_id"] is None
    assert row["random_topk_any_feasible_probability"] == 0.0
    assert row["random_topk_best_time_hit_probability"] == 0.0
    assert row["random_topk_expected_time_regret_ratio"] is None


def test_optimize_training_can_measure_only_simulation_ranked_topk():
    torch.manual_seed(0)
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)

    result = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=4,
            validation_top_k=1,
            selection_objective="min_time_then_peak",
        ),
    )

    assert result.analysis is not None
    assert len(result.analysis.baseline_results) > 1
    assert len(result.measured_candidates) == 1
    assert len(result.candidate_attempts) == 1
    assert result.candidate_attempts[0]["status"] == "measured"
    assert result.candidate_attempts[0]["validation_elapsed_us"] > 0
    assert result.optimization_metrics["candidate_validation_count"] == 1


def test_optimize_training_simulation_only_realizes_without_candidate_benchmark():
    torch.manual_seed(0)
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)

    result = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=4,
            validation_top_k=0,
            selection_objective="min_time_then_peak",
        ),
    )

    assert result.executable.evidence_source == "simulated"
    assert result.executable.correctness_passed is None
    assert result.executable.measured_peak_bytes == result.selected_plan.estimated_peak_bytes
    assert result.executable.measured_step_us == result.selected_plan.estimated_step_us
    assert result.executable.phase_metrics["measurement_repeats"] == 0
    assert result.executable.phase_metrics["measurement_warmup_steps"] == 0
    assert result.measured_candidates == ()
    assert result.candidate_attempts == ()
    assert result.dry_run is None
    assert result.optimization_metrics["candidate_validation_count"] == 0
    assert result.optimization_metrics["candidate_validation_measurement_us"] == 0.0
    assert result.optimization_metrics["candidate_realization_count"] == 1
    assert result.optimization_metrics["candidate_realization_us"] > 0.0

    summary = summarize_result(result)
    assert summary["measured"]["evidence_source"] == "simulated"
    assert summary["measured"]["is_measured"] is False
    assert summary["measured"]["correctness_passed"] is None
    assert summary["measured"]["actual_memory_timeline"] == ()
    assert summary["topk_correction"]["selected"] is None

    step = result.executor.step(*args, **kwargs)
    assert step.optimizer_step_performed is True


def test_compiler_refinement_reranks_without_candidate_gpu_measurement():
    torch.manual_seed(0)
    task = TrainingTaskRegistry.with_defaults().get("tiny_residual_w8")
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    args, kwargs = task.build_batch(2)

    result = optimize_training(
        model,
        args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=1 << 28,
        config=PeakAwareConfig(
            safety_margin_bytes=0,
            safety_margin_ratio=0.0,
            top_k=4,
            validation_top_k=0,
            capture_backend="aot",
            compiler_refinement_top_k=2,
            selection_objective="min_time_then_peak",
        ),
    )

    metrics = result.optimization_metrics
    assert result.measured_candidates == ()
    assert result.candidate_attempts == ()
    assert metrics["candidate_validation_measurement_us"] == 0.0
    assert metrics["compiler_refinement_source"] == "lowered_fx_l2_liveness"
    assert metrics["compiler_refinement_requested_count"] == 2
    assert metrics["compiler_refinement_success_count"] == 2
    assert metrics["compiler_refinement_candidate_gpu_measurements_used"] == 0
    assert result.executable.phase_metrics["candidate_gpu_measurements_used"] == 0
    assert result.executable.phase_metrics["compiler_refinement_source"] == (
        "lowered_fx_l2_liveness"
    )
    assert result.executable.phase_metrics[
        "lowered_fx_l2_simulated_memory_event_trace"
    ]


def test_compare_actual_measurement_runs_uses_observed_paired_costs():
    exhaustive = _record()
    exhaustive["optimization_candidate_validation_measurement_us"] = 1_000.0
    exhaustive["candidate_attempts"] = tuple(
        {"plan_id": candidate["plan_id"]}
        for candidate in exhaustive["measured_plan_results"]
    )
    guided = {
        **exhaustive,
        "optimization_candidate_validation_measurement_us": 600.0,
        "candidate_attempts": tuple(
            {"plan_id": candidate["plan_id"]}
            for candidate in exhaustive["measured_plan_results"][:2]
        ),
        "measured_plan_results": exhaustive["measured_plan_results"][:2],
        "selected_plan_id": "all_save",
    }

    comparison = compare_actual_measurement_runs([exhaustive], [guided])

    assert comparison["paired_record_count"] == 1
    assert comparison["full_measurement_candidate_count"] == 4
    assert comparison["peakaware_measured_candidate_count"] == 2
    assert comparison["candidate_measurement_reduction_rate"] == pytest.approx(0.5)
    assert comparison["gpu_hours_reduction_rate"] == pytest.approx(0.4)
    assert comparison["candidate_validation_speedup"] == pytest.approx(1_000 / 600)
    assert comparison["total_search_speedup"] == pytest.approx(1.0)
    assert comparison["all_strategy_actual_run_speedup"] == pytest.approx(1.0)
    assert comparison["paired_total_search_speedup_p50"] == pytest.approx(1.0)
    assert comparison["full_candidate_attempt_count"] == 4
    assert comparison["peakaware_candidate_attempt_count"] == 2
    assert comparison["guided_candidate_coverage_rate"] == pytest.approx(1.0)
    assert comparison["guided_shortlist_oracle_coverage_rate"] == pytest.approx(1.0)
    assert comparison["guided_shortlist_replay_mean_time_regret_ratio"] == pytest.approx(0.0)
    assert comparison["grouped_shortlist_replay"]["oracle_coverage_rate"] == pytest.approx(1.0)
    assert comparison["grouped_shortlist_replay"]["mean_time_regret_ratio"] == pytest.approx(0.0)
    assert comparison["by_task"]["toy"]["all_strategy_actual_run_speedup"] == pytest.approx(1.0)
    assert comparison["exhaustive_failed_record_count"] == 0
    assert comparison["guided_unmatched_ok_record_count"] == 0


def test_structural_diverse_actual_comparison_does_not_report_ranked_membership_mismatch():
    exhaustive = _record()
    exhaustive["candidate_attempts"] = tuple(
        {"plan_id": candidate["plan_id"]}
        for candidate in exhaustive["measured_plan_results"]
    )
    guided = {
        **exhaustive,
        "candidate_attempts": (
            {"plan_id": "all_save"},
            {"plan_id": "fast_oracle"},
        ),
        "measured_plan_results": (
            exhaustive["measured_plan_results"][0],
            exhaustive["measured_plan_results"][1],
        ),
        "selected_plan_id": "fast_oracle",
    }

    comparison = compare_actual_measurement_runs(
        [exhaustive],
        [guided],
        guided_selection_policy="structural_diverse",
    )

    assert comparison["guided_selection_policy"] == "structural_diverse"
    assert comparison["ranked_topk_membership_match_rate"] is None
    assert comparison["topk_membership_match_rate"] is None
    assert comparison["guided_candidate_coverage_rate"] == pytest.approx(1.0)
    assert comparison["guided_shortlist_oracle_coverage_rate"] == pytest.approx(1.0)
    assert comparison["grouped_shortlist_replay"]["rows"][0]["replay_selected_plan_id"] == "fast_oracle"


def test_simulation_only_actual_comparison_counts_zero_benchmarks_and_ranked_selection():
    exhaustive = _record()
    exhaustive["candidate_attempts"] = tuple(
        {"plan_id": candidate["plan_id"]}
        for candidate in exhaustive["measured_plan_results"]
    )
    guided = {
        **exhaustive,
        "optimization_total_us": 500.0,
        "optimization_candidate_validation_measurement_us": 0.0,
        "candidate_attempts": (),
        "measured_plan_results": (),
        "selected_plan_id": "all_save",
    }

    comparison = compare_actual_measurement_runs(
        [exhaustive],
        [guided],
        guided_selection_policy="simulation_only_ranked",
    )

    assert comparison["peakaware_candidate_attempt_count"] == 0
    assert comparison["peakaware_measured_candidate_count"] == 0
    assert comparison["all_strategy_actual_run_speedup"] == pytest.approx(3.0)
    assert comparison["guided_selection_mean_time_regret_ratio"] == pytest.approx(0.5)
    assert comparison["ranked_topk_membership_match_rate"] == pytest.approx(1.0)
    assert comparison["guided_candidate_coverage_rate"] == pytest.approx(1.0)
    assert comparison["grouped_final_selection"]["guided_selected_oracle_rate"] == 0.0
    assert comparison["grouped_final_selection"]["mean_time_regret_ratio"] == pytest.approx(0.5)
    assert comparison["guided_shortlist_oracle_coverage_rate"] == pytest.approx(0.0)
    assert comparison["guided_shortlist_replay_mean_time_regret_ratio"] == pytest.approx(0.5)
    assert comparison["grouped_shortlist_replay"]["oracle_coverage_rate"] == pytest.approx(0.0)
    assert comparison["grouped_shortlist_replay"]["mean_time_regret_ratio"] == pytest.approx(0.5)
