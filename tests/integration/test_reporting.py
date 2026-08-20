import json
import subprocess
import sys
from dataclasses import replace

import torch

from peakaware import PeakAwareConfig, optimize_training
from peakaware.contracts import FailureRecord
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import (
    export_plan_artifact_json,
    export_result_json,
    load_plan_artifact_json,
    render_text_report,
    summarize_plan_artifact,
    summarize_result,
    validate_plan_artifact_identity,
)


def test_reporting_summarizes_result_and_exports_json(tmp_path):
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
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0),
    )

    summary = summarize_result(result)
    plan_artifact = summarize_plan_artifact(result)
    text = render_text_report(result)
    path = tmp_path / "report.json"
    plan_path = tmp_path / "plan.json"
    exported = export_result_json(result, path)
    exported_plan = export_plan_artifact_json(result, plan_path)

    assert result.analysis is not None
    assert summary["selected_plan_id"] == result.selected_plan.plan_id
    assert summary["selected_plan_key"]
    assert summary["graph_key"] == result.selected_plan.graph_key
    assert summary["selected_saved_value_ids"] == tuple(sorted(result.selected_plan.saved_value_ids))
    assert set(summary["selected_saved_value_ids"]).issubset(summary["selected_effective_saved_value_ids"])
    assert summary["estimated_peak_bytes"] == result.selected_plan.estimated_peak_bytes
    assert summary["selection_objective"] == "min_peak_then_time"
    assert "peak_phase" in summary["measured"]
    assert "reserved_peak_bytes" in summary["measured"]
    assert "early_stop" in summary
    assert all(item["ranking_provenance"]["risk_score"]["range"] == "[0, 1]" for item in summary["plans"])
    assert all(item["ranking_provenance"]["confidence"]["direction"] == "higher_is_better" for item in summary["plans"])
    assert all("plan_id" in item["ranking_provenance"]["stable_tie_break"] for item in summary["plans"])
    assert all(item["peak_snapshot"]["phase"] in {"fw", "bw", "optimizer"} for item in summary["plans"])
    assert all("live_storage_ids" in item["peak_snapshot"] for item in summary["plans"])
    assert all("workspace_bytes" in item["peak_snapshot"] for item in summary["plans"])
    assert summary["capture_failures"] == []
    failed_result = replace(
        result,
        analysis=replace(
            result.analysis,
            capture_failures=(
                FailureRecord(
                    stage="capture_aot",
                    error_type="RuntimeError",
                    message="synthetic aot failure",
                    recovered=True,
                    next_fallback="fx",
                    applied_adapters=("aot_autograd", "default_partition"),
                    applied_plugins=(),
                ),
            ),
        ),
    )
    failed_summary = summarize_result(failed_result)
    diagnostic_by_plan = {item["plan_id"]: item for item in summary["plan_diagnostics"]}
    assert {"all_save", "torch_min_cut", "block_checkpoint"}.issubset(diagnostic_by_plan)
    assert all("expected_saved_reduction" in item for item in diagnostic_by_plan.values())
    assert all(item["expectation"]["strategy_status"] == "available" for item in diagnostic_by_plan.values())
    assert all("candidate" in item["expectation"]["strategy_provenance"] for item in diagnostic_by_plan.values())
    assert diagnostic_by_plan["torch_min_cut"]["expectation"]["strategy_expected_saved_reduction"] is not None
    assert all("normalized_saved_reduction" in item["expectation"] for item in diagnostic_by_plan.values())
    assert all("strategy_estimation_gap" in item["expectation"] for item in diagnostic_by_plan.values())
    assert all("realization_gap" in item["expectation"] for item in diagnostic_by_plan.values())
    assert all("repair_hints" in item for item in diagnostic_by_plan.values())
    assert all(0 <= item["confidence"] <= 1 for item in diagnostic_by_plan.values())
    assert all(item["evidence"] for item in diagnostic_by_plan.values())
    assert all(item["counterfactuals"] for item in diagnostic_by_plan.values())
    assert all("candidate_peak" in item["counterfactuals"][0] for item in diagnostic_by_plan.values())
    assert summary["search_diagnostics"]["diagnostic_hints_enabled"] is True
    assert "diagnostic_hint_count" in summary["search_diagnostics"]
    assert "feasible_after_repair_count" in summary["search_diagnostics"]
    assert summary["diagnostic"]["counterfactuals"][-1]["level"] == "D5"
    assert "baseline_peak" in summary["diagnostic"]["counterfactuals"][0]
    assert summary["diagnostic"]["expectation"]["strategy_status"] == "available"
    assert 0 <= summary["diagnostic"]["confidence"] <= 1
    assert "repair_hints" in summary["diagnostic"]
    assert summary["diagnostic"]["evidence"]
    assert {"evidence_id", "root_cause", "metric", "value", "threshold", "direction", "description"}.issubset(
        summary["diagnostic"]["evidence"][0]
    )
    assert summary["measured_candidates"]
    assert summary["measured"]["actual_memory_timeline"]
    assert summary["selected_simulated_memory_timeline"]
    assert summary["selected_memory_timeline_fit"]["point_count"] > 0
    assert "simulated_memory_timeline" in summary["plans"][0]
    assert "actual_memory_timeline" in summary["measured_candidates"][0]
    assert summary["topk_correction"]["selected"]["plan_id"] == result.selected_plan.plan_id
    assert summary["topk_correction"]["simulation_accuracy"]["candidate_count"] == len(
        summary["topk_correction"]["candidates"]
    )
    assert summary["topk_correction"]["simulation_accuracy"]["max_absolute_error_bytes"] >= 0
    assert summary["topk_correction"]["simulation_accuracy"]["p50_absolute_error_bytes"] is not None
    assert summary["topk_correction"]["simulation_accuracy"]["p90_absolute_error_bytes"] is not None
    assert summary["topk_correction"]["simulation_accuracy"]["p50_absolute_relative_error"] is not None
    assert summary["topk_correction"]["simulation_accuracy"]["p90_absolute_relative_error"] is not None
    assert summary["topk_correction"]["simulation_accuracy"]["within_10_percent_rate"] is not None
    assert "phase_classification_accuracy" in summary["topk_correction"]["simulation_accuracy"]
    assert "feasible_classification_accuracy" in summary["topk_correction"]["simulation_accuracy"]
    assert "error_bytes" in summary["measured_candidates"][0]["prediction_error"]
    assert "phase_match" in summary["measured_candidates"][0]["prediction_error"]
    assert "feasibility_match" in summary["measured_candidates"][0]["prediction_error"]
    assert summary["optimization_cost"]["total_optimization_us"] is not None
    assert summary["optimization_cost"]["actual_joint_capture_count"] == 1
    assert "amortization_steps" in summary["optimization_cost"]
    assert summary["cache"]["total_hits"] == result.cache_stats.total_hits
    assert plan_artifact["plan_id"] == result.selected_plan.plan_id
    assert plan_artifact["plan_key"] == summary["selected_plan_key"]
    assert plan_artifact["saved_value_ids"] == tuple(sorted(result.selected_plan.saved_value_ids))
    assert plan_artifact["effective_saved_value_ids"] == summary["selected_effective_saved_value_ids"]
    assert plan_artifact["ranking_provenance"]["risk_score"]["direction"] == "lower_is_better"
    assert plan_artifact["estimated_peak_bytes"] == summary["estimated_peak_bytes"]
    assert plan_artifact["peak_snapshot"]["live_bytes"] > 0
    assert "recomputed_bytes" in plan_artifact["peak_snapshot"]
    assert plan_artifact["correctness"]["gradients_match"] is True
    assert summary["dry_run"]["replay_mode"] in {"lowered_aot", "eager_baseline"}
    assert plan_artifact["correctness"]["replay_mode"] == summary["dry_run"]["replay_mode"]
    loaded_plan = load_plan_artifact_json(plan_path)
    validation = validate_plan_artifact_identity(loaded_plan)
    assert validation["valid"] is True
    assert validation["expected_plan_key"] == summary["selected_plan_key"]
    loaded_plan["plan_key"] = "bad-key"
    assert validate_plan_artifact_identity(loaded_plan)["valid"] is False
    loaded_plan["effective_saved_value_ids"] = ["not-an-int"]
    assert validate_plan_artifact_identity(loaded_plan)["errors"] == ("artifact fields have invalid types",)
    assert "Selected plan:" in text
    assert json.loads(exported)["selected_plan_id"] == result.selected_plan.plan_id
    assert json.loads(exported)["selected_plan_key"] == summary["selected_plan_key"]
    assert json.loads(exported_plan)["graph_key"] == result.selected_plan.graph_key
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_id"] == result.selected_plan.plan_id
    assert json.loads(path.read_text(encoding="utf-8"))["measured"]["correctness_passed"] is True
    assert len(failed_summary["capture_failures"]) == 1
    failure = failed_summary["capture_failures"][0]
    assert failure["stage"] == "capture_aot"
    assert failure["recovered"] is True
    assert failure["next_fallback"] == "fx"
    assert failure["applied_adapters"] == ("aot_autograd", "default_partition")
    assert failure["applied_plugins"] == ()


def test_validate_plan_artifacts_script_reports_identity_results(tmp_path):
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
        config=PeakAwareConfig(safety_margin_bytes=0, safety_margin_ratio=0.0, top_k=1),
    )
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "validation.json"
    export_plan_artifact_json(result, plan_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_plan_artifacts.py",
            str(plan_path),
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
    assert stdout_payload["artifact_count"] == 1
    assert stdout_payload["valid_count"] == 1
    assert stdout_payload["invalid_count"] == 0
