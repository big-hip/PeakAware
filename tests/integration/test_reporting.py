import json

import torch

from peakaware import PeakAwareConfig, optimize_training
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import (
    export_plan_artifact_json,
    export_result_json,
    render_text_report,
    summarize_plan_artifact,
    summarize_result,
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

    assert summary["selected_plan_id"] == result.selected_plan.plan_id
    assert summary["selected_plan_key"]
    assert summary["graph_key"] == result.selected_plan.graph_key
    assert summary["selected_saved_value_ids"] == tuple(sorted(result.selected_plan.saved_value_ids))
    assert set(summary["selected_saved_value_ids"]).issubset(summary["selected_effective_saved_value_ids"])
    assert summary["estimated_peak_bytes"] == result.selected_plan.estimated_peak_bytes
    assert "reserved_peak_bytes" in summary["measured"]
    assert "early_stop" in summary
    diagnostic_by_plan = {item["plan_id"]: item for item in summary["plan_diagnostics"]}
    assert {"all_save", "torch_min_cut", "block_checkpoint"}.issubset(diagnostic_by_plan)
    assert all("expected_saved_reduction" in item for item in diagnostic_by_plan.values())
    assert all("repair_hints" in item for item in diagnostic_by_plan.values())
    assert all(item["counterfactuals"] for item in diagnostic_by_plan.values())
    assert summary["diagnostic"]["counterfactuals"][-1]["level"] == "D5"
    assert "repair_hints" in summary["diagnostic"]
    assert summary["measured_candidates"]
    assert summary["topk_correction"]["selected"]["plan_id"] == result.selected_plan.plan_id
    assert "error_bytes" in summary["measured_candidates"][0]["prediction_error"]
    assert summary["cache"]["total_hits"] == result.cache_stats.total_hits
    assert plan_artifact["plan_id"] == result.selected_plan.plan_id
    assert plan_artifact["plan_key"] == summary["selected_plan_key"]
    assert plan_artifact["saved_value_ids"] == tuple(sorted(result.selected_plan.saved_value_ids))
    assert plan_artifact["effective_saved_value_ids"] == summary["selected_effective_saved_value_ids"]
    assert plan_artifact["correctness"]["gradients_match"] is True
    assert "Selected plan:" in text
    assert json.loads(exported)["selected_plan_id"] == result.selected_plan.plan_id
    assert json.loads(exported)["selected_plan_key"] == summary["selected_plan_key"]
    assert json.loads(exported_plan)["graph_key"] == result.selected_plan.graph_key
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_id"] == result.selected_plan.plan_id
    assert json.loads(path.read_text(encoding="utf-8"))["measured"]["correctness_passed"] is True
