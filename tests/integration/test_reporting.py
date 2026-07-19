import json

import torch

from peakaware import PeakAwareConfig, optimize_training
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import export_result_json, render_text_report, summarize_result


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
    text = render_text_report(result)
    path = tmp_path / "report.json"
    exported = export_result_json(result, path)

    assert summary["selected_plan_id"] == result.selected_plan.plan_id
    assert summary["graph_key"] == result.selected_plan.graph_key
    assert summary["selected_saved_value_ids"] == tuple(sorted(result.selected_plan.saved_value_ids))
    assert summary["estimated_peak_bytes"] == result.selected_plan.estimated_peak_bytes
    assert "reserved_peak_bytes" in summary["measured"]
    assert summary["diagnostic"]["counterfactuals"][-1]["level"] == "D5"
    assert summary["measured_candidates"]
    assert summary["topk_correction"]["selected"]["plan_id"] == result.selected_plan.plan_id
    assert "error_bytes" in summary["measured_candidates"][0]["prediction_error"]
    assert summary["cache"]["total_hits"] == result.cache_stats.total_hits
    assert "Selected plan:" in text
    assert json.loads(exported)["selected_plan_id"] == result.selected_plan.plan_id
    assert json.loads(path.read_text(encoding="utf-8"))["measured"]["correctness_passed"] is True
