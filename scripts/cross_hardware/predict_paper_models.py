"""Predict Ascend-visible peak for the paper's actual 4 models with PeakAware.

Runs locally (CPU) with the Ascend910B hardware config, simulation-only
(validation_top_k=0), generous budget so the selected plan is all-save (which
is what the Ascend eager measurement observes). Same registry definitions as
the paper: resnet50, vit_b_16, bert_base, gpt2 at microbatch 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # PeakAware/ 根（scripts/cross_hardware → 上两级）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet50", "vit_b_16", "bert_base", "gpt2"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hardware", type=str, default="Ascend910B")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.environ["PEAKAWARE_COSTMODEL_HARDWARE"] = f"{args.hardware},{args.hardware}"

    from peakaware import PeakAwareConfig, optimize_training
    from peakaware.models.registry import (
        build_bert_base_task,
        build_gpt2_task,
        build_resnet50_task,
        build_vit_b16_task,
    )

    task = {
        "resnet50": build_resnet50_task(),
        "vit_b_16": build_vit_b16_task(),
        "bert_base": build_bert_base_task(),
        "gpt2": build_gpt2_task(),
    }[args.model]

    model = task.build_model()
    args_tuple, kwargs = task.build_batch(args.batch)
    optimizer = task.build_optimizer(model)
    loss_fn = task.loss_fn

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    budget_mib = max(1024, int(param_bytes * 100 / (1 << 20)) + 256)

    config = PeakAwareConfig(
        validation_top_k=0,
        capture_backend="auto",
        enable_compile=False,
        enable_inductor=False,
        safety_margin_bytes=0,
        safety_margin_ratio=0.0,
        search_algorithm="pareto_beam",
        selection_objective="min_time_then_peak",
        max_beam_candidates=256,
        beam_width=32,
        beam_candidate_overflow_policy="coarsen_tail",
    )
    result = optimize_training(
        model, args_tuple,
        loss_fn=loss_fn, optimizer=optimizer,
        memory_budget_bytes=budget_mib << 20,
        config=config,
    )
    selected = result.selected_plan
    all_save_entry = next(
        (
            entry
            for entry in (result.analysis.baseline_results if result.analysis else ())
            if getattr(entry.plan, "plan_id", "").startswith("beam_all_save")
        ),
        None,
    )
    all_save_peak = (
        int(all_save_entry.plan.estimated_peak_bytes) if all_save_entry is not None else None
    )
    all_save_time = (
        round(all_save_entry.plan.estimated_step_us, 2) if all_save_entry is not None else None
    )
    composition = None
    if all_save_entry is not None:
        snap = all_save_entry.simulation.peak_snapshot
        composition = {
            "phase": snap.phase,
            "parameter_MB": round(snap.parameter_bytes / 1e6, 3),
            "gradient_MB": round(snap.gradient_bytes / 1e6, 3),
            "optimizer_MB": round(snap.optimizer_bytes / 1e6, 3),
            "saved_activation_MB": round(snap.saved_activation_bytes / 1e6, 3),
            "workspace_MB": round(snap.workspace_bytes / 1e6, 3),
        }
    payload = {
        "model": args.model,
        "display_name": task.workload.display_name if task.workload else args.model,
        "hardware": args.hardware,
        "batch": args.batch,
        "budget_mib": budget_mib,
        "selected_plan_id": selected.plan_id,
        "selected_peak_MB": round(selected.estimated_peak_bytes / 1e6, 3),
        "all_save_peak_bytes": all_save_peak,
        "all_save_peak_MB": round(all_save_peak / 1e6, 3) if all_save_peak is not None else None,
        "all_save_step_us": all_save_time,
        "composition": composition,
        "param_MB": round(param_bytes / 1e6, 3),
        "candidate_count": result.optimization_metrics.get("candidate_count"),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
