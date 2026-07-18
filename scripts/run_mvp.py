from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware import PeakAwareConfig, optimize_training
from peakaware.microbatch import optimize_microbatches
from peakaware.models import TrainingTaskRegistry
from peakaware.reporting import summarize_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PeakAware MVP pipeline.")
    parser.add_argument("--task", default="tiny_residual_w8")
    parser.add_argument("--budget-mib", type=int, default=256)
    parser.add_argument("--microbatches", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    registry = TrainingTaskRegistry.with_defaults()
    task = registry.get(args.task)
    budget = args.budget_mib << 20
    config = PeakAwareConfig(enable_compile=False, safety_margin_bytes=0, safety_margin_ratio=0.0)
    if args.microbatches:
        sizes = tuple(int(item) for item in args.microbatches.split(",") if item)
        result = optimize_microbatches(task, sizes, memory_budget_bytes=budget, config=config)
        payload = {
            "mode": "microbatch",
            "selected_microbatch": result.selected.microbatch_size,
            "candidates": [
                {
                    "microbatch": candidate.microbatch_size,
                    "samples_per_second": candidate.useful_samples_per_second,
                    "plan_id": candidate.result.selected_plan.plan_id,
                    "estimated_peak_bytes": candidate.result.selected_plan.estimated_peak_bytes,
                }
                for candidate in result.candidates
            ],
            "selected_report": summarize_result(result.selected.result),
        }
    else:
        model = task.build_model()
        optimizer = task.build_optimizer(model)
        batch_args, batch_kwargs = task.build_batch(4)
        result = optimize_training(
            model,
            batch_args,
            example_kwargs=batch_kwargs,
            loss_fn=task.loss_fn,
            optimizer=optimizer,
            memory_budget_bytes=budget,
            config=config,
        )
        payload = {
            "mode": "single",
            "plan_id": result.selected_plan.plan_id,
            "feasibility": result.feasibility.status,
            "estimated_peak_bytes": result.selected_plan.estimated_peak_bytes,
            "measured_peak_bytes": result.executable.measured_peak_bytes,
            "dry_run_passed": result.dry_run is not None and result.dry_run.gradients_match,
            "fallback_plan_ids": result.fallback_plan_ids,
            "report": summarize_result(result),
        }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.report_json is not None:
        args.report_json.write_text(text + "\n", encoding="utf-8")
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
