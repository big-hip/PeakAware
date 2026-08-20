from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils import _pytree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware import PeakAwareConfig, optimize_training
from peakaware.memory.fx_timeline import summarize_lowered_fx_l2_event_trace
from peakaware.models import TrainingTaskRegistry


def _parse_csv(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _move_to_device(value: Any, device: torch.device) -> Any:
    return _pytree.tree_map(
        lambda item: item.to(device) if isinstance(item, torch.Tensor) else item,
        value,
    )


def _comparison_rows(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        task_name = str(record.get("task_name") or "")
        for row in record.get("measured_plan_results") or ():
            plan_id = str(row.get("plan_id") or "")
            if task_name and plan_id:
                result[(task_name, plan_id)] = row
    return result


def _phase_comparison(
    predicted: dict[str, Any],
    measured: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if measured is None:
        return None
    phase_metrics = measured.get("phase_metrics") or {}
    actual_by_phase = {
        "fw": measured.get("fw_peak_bytes", phase_metrics.get("fw_peak_bytes")),
        "after_fw": measured.get(
            "after_fw_allocated_bytes",
            phase_metrics.get("after_fw_allocated_bytes"),
        ),
        "bw": measured.get("bw_peak_bytes", phase_metrics.get("bw_peak_bytes")),
        "optimizer": measured.get(
            "optimizer_peak_bytes",
            phase_metrics.get("optimizer_peak_bytes"),
        ),
    }
    predicted_by_phase = predicted["phase_peak_bytes"]
    rows = {}
    for phase, actual in actual_by_phase.items():
        if actual is None or int(actual) <= 0:
            continue
        value = int(predicted_by_phase[phase])
        rows[phase] = {
            "predicted_bytes": value,
            "measured_bytes": int(actual),
            "signed_error_ratio": (value - int(actual)) / int(actual),
            "absolute_percentage_error": abs(value - int(actual)) / int(actual),
        }
    measured_peak = measured.get("measured_peak_bytes")
    if measured_peak is None or int(measured_peak) <= 0:
        return {"phase_rows": rows}
    predicted_peak = int(predicted["estimated_peak_bytes"])
    return {
        "phase_rows": rows,
        "overall": {
            "predicted_bytes": predicted_peak,
            "measured_bytes": int(measured_peak),
            "signed_error_ratio": (predicted_peak - int(measured_peak)) / int(measured_peak),
            "absolute_percentage_error": abs(predicted_peak - int(measured_peak)) / int(measured_peak),
        },
        "comparison_role": "post_prediction_audit_only",
        "used_for_prediction_or_ranking": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit lowered-FX L2 planner refinement without executing or profiling "
            "any candidate training step."
        )
    )
    parser.add_argument(
        "--tasks",
        default="bert_base_full_s128,gpt2_small_full_s128",
    )
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--budget-mib", type=int, default=45000)
    parser.add_argument("--compiler-refinement-top-k", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--comparison-records", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.microbatch <= 0 or args.budget_mib <= 0:
        raise ValueError("microbatch and budget must be positive")
    if args.compiler_refinement_top_k <= 0:
        raise ValueError("compiler refinement top-k must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable")
    comparisons = _comparison_rows(args.comparison_records)
    registry = TrainingTaskRegistry.with_defaults()
    output_rows: list[dict[str, Any]] = []

    for task_name in _parse_csv(args.tasks):
        task = registry.get(task_name)
        model = task.build_model().to(device)
        optimizer = task.build_optimizer(model)
        example_args, example_kwargs = task.build_batch(args.microbatch)
        example_args = _move_to_device(example_args, device)
        example_kwargs = _move_to_device(example_kwargs, device)
        result = optimize_training(
            model,
            example_args,
            example_kwargs=example_kwargs,
            loss_fn=task.loss_fn,
            optimizer=optimizer,
            memory_budget_bytes=args.budget_mib << 20,
            config=PeakAwareConfig(
                safety_margin_bytes=0,
                safety_margin_ratio=0.0,
                top_k=12,
                max_greedy_candidates=12,
                validation_top_k=0,
                capture_backend="aot",
                compiler_refinement_top_k=args.compiler_refinement_top_k,
                selection_objective="min_time_then_peak",
                profile_db_path=args.profile_db,
                enable_compile=args.enable_compile,
            ),
        )
        plan_rows = []
        for candidate in result.analysis.baseline_results:
            trace = candidate.simulation.simulated_memory_event_trace
            refinement = candidate.simulation.cost_breakdown.get("compiler_refinement")
            if not trace or refinement is None:
                continue
            summary = summarize_lowered_fx_l2_event_trace(trace)
            plan_rows.append(
                {
                    "plan_id": candidate.plan.plan_id,
                    "coarse_estimated_peak_bytes": int(
                        refinement["coarse_estimated_peak_bytes"]
                    ),
                    "coarse_estimated_step_us": float(
                        refinement["coarse_estimated_step_us"]
                    ),
                    "refined_estimated_peak_bytes": int(
                        summary["estimated_peak_bytes"]
                    ),
                    "refined_estimated_step_us": float(
                        refinement["refined_estimated_step_us"]
                    ),
                    "refined_peak_phase": summary["peak_phase"],
                    "refined_peak_row": {
                        key: summary["peak_row"].get(key)
                        for key in (
                            "event",
                            "op_name",
                            "target",
                            "bytes",
                            "fixed_bytes",
                            "payload_bytes",
                            "workspace_bytes",
                            "memory_model_kind",
                            "source",
                        )
                    },
                    "refined_phase_peak_bytes": summary["phase_peak_bytes"],
                    "refined_phase_peak_rows": {
                        phase: None
                        if row is None
                        else {
                            key: row.get(key)
                            for key in (
                                "event",
                                "op_name",
                                "target",
                                "bytes",
                                "fixed_bytes",
                                "payload_bytes",
                                "workspace_bytes",
                                "memory_model_kind",
                                "source",
                            )
                        }
                        for phase, row in summary["phase_peak_rows"].items()
                    },
                    "refined_after_fw_retained_bytes": int(
                        summary["after_fw_retained_bytes"]
                    ),
                    "lowered_structure": refinement.get("lowered_structure"),
                    "outer_aot_capture_failure": refinement.get(
                        "outer_aot_capture_failure"
                    ),
                    "comparison": _phase_comparison(
                        summary,
                        comparisons.get((task_name, candidate.plan.plan_id)),
                    ),
                }
            )
        output_rows.append(
            {
                "task_name": task_name,
                "microbatch_size": args.microbatch,
                "budget_bytes": args.budget_mib << 20,
                "selected_plan_id": result.selected_plan.plan_id,
                "candidate_gpu_measurements_used": 0,
                "selection_mode": "simulation_only_lowered_fx_l2_refinement",
                "compiler_refinement_source": result.optimization_metrics[
                    "compiler_refinement_source"
                ],
                "compiler_refinement_requested_count": result.optimization_metrics[
                    "compiler_refinement_requested_count"
                ],
                "compiler_refinement_success_count": result.optimization_metrics[
                    "compiler_refinement_success_count"
                ],
                "compiler_refinement_failure_count": result.optimization_metrics[
                    "compiler_refinement_failure_count"
                ],
                "capture_us": result.optimization_metrics["capture_us"],
                "coarse_analysis_us": result.optimization_metrics["analysis_us"],
                "compiler_refinement_us": result.optimization_metrics[
                    "compiler_refinement_us"
                ],
                "total_optimization_us": result.optimization_metrics[
                    "total_optimization_us"
                ],
                "plans": plan_rows,
            }
        )
        del result, model, optimizer, example_args, example_kwargs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": "lowered-fx-l2-refinement-audit-v1",
        "prediction_mode": "uncalibrated_no_candidate_gpu_execution",
        "comparison_records_used_for_prediction_or_ranking": False,
        "tasks": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
