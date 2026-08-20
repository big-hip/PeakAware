from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_timeline_readiness import audit_records
from scripts.analyze_simulation_accuracy import analyze_simulation_accuracy
from scripts.evaluate_main_experiment_status import evaluate_main_experiment_status
from scripts.summarize_failure_taxonomy import summarize_failure_taxonomy
from scripts.summarize_selected_regret import summarize_selected_regret


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pure post-processing bundle for the PeakAware main experiment: selected regret, "
            "failure taxonomy, timeline/GPU readiness, main status, and an optional timeline figure."
        )
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--require-gpu-util", action="store_true")
    parser.add_argument("--allow-phase-anchor-fallback", action="store_true")
    parser.add_argument("--phase-anchor-estimate", choices=("raw", "calibrated"), default="calibrated")
    parser.add_argument("--skip-timeline-figure", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records_json)
    derived_dir = args.output_root / "derived"
    figures_dir = args.output_root / "figures"
    derived_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_regret = summarize_selected_regret(records)
    failure_taxonomy = summarize_failure_taxonomy(records)
    simulation_accuracy = analyze_simulation_accuracy(records)
    timeline_readiness = audit_records(records, require_gpu_util=args.require_gpu_util)
    main_status = evaluate_main_experiment_status(
        records,
        selected_regret=selected_regret,
        failure_taxonomy=failure_taxonomy,
        require_gpu_util=args.require_gpu_util,
    )
    main_status["records_json"] = str(args.records_json)
    main_status["selected_regret_json"] = str(derived_dir / "selected_regret.json")
    main_status["failure_taxonomy_json"] = str(derived_dir / "failure_taxonomy.json")

    _write_json(derived_dir / "selected_regret.json", selected_regret)
    _write_json(derived_dir / "failure_taxonomy.json", failure_taxonomy)
    _write_json(derived_dir / "simulation_accuracy.json", simulation_accuracy)
    _write_json(derived_dir / "timeline_readiness_audit.json", timeline_readiness)
    _write_json(derived_dir / "main_experiment_status.json", main_status)

    timeline_manifest: dict[str, Any] | None = None
    timeline_error: str | None = None
    if not args.skip_timeline_figure:
        timeline_output_dir = figures_dir / "F_actual_vs_simulated_memory_timeline"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_timeline_artifact.py"),
            "--records-json",
            str(args.records_json),
            "--output-dir",
            str(timeline_output_dir),
            "--record-index",
            str(args.record_index),
            "--x-axis",
            "time",
            "--time-align",
            "matched-total",
        ]
        if args.task_name is not None:
            command.extend(("--task-name", args.task_name))
        if args.variant_name is not None:
            command.extend(("--variant-name", args.variant_name))
        if args.plan_id is not None:
            command.extend(("--plan-id", args.plan_id))
        if args.require_gpu_util:
            command.append("--require-gpu-util")
        if args.allow_phase_anchor_fallback:
            command.extend(("--allow-phase-anchor-fallback", "--phase-anchor-estimate", args.phase_anchor_estimate))
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            manifest_path = timeline_output_dir / "timeline_artifact_manifest.json"
            timeline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            timeline_error = (completed.stderr or completed.stdout or "").strip()

    manifest = {
        "schema_version": "0.1",
        "records_json": str(args.records_json),
        "output_root": str(args.output_root),
        "derived_files": sorted(path.name for path in derived_dir.iterdir() if path.is_file()),
        "figure_dirs": sorted(path.name for path in figures_dir.iterdir() if path.is_dir()),
        "main_question_status": main_status["main_question_status"],
        "blockers": main_status["blockers"],
        "timeline_ready": timeline_readiness["timeline_ready"],
        "gpu_ready": timeline_readiness["gpu_ready"],
        "simulation_accuracy": {
            "candidate_count": simulation_accuracy["candidate_count"],
            "raw_memory_mape": simulation_accuracy["memory"]["raw"]["mape"],
            "calibrated_memory_mape": simulation_accuracy["memory"]["calibrated"]["mape"],
            "raw_time_mape": simulation_accuracy["time"]["raw_costmodel"]["mape"],
            "calibrated_time_mape": simulation_accuracy["time"]["all_save_scale_calibrated"]["mape"],
            "raw_memory_unsafe_false_positive_count": simulation_accuracy["memory"][
                "raw_unsafe_false_positive_count"
            ],
            "calibrated_memory_unsafe_false_positive_count": simulation_accuracy["memory"][
                "calibrated_unsafe_false_positive_count"
            ],
        },
        "timeline_manifest": timeline_manifest,
        "timeline_error": timeline_error,
        "timeline_note": (
            "continuous_event_timeline requires selected actual sampled CUDA memory trace and simulated liveness/event "
            "trace. phase_anchor_fallback is a labelled transitional figure and must not be reported as a continuous "
            "actual-vs-simulated execution trace."
        ),
        "gpu_note": (
            "L0 GPU utilization is accepted only when a sampled GPU trace exists or a summary reports status=ok with "
            "sample_count>0. Nsight/torch.profiler traces should be added separately for per-op compute attribution."
        ),
    }
    manifest_path = args.output_root / "main_experiment_artifact_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "main_question_status": manifest["main_question_status"],
                "blockers": manifest["blockers"],
                "timeline_ready": manifest["timeline_ready"],
                "gpu_ready": manifest["gpu_ready"],
                "timeline_error": timeline_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
