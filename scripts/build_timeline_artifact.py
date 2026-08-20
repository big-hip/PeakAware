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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an auditable actual-vs-simulated memory timeline artifact from PeakAware records. "
            "When continuous traces are missing, optionally emits a clearly labelled phase-anchor fallback."
        )
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--x-axis", choices=("phase", "time"), default="time")
    parser.add_argument("--time-align", choices=("matched-total", "raw-costmodel"), default="matched-total")
    parser.add_argument("--allow-phase-anchor-fallback", action="store_true")
    parser.add_argument("--phase-anchor-estimate", choices=("raw", "calibrated"), default="calibrated")
    parser.add_argument("--require-gpu-util", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records_json)
    candidates = _matching_records(
        records,
        task_name=args.task_name,
        variant_name=args.variant_name,
        plan_id=args.plan_id,
    )
    if not candidates:
        raise SystemExit("no matching successful records")
    if args.record_index < 0 or args.record_index >= len(candidates):
        raise SystemExit(f"record-index out of range: {args.record_index} for {len(candidates)} matching records")

    selected_record = candidates[args.record_index]
    selected_audit = audit_records([selected_record], require_gpu_util=args.require_gpu_util)
    all_audit = audit_records(records, require_gpu_util=args.require_gpu_util)
    mode = "continuous_event_timeline" if _event_timeline_ready(selected_record) else "phase_anchor_fallback"
    if mode == "phase_anchor_fallback" and not args.allow_phase_anchor_fallback:
        raise SystemExit(
            "selected record is missing continuous sampled/event timeline fields; rerun experiments or pass "
            "--allow-phase-anchor-fallback for a labelled transitional figure"
        )
    if args.require_gpu_util and not selected_audit["gpu_ready"]:
        raise SystemExit("selected record is missing GPU utilization trace fields")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "timeline_readiness_audit.json"
    audit_path.write_text(json.dumps(all_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figure_dir = args.output_dir / (
        "F_actual_vs_simulated_event_timeline"
        if mode == "continuous_event_timeline"
        else "F_actual_vs_simulated_phase_anchor_fallback"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_memory_timeline_fit.py"),
        "--records-json",
        str(args.records_json),
        "--output-dir",
        str(figure_dir),
        "--record-index",
        str(args.record_index),
        "--x-axis",
        args.x_axis,
        "--time-align",
        args.time_align,
    ]
    if args.task_name is not None:
        command.extend(("--task-name", args.task_name))
    if args.variant_name is not None:
        command.extend(("--variant-name", args.variant_name))
    if args.plan_id is not None:
        command.extend(("--plan-id", args.plan_id))
    if mode == "phase_anchor_fallback":
        command.extend(("--phase-anchor-fallback", "--phase-anchor-estimate", args.phase_anchor_estimate))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    manifest = {
        "records_json": str(args.records_json),
        "output_dir": str(args.output_dir),
        "mode": mode,
        "selected_record": _record_brief(selected_record),
        "selected_audit": selected_audit,
        "readiness_audit": str(audit_path),
        "figure_dir": str(figure_dir),
        "generated_files": sorted(path.name for path in figure_dir.iterdir() if path.is_file()),
        "command": command,
        "generator_stdout": completed.stdout,
    }
    manifest_path = args.output_dir / "timeline_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _matching_records(
    records: list[dict[str, Any]],
    *,
    task_name: str | None,
    variant_name: str | None,
    plan_id: str | None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("status") == "ok"
        and (task_name is None or record.get("task_name") == task_name)
        and (variant_name is None or record.get("variant_name") == variant_name)
        and (plan_id is None or record.get("selected_plan_id") == plan_id)
    ]


def _event_timeline_ready(record: dict[str, Any]) -> bool:
    has_actual_sampled = _has_points(record.get("selected_actual_overall_sampled_memory_trace")) or _has_points(
        record.get("selected_actual_sampled_memory_trace")
    )
    has_simulated_event = _has_points(record.get("selected_lowered_fx_l2_simulated_memory_event_trace")) or _has_points(
        record.get("selected_simulated_memory_event_trace")
    )
    if has_actual_sampled and has_simulated_event:
        return True
    selected_plan_id = record.get("selected_plan_id")
    for row in record.get("measured_plan_results", ()):
        if not isinstance(row, dict) or row.get("plan_id") != selected_plan_id:
            continue
        return has_actual_sampled and (
            _has_points(row.get("lowered_fx_l2_simulated_memory_event_trace"))
            or _has_points(row.get("simulated_memory_event_trace"))
        )
    return False


def _has_points(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0


def _record_brief(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": record.get("task_name"),
        "variant_name": record.get("variant_name"),
        "microbatch_size": record.get("microbatch_size"),
        "budget_bytes": record.get("budget_bytes"),
        "selected_plan_id": record.get("selected_plan_id"),
        "matrix_pass_index": record.get("matrix_pass_index"),
        "matrix_pass_count": record.get("matrix_pass_count"),
    }


if __name__ == "__main__":
    main()
