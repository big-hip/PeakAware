from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from peakaware.memory.timeline import fit_memory_timelines, merge_timeline_rows
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _timeline_path = ROOT / "peakaware" / "memory" / "timeline.py"
    _timeline_spec = importlib.util.spec_from_file_location("_peakaware_timeline_standalone", _timeline_path)
    if _timeline_spec is None or _timeline_spec.loader is None:
        raise
    _timeline_module = importlib.util.module_from_spec(_timeline_spec)
    _timeline_spec.loader.exec_module(_timeline_module)
    fit_memory_timelines = _timeline_module.fit_memory_timelines
    merge_timeline_rows = _timeline_module.merge_timeline_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export phase-normalized actual-vs-simulated memory timeline fitting artifacts."
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--x-axis", choices=("phase", "time"), default="time")
    parser.add_argument("--time-align", choices=("matched-total", "raw-costmodel"), default="matched-total")
    parser.add_argument("--phase-anchor-fallback", action="store_true")
    parser.add_argument("--phase-anchor-estimate", choices=("raw", "calibrated"), default="calibrated")
    parser.add_argument("--all-matching", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records_json)
    candidates = [
        record
        for record in records
        if record.get("status") == "ok"
        and (args.task_name is None or record.get("task_name") == args.task_name)
        and (args.variant_name is None or record.get("variant_name") == args.variant_name)
        and (args.plan_id is None or record.get("selected_plan_id") == args.plan_id)
    ]
    if not candidates:
        raise SystemExit("no matching successful records")
    if not args.all_matching and (args.record_index < 0 or args.record_index >= len(candidates)):
        raise SystemExit(f"record-index out of range: {args.record_index} for {len(candidates)} matching records")

    selected_records = candidates if args.all_matching else [candidates[args.record_index]]
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_summaries: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records):
        label = _record_label(record, args.record_index if not args.all_matching else index)
        output_dir = args.output_dir / label if args.all_matching else args.output_dir
        summary, rows = _write_record_artifacts(
            record,
            records_json=args.records_json,
            output_dir=output_dir,
            matching_record_count=len(candidates),
            selected_record_index=args.record_index if not args.all_matching else index,
            x_axis=args.x_axis,
            time_align=args.time_align,
            phase_anchor_fallback=args.phase_anchor_fallback,
            phase_anchor_estimate=args.phase_anchor_estimate,
        )
        aggregate_rows.extend(rows)
        aggregate_summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.all_matching:
        _write_csv(args.output_dir / "source.csv", aggregate_rows)
        (args.output_dir / "fit_summary.json").write_text(
            json.dumps({"records": aggregate_summaries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"record_count": len(aggregate_summaries)}, indent=2, sort_keys=True))
    else:
        print(json.dumps(aggregate_summaries[0]["fit"], indent=2, sort_keys=True))


def _write_record_artifacts(
    record: dict[str, Any],
    *,
    records_json: Path,
    output_dir: Path,
    matching_record_count: int,
    selected_record_index: int,
    x_axis: str,
    time_align: str,
    phase_anchor_fallback: bool,
    phase_anchor_estimate: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual = tuple(record.get("selected_actual_memory_timeline") or ())
    simulated = tuple(record.get("selected_simulated_memory_timeline") or ())
    sampled_rows = _sampled_trace_rows(record)
    simulated_event_rows = _simulated_event_rows(record, sampled_rows=sampled_rows) if sampled_rows else []
    if not actual or not simulated:
        if sampled_rows and simulated_event_rows:
            actual = ()
            simulated = ()
        elif not phase_anchor_fallback:
            raise SystemExit(
                "matching record does not include phase-normalized timeline fields or continuous sampled/event "
                "timeline fields; rerun experiments with the current code"
            )
        else:
            actual = _phase_anchor_actual_timeline(record)
            simulated = _phase_anchor_simulated_timeline(record, estimate_kind=phase_anchor_estimate)
            if not actual or not simulated:
                raise SystemExit(
                    "matching record does not include timeline fields and cannot be converted to phase-anchor timeline"
                )

    fit = fit_memory_timelines(actual, simulated)
    rows = merge_timeline_rows(
        record={
            **record,
            "actual_memory_trace": tuple(record.get("selected_actual_memory_trace") or ()),
            "time_align": time_align,
        },
        actual=actual,
        simulated=simulated,
        offset_fitted_simulated_bytes=float(fit.get("offset_fitted_simulated_bytes") or 0.0),
        x_axis=x_axis,
    )
    summary = {
        "records_json": str(records_json),
        "matching_record_count": matching_record_count,
        "selected_record_index": selected_record_index,
        "record": {
            "task_name": record.get("task_name"),
            "variant_name": record.get("variant_name"),
            "microbatch_size": record.get("microbatch_size"),
            "budget_bytes": record.get("budget_bytes"),
            "selected_plan_id": record.get("selected_plan_id"),
            "matrix_pass_index": record.get("matrix_pass_index"),
            "matrix_pass_count": record.get("matrix_pass_count"),
        },
        "fit": fit,
        "source_kind": "phase_normalized_timeline" if rows else "continuous_event_timeline_only",
        "phase_anchor_fallback": bool(phase_anchor_fallback and rows),
        "phase_anchor_estimate": phase_anchor_estimate if phase_anchor_fallback else None,
        "x_axis": x_axis,
        "time_align": time_align,
        "source_note": (
            "Actual points come from measured FW/BW/optimizer phase anchors. In matched-total mode, simulated "
            "phase anchors are evenly normalized over the measured step duration; raw-costmodel mode evenly "
            "normalizes anchors over the selected plan's estimated step time. This is not an op-level or "
            "continuous sampled CUDA memory trace."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "source.csv", rows)
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        (output_dir / "figure.svg").write_text(_render_svg(record, rows, fit, x_axis=x_axis), encoding="utf-8")
    if sampled_rows:
        _write_sampled_csv(output_dir / "actual_sampled_source.csv", sampled_rows)
        (output_dir / "actual_sampled_figure.svg").write_text(
            _render_sampled_svg(record, sampled_rows, rows, x_axis=x_axis),
            encoding="utf-8",
        )
    if sampled_rows and simulated_event_rows:
        _write_simulated_event_csv(output_dir / "simulated_event_source.csv", simulated_event_rows)
        event_fit = _event_trace_fit(sampled_rows, simulated_event_rows)
        event_fit.update(_event_alignment_metadata(record, simulated_event_rows))
        (output_dir / "event_fit_summary.json").write_text(
            json.dumps(event_fit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "actual_vs_simulated_event_figure.svg").write_text(
            _render_event_svg(record, sampled_rows, simulated_event_rows, event_fit),
            encoding="utf-8",
        )
        gpu_rows = _gpu_util_rows(record)
        if gpu_rows:
            _write_gpu_util_csv(output_dir / "gpu_util_source.csv", gpu_rows)
            (output_dir / "actual_vs_simulated_event_gpu_figure.svg").write_text(
                _render_event_gpu_svg(record, sampled_rows, simulated_event_rows, gpu_rows, event_fit),
                encoding="utf-8",
            )
    return summary, rows


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with a records list")
    return [dict(item) for item in payload]


def _phase_anchor_actual_timeline(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    phase_metrics = _selected_phase_metrics(record)
    values = {
        "start": 0,
        "fw_peak": _first_int(phase_metrics.get("fw_peak_bytes"), record.get("measured_fw_peak_bytes")),
        "after_fw": _first_int(
            phase_metrics.get("after_fw_allocated_bytes"),
            record.get("measured_after_fw_allocated_bytes"),
        ),
        "bw_peak": _first_int(phase_metrics.get("bw_peak_bytes"), record.get("measured_bw_peak_bytes")),
        "optimizer_peak": _first_int(
            phase_metrics.get("optimizer_peak_bytes"),
            record.get("measured_optimizer_peak_bytes"),
        ),
        "overall_peak": _first_int(phase_metrics.get("overall_peak_bytes"), record.get("measured_peak_bytes")),
    }
    return _timeline_from_phase_values(values)


def _phase_anchor_simulated_timeline(record: dict[str, Any], *, estimate_kind: str) -> tuple[dict[str, Any], ...]:
    plan_result = _selected_plan_result(record)
    if estimate_kind == "calibrated":
        estimated_peak = _first_int(
            plan_result.get("calibrated_estimated_peak_bytes"),
            record.get("selected_calibrated_estimated_peak_bytes"),
            record.get("selected_estimated_peak_bytes"),
        )
        peak_phase = (
            plan_result.get("all_save_phase_calibrated_estimated_peak_phase")
            or record.get("selected_calibrated_peak_phase")
            or record.get("selected_peak_phase")
        )
    else:
        estimated_peak = _first_int(plan_result.get("estimated_peak_bytes"), record.get("selected_estimated_peak_bytes"))
        peak_phase = record.get("selected_peak_phase") or plan_result.get("estimated_peak_phase")
    if estimated_peak is None:
        return ()
    phase_name = _normalize_peak_phase(peak_phase)
    values: dict[str, Any] = {"start": 0, "overall_peak": estimated_peak}
    if phase_name is not None:
        values[phase_name] = estimated_peak
    return _timeline_from_phase_values(values)


def _timeline_from_phase_values(values: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    order = ("start", "fw_peak", "after_fw", "bw_peak", "optimizer_peak", "overall_peak")
    return tuple({"phase": phase, "bytes": int(values[phase])} for phase in order if values.get(phase) is not None)


def _selected_phase_metrics(record: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_plan_result(record)
    phase_metrics = selected.get("phase_metrics")
    return dict(phase_metrics) if isinstance(phase_metrics, dict) else {}


def _selected_plan_result(record: dict[str, Any]) -> dict[str, Any]:
    selected_plan_id = record.get("selected_plan_id")
    for row in record.get("measured_plan_results", ()):
        if isinstance(row, dict) and row.get("plan_id") == selected_plan_id:
            return dict(row)
    return {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _normalize_peak_phase(value: Any) -> str | None:
    phase = str(value or "").lower()
    if phase in {"fw", "forward", "fw_peak"}:
        return "fw_peak"
    if phase in {"bw", "backward", "bw_peak"}:
        return "bw_peak"
    if phase in {"opt", "optimizer", "optimizer_peak"}:
        return "optimizer_peak"
    if phase in {"overall", "overall_peak", "true_peak"}:
        return "overall_peak"
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "task_name",
        "variant_name",
        "microbatch_size",
        "budget_bytes",
        "selected_plan_id",
        "matrix_pass_index",
        "phase",
        "position",
        "actual_x",
        "simulated_x",
        "actual_bytes",
        "simulated_bytes",
        "offset_fitted_simulated_bytes",
        "error_bytes",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sampled_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "task_name",
        "variant_name",
        "microbatch_size",
        "budget_bytes",
        "selected_plan_id",
        "trajectory",
        "phase",
        "event",
        "time_us",
        "allocated_bytes",
        "reserved_bytes",
        "sampled_peak_bytes",
        "peak_bytes",
        "allocator_peak_bytes",
        "allocator_reserved_peak_bytes",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_simulated_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "task_name",
        "variant_name",
        "microbatch_size",
        "budget_bytes",
        "selected_plan_id",
        "phase",
        "event",
        "time_us",
        "raw_time_us",
        "bytes",
        "fixed_bytes",
        "payload_bytes",
        "workspace_bytes",
        "optimizer_bytes",
        "gradient_bytes",
        "parameter_bytes",
        "buffer_bytes",
        "live_storage_count",
        "trace_kind",
        "memory_model_kind",
        "op_id",
        "op_name",
        "source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gpu_util_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "task_name",
        "variant_name",
        "microbatch_size",
        "budget_bytes",
        "selected_plan_id",
        "time_us",
        "event",
        "device_index",
        "gpu_util_percent",
        "memory_util_percent",
        "memory_used_mib",
        "power_w",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sampled_trace_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = (
        record.get("selected_actual_overall_sampled_memory_trace")
        or record.get("selected_actual_sampled_memory_trace")
        or ()
    )
    trajectory = "overall" if record.get("selected_actual_overall_sampled_memory_trace") else "phase"
    rows = []
    for point in trace:
        if not isinstance(point, dict) or point.get("time_us") is None:
            continue
        allocator_peak = int(point.get("allocator_peak_bytes") or point.get("peak_bytes") or 0)
        rows.append(
            {
                "task_name": record.get("task_name"),
                "variant_name": record.get("variant_name"),
                "microbatch_size": record.get("microbatch_size"),
                "budget_bytes": record.get("budget_bytes"),
                "selected_plan_id": record.get("selected_plan_id"),
                "trajectory": trajectory,
                "phase": point.get("phase"),
                "event": point.get("event", "sample"),
                "time_us": float(point.get("time_us") or 0.0),
                "allocated_bytes": int(point.get("allocated_bytes") or 0),
                "reserved_bytes": int(point.get("reserved_bytes") or 0),
                "sampled_peak_bytes": int(point.get("sampled_peak_bytes") or point.get("allocated_bytes") or 0),
                "peak_bytes": allocator_peak,
                "allocator_peak_bytes": allocator_peak,
                "allocator_reserved_peak_bytes": int(point.get("allocator_reserved_peak_bytes") or 0),
            }
        )
    return rows


def _gpu_util_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("selected_gpu_util_trace") or record.get("gpu_util_trace") or ()
    rows = []
    for point in trace:
        if not isinstance(point, dict) or point.get("time_us") is None:
            continue
        rows.append(
            {
                "task_name": record.get("task_name"),
                "variant_name": record.get("variant_name"),
                "microbatch_size": record.get("microbatch_size"),
                "budget_bytes": record.get("budget_bytes"),
                "selected_plan_id": record.get("selected_plan_id"),
                "time_us": float(point.get("time_us") or 0.0),
                "event": point.get("event", "sample"),
                "device_index": point.get("device_index"),
                "gpu_util_percent": _float_or_empty(point.get("gpu_util_percent")),
                "memory_util_percent": _float_or_empty(point.get("memory_util_percent")),
                "memory_used_mib": _float_or_empty(point.get("memory_used_mib")),
                "power_w": _float_or_empty(point.get("power_w")),
                "sm_clock_mhz": _float_or_empty(point.get("sm_clock_mhz")),
                "mem_clock_mhz": _float_or_empty(point.get("mem_clock_mhz")),
                "source": point.get("source", "unknown"),
            }
        )
    return rows


def _float_or_empty(value: Any) -> float | str:
    if value is None:
        return ""
    return float(value)


def _phase_durations_us(record: dict[str, Any], sampled_rows: list[dict[str, Any]]) -> dict[str, float]:
    fw_us = float(record.get("measured_fw_us") or 0.0)
    bw_us = float(record.get("measured_bw_us") or 0.0)
    optimizer_us = float(record.get("measured_optimizer_us") or 0.0)
    if fw_us > 0.0 or bw_us > 0.0 or optimizer_us > 0.0:
        return {"fw": max(fw_us, 0.0), "bw": max(bw_us, 0.0), "optimizer": max(optimizer_us, 0.0)}

    phase_ranges: dict[str, list[float]] = {}
    for row in sampled_rows:
        phase = str(row.get("phase") or "")
        if phase not in {"fw", "bw", "optimizer"}:
            continue
        phase_ranges.setdefault(phase, []).append(float(row["time_us"]))
    durations = {
        phase: max(values) - min(values) if values else 0.0
        for phase, values in phase_ranges.items()
    }
    return {
        "fw": max(float(durations.get("fw", 0.0)), 0.0),
        "bw": max(float(durations.get("bw", 0.0)), 0.0),
        "optimizer": max(float(durations.get("optimizer", 0.0)), 0.0),
    }


def _raw_phase_bounds(trace: tuple[Any, ...]) -> dict[str, tuple[float, float]]:
    fw_end = max(
        (float(point.get("time_us") or 0.0) for point in trace if isinstance(point, dict) and point.get("phase") in {"fw", "after_fw"}),
        default=0.0,
    )
    bw_end = max(
        (float(point.get("time_us") or 0.0) for point in trace if isinstance(point, dict) and point.get("phase") == "bw"),
        default=fw_end,
    )
    optimizer_end = max(
        (
            float(point.get("time_us") or 0.0)
            for point in trace
            if isinstance(point, dict) and point.get("phase") in {"optimizer", "overall"}
        ),
        default=bw_end,
    )
    return {
        "start": (0.0, fw_end),
        "fw": (0.0, fw_end),
        "after_fw": (0.0, fw_end),
        "bw": (fw_end, bw_end),
        "optimizer": (bw_end, optimizer_end),
        "overall": (bw_end, optimizer_end),
    }


def _align_event_time_us(
    point: dict[str, Any],
    *,
    bounds: dict[str, tuple[float, float]],
    durations: dict[str, float],
    total_scale: float,
) -> float:
    raw_time = float(point.get("time_us") or 0.0)
    phase = str(point.get("phase") or "")
    if phase in {"fw", "after_fw", "start"}:
        raw_start, raw_end = bounds["fw"]
        raw_span = max(raw_end - raw_start, 0.0)
        return (raw_time - raw_start) / raw_span * durations.get("fw", 0.0) if raw_span > 0 else 0.0
    if phase == "bw":
        raw_start, raw_end = bounds["bw"]
        raw_span = max(raw_end - raw_start, 0.0)
        actual_start = durations.get("fw", 0.0)
        return actual_start + ((raw_time - raw_start) / raw_span * durations.get("bw", 0.0) if raw_span > 0 else 0.0)
    if phase in {"optimizer", "overall"}:
        raw_start, raw_end = bounds["optimizer"]
        raw_span = max(raw_end - raw_start, 0.0)
        actual_start = durations.get("fw", 0.0) + durations.get("bw", 0.0)
        return actual_start + (
            (raw_time - raw_start) / raw_span * durations.get("optimizer", 0.0) if raw_span > 0 else 0.0
        )
    return raw_time * total_scale


def _simulated_event_rows(record: dict[str, Any], *, sampled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace = record.get("selected_lowered_fx_l2_simulated_memory_event_trace") or ()
    trace_kind = "selected_lowered_fx_l2"
    if not trace:
        trace = record.get("selected_simulated_memory_event_trace") or ()
        trace_kind = "selected_ir_liveness"
    if not trace:
        selected_plan_id = record.get("selected_plan_id")
        for row in record.get("measured_plan_results", ()):
            if isinstance(row, dict) and row.get("plan_id") == selected_plan_id:
                trace = row.get("lowered_fx_l2_simulated_memory_event_trace") or ()
                trace_kind = "measured_plan_lowered_fx_l2"
                if not trace:
                    trace = row.get("simulated_memory_event_trace") or ()
                    trace_kind = "measured_plan_ir_liveness"
                break
    if not trace:
        return []
    raw_max = max((float(point.get("time_us") or 0.0) for point in trace if isinstance(point, dict)), default=0.0)
    sampled_max = max((float(row["time_us"]) for row in sampled_rows), default=0.0)
    total_scale = sampled_max / raw_max if raw_max > 0 and sampled_max > 0 else 1.0
    durations = _phase_durations_us(record, sampled_rows)
    duration_total = sum(durations.get(phase, 0.0) for phase in ("fw", "bw", "optimizer"))
    if duration_total <= 0.0 and sampled_max > 0.0:
        durations = {"fw": sampled_max / 3.0, "bw": sampled_max / 3.0, "optimizer": sampled_max / 3.0}
    bounds = _raw_phase_bounds(tuple(trace))
    rows = []
    for point in trace:
        if not isinstance(point, dict):
            continue
        raw_time = float(point.get("time_us") or 0.0)
        rows.append(
            {
                "task_name": record.get("task_name"),
                "variant_name": record.get("variant_name"),
                "microbatch_size": record.get("microbatch_size"),
                "budget_bytes": record.get("budget_bytes"),
                "selected_plan_id": record.get("selected_plan_id"),
                "phase": point.get("phase"),
                "event": point.get("event"),
                "time_us": _align_event_time_us(
                    point,
                    bounds=bounds,
                    durations=durations,
                    total_scale=total_scale,
                ),
                "raw_time_us": raw_time,
                "bytes": int(point.get("bytes") or 0),
                "fixed_bytes": int(point.get("fixed_bytes") or 0),
                "payload_bytes": int(point.get("payload_bytes") or 0),
                "workspace_bytes": int(point.get("workspace_bytes") or 0),
                "optimizer_bytes": int(point.get("optimizer_bytes") or 0),
                "gradient_bytes": int(point.get("gradient_bytes") or 0),
                "parameter_bytes": int(point.get("parameter_bytes") or 0),
                "buffer_bytes": int(point.get("buffer_bytes") or 0),
                "live_storage_count": int(point.get("live_storage_count") or 0),
                "op_id": point.get("op_id"),
                "op_name": point.get("op_name"),
                "source": point.get("source"),
                "trace_kind": trace_kind,
                "memory_model_kind": point.get("memory_model_kind") or "aten_ir_liveness_l2_components",
            }
        )
    return rows


def _event_alignment_metadata(record: dict[str, Any], simulated_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aligned_times: dict[str, list[float]] = {}
    raw_times: dict[str, list[float]] = {}
    for row in simulated_rows:
        phase = str(row.get("phase") or "")
        if phase not in {"start", "fw", "after_fw", "bw", "optimizer", "overall"}:
            continue
        phase_key = "fw" if phase in {"start", "after_fw"} else "optimizer" if phase == "overall" else phase
        aligned_times.setdefault(phase_key, []).append(float(row.get("time_us") or 0.0))
        raw_times.setdefault(phase_key, []).append(float(row.get("raw_time_us") or 0.0))

    def span(values: list[float]) -> float:
        return max(values) - min(values) if values else 0.0

    measured_phase_us = {
        "fw": float(record.get("measured_fw_us") or 0.0),
        "bw": float(record.get("measured_bw_us") or 0.0),
        "optimizer": float(record.get("measured_optimizer_us") or 0.0),
    }
    aligned_phase_span_us = {
        phase: measured_phase_us[phase] if measured_phase_us[phase] > 0.0 else span(aligned_times.get(phase, []))
        for phase in ("fw", "bw", "optimizer")
    }
    raw_fw_end = max(raw_times.get("fw", [0.0]))
    raw_bw_end = max(raw_times.get("bw", [raw_fw_end]))
    raw_optimizer_end = max(raw_times.get("optimizer", [raw_bw_end]))
    memory_model_kinds = tuple(
        sorted(
            {
                str(row.get("memory_model_kind"))
                for row in simulated_rows
                if row.get("memory_model_kind")
            }
        )
    )
    trace_kinds = tuple(sorted({str(row.get("trace_kind")) for row in simulated_rows if row.get("trace_kind")}))
    return {
        "time_alignment_kind": "phasewise_costmodel_to_measured",
        "memory_model_kind": memory_model_kinds[0] if len(memory_model_kinds) == 1 else "mixed_or_legacy_liveness",
        "memory_model_kinds": memory_model_kinds,
        "trace_kinds": trace_kinds,
        "measured_phase_us": measured_phase_us,
        "aligned_phase_span_us": aligned_phase_span_us,
        "raw_costmodel_phase_span_us": {
            "fw": max(raw_fw_end, 0.0),
            "bw": max(raw_bw_end - raw_fw_end, 0.0),
            "optimizer": max(raw_optimizer_end - raw_bw_end, 0.0),
        },
        "component_fields": (
            "fixed_bytes",
            "payload_bytes",
            "workspace_bytes",
            "optimizer_bytes",
            "gradient_bytes",
            "parameter_bytes",
            "buffer_bytes",
            "live_storage_count",
        ),
    }


def _event_trace_fit(actual_rows: list[dict[str, Any]], simulated_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sim_points = sorted((float(row["time_us"]), int(row["bytes"])) for row in simulated_rows)
    if not actual_rows or not sim_points:
        return {"point_count": 0}
    errors = []
    peak_errors = []
    for row in actual_rows:
        actual = int(row["allocated_bytes"])
        actual_peak = int(row["allocator_peak_bytes"])
        simulated = _interpolate_step(sim_points, float(row["time_us"]))
        errors.append(actual - simulated)
        peak_errors.append(actual_peak - simulated)
    squared = [error * error for error in errors]
    abs_errors = [abs(error) for error in errors]
    peak_actual = max(int(row["allocator_peak_bytes"]) for row in actual_rows)
    peak_sim = max(value for _time, value in sim_points)
    return {
        "point_count": len(errors),
        "mae_bytes": sum(abs_errors) / len(abs_errors),
        "rmse_bytes": (sum(squared) / len(squared)) ** 0.5,
        "peak_error_bytes": peak_actual - peak_sim,
        "actual_peak_bytes": peak_actual,
        "simulated_peak_bytes": peak_sim,
        "allocated_peak_error_bytes": max(int(row["allocated_bytes"]) for row in actual_rows) - peak_sim,
        "allocator_peak_mae_bytes": sum(abs(error) for error in peak_errors) / len(peak_errors),
    }


def _interpolate_step(points: list[tuple[float, int]], x_value: float) -> int:
    current = points[0][1]
    for time_us, value in points:
        if time_us > x_value:
            break
        current = value
    return current


def _record_label(record: dict[str, Any], index: int) -> str:
    parts = (
        f"{index:03d}",
        str(record.get("task_name", "task")),
        f"mb{record.get('microbatch_size', 'x')}",
        f"budget{int(record.get('budget_bytes') or 0) >> 20}mib",
        str(record.get("selected_plan_id", "plan")),
    )
    text = "_".join(parts)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _render_svg(record: dict[str, Any], rows: list[dict[str, Any]], fit: dict[str, Any], *, x_axis: str) -> str:
    width = 920
    height = 520
    left = 88
    right = 28
    top = 56
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_bytes = max(
        max(row["actual_bytes"], row["simulated_bytes"], row["offset_fitted_simulated_bytes"])
        for row in rows
    )
    max_mib = max_bytes / 1048576.0
    y_max = max(1.0, max_mib * 1.12)
    x_min = min(min(float(row["actual_x"]), float(row["simulated_x"])) for row in rows)
    x_max = max(max(float(row["actual_x"]), float(row["simulated_x"])) for row in rows)

    def x_pos(position: float) -> float:
        return left + (position - x_min) / max(x_max - x_min, 1.0) * plot_w

    def y_pos(value_bytes: int) -> float:
        return top + (1.0 - (value_bytes / 1048576.0) / y_max) * plot_h

    actual_points = " ".join(f"{x_pos(float(row['actual_x'])):.1f},{y_pos(int(row['actual_bytes'])):.1f}" for row in rows)
    simulated_points = " ".join(
        f"{x_pos(float(row['simulated_x'])):.1f},{y_pos(int(row['simulated_bytes'])):.1f}" for row in rows
    )
    fitted_points = " ".join(
        f"{x_pos(float(row['simulated_x'])):.1f},{y_pos(float(row['offset_fitted_simulated_bytes'])):.1f}"
        for row in rows
    )
    y_ticks = [0.0, y_max / 4, y_max / 2, y_max * 3 / 4, y_max]
    phase_labels = {
        "start": "Start",
        "fw_peak": "FW peak",
        "after_fw": "After FW",
        "bw_peak": "BW peak",
        "optimizer_peak": "OPT peak",
        "overall_peak": "Overall",
    }
    grid = []
    for tick in y_ticks:
        y = top + (1.0 - tick / y_max) * plot_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#42526e">{tick:.1f}</text>'
        )
    x_labels = []
    for row in rows:
        x = x_pos(float(row["actual_x"]))
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - 48}" text-anchor="middle" font-size="12" fill="#42526e">'
            f'{escape(phase_labels.get(str(row["phase"]), str(row["phase"])))}'
            "</text>"
        )

    title = f"{record.get('task_name')} / {record.get('selected_plan_id')}"
    subtitle = (
        f"raw RMSE={_mib(fit.get('rmse_bytes')):.2f} MiB, "
        f"fitted RMSE={_mib(fit.get('offset_fitted_rmse_bytes')):.2f} MiB, "
        f"r={_number(fit.get('pearson_correlation'))}"
    )
    dots = []
    for row in rows:
        ax = x_pos(float(row["actual_x"]))
        ay = y_pos(int(row["actual_bytes"]))
        sx = x_pos(float(row["simulated_x"]))
        sy = y_pos(int(row["simulated_bytes"]))
        dots.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="4.5" fill="#1f77b4"/>')
        dots.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.5" fill="#d62728"/>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="30" font-size="20" font-weight="700" fill="#172b4d">{escape(title)}</text>
  <text x="{left}" y="50" font-size="13" fill="#5e6c84">{escape(subtitle)}</text>
  <g>{"".join(grid)}</g>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <polyline points="{actual_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <polyline points="{simulated_points}" fill="none" stroke="#d62728" stroke-width="2" stroke-dasharray="4 5" opacity="0.55"/>
  <polyline points="{fitted_points}" fill="none" stroke="#d62728" stroke-width="3" stroke-dasharray="8 4"/>
  <g>{"".join(dots)}</g>
  <g>{"".join(x_labels)}</g>
  <text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#42526e">Memory (MiB)</text>
  <text x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="13" fill="#42526e">{escape(_x_axis_label(x_axis))}</text>
  <line x1="{width - 234}" y1="30" x2="{width - 190}" y2="30" stroke="#1f77b4" stroke-width="3"/>
  <text x="{width - 180}" y="34" font-size="13" fill="#172b4d">Actual</text>
  <line x1="{width - 124}" y1="30" x2="{width - 80}" y2="30" stroke="#d62728" stroke-width="3" stroke-dasharray="7 5"/>
  <text x="{width - 70}" y="34" font-size="13" fill="#172b4d">Fitted</text>
</svg>
"""


def _render_sampled_svg(
    record: dict[str, Any],
    sampled_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    *,
    x_axis: str,
) -> str:
    width = 920
    height = 520
    left = 88
    right = 28
    top = 56
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_bytes = max(
        max(row["allocated_bytes"], row["reserved_bytes"], row["allocator_peak_bytes"]) for row in sampled_rows
    )
    if phase_rows:
        max_bytes = max(
            max_bytes,
            max(max(row["simulated_bytes"], row["offset_fitted_simulated_bytes"]) for row in phase_rows),
        )
    y_max = max(1.0, max_bytes / 1048576.0 * 1.12)
    x_min = min(float(row["time_us"]) for row in sampled_rows)
    x_max = max(float(row["time_us"]) for row in sampled_rows)

    def x_pos(position: float) -> float:
        return left + (position - x_min) / max(x_max - x_min, 1.0) * plot_w

    def y_pos(value_bytes: float) -> float:
        return top + (1.0 - (value_bytes / 1048576.0) / y_max) * plot_h

    actual_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['allocated_bytes'])):.1f}"
        for row in sampled_rows
    )
    reserved_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['reserved_bytes'])):.1f}"
        for row in sampled_rows
    )
    allocator_peak_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['allocator_peak_bytes'])):.1f}"
        for row in sampled_rows
    )
    simulated_points = " ".join(
        f"{x_pos(float(row['simulated_x'])):.1f},{y_pos(float(row['offset_fitted_simulated_bytes'])):.1f}"
        for row in phase_rows
    )
    y_ticks = [0.0, y_max / 4, y_max / 2, y_max * 3 / 4, y_max]
    grid = []
    for tick in y_ticks:
        y = top + (1.0 - tick / y_max) * plot_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#42526e">{tick:.1f}</text>'
        )
    simulated_polyline = (
        f'<polyline points="{simulated_points}" fill="none" stroke="#d62728" stroke-width="3" stroke-dasharray="8 4"/>'
        if simulated_points
        else ""
    )
    title = f"{record.get('task_name')} / {record.get('selected_plan_id')} sampled CUDA memory"
    trajectory = sampled_rows[0].get("trajectory", "phase") if sampled_rows else "phase"
    subtitle = f"Actual {trajectory} sampled allocated/reserved memory and CUDA allocator peak"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="30" font-size="20" font-weight="700" fill="#172b4d">{escape(title)}</text>
  <text x="{left}" y="50" font-size="13" fill="#5e6c84">{escape(subtitle)}</text>
  <g>{"".join(grid)}</g>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <polyline points="{reserved_points}" fill="none" stroke="#7a869a" stroke-width="2" opacity="0.65"/>
  <polyline points="{actual_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <polyline points="{allocator_peak_points}" fill="none" stroke="#9467bd" stroke-width="2.4" stroke-dasharray="3 4"/>
  {simulated_polyline}
  <text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#42526e">Memory (MiB)</text>
  <text x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="13" fill="#42526e">{escape(_x_axis_label(x_axis))}</text>
  <line x1="{width - 382}" y1="30" x2="{width - 342}" y2="30" stroke="#1f77b4" stroke-width="3"/>
  <text x="{width - 334}" y="34" font-size="13" fill="#172b4d">Allocated</text>
  <line x1="{width - 252}" y1="30" x2="{width - 212}" y2="30" stroke="#7a869a" stroke-width="2"/>
  <text x="{width - 204}" y="34" font-size="13" fill="#172b4d">Reserved</text>
  <line x1="{width - 122}" y1="30" x2="{width - 82}" y2="30" stroke="#9467bd" stroke-width="2.4" stroke-dasharray="3 4"/>
  <text x="{width - 74}" y="34" font-size="13" fill="#172b4d">Peak</text>
</svg>
"""


def _render_event_svg(
    record: dict[str, Any],
    sampled_rows: list[dict[str, Any]],
    simulated_rows: list[dict[str, Any]],
    fit: dict[str, Any],
) -> str:
    width = 960
    height = 540
    left = 88
    right = 30
    top = 58
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_bytes = max(
        max(max(row["allocated_bytes"], row["allocator_peak_bytes"]) for row in sampled_rows),
        max(int(row["bytes"]) for row in simulated_rows),
    )
    y_max = max(1.0, max_bytes / 1048576.0 * 1.12)
    x_min = 0.0
    x_max = max(
        max(float(row["time_us"]) for row in sampled_rows),
        max(float(row["time_us"]) for row in simulated_rows),
        1.0,
    )

    def x_pos(position: float) -> float:
        return left + (position - x_min) / max(x_max - x_min, 1.0) * plot_w

    def y_pos(value_bytes: float) -> float:
        return top + (1.0 - (value_bytes / 1048576.0) / y_max) * plot_h

    actual_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['allocated_bytes'])):.1f}"
        for row in sampled_rows
    )
    allocator_peak_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['allocator_peak_bytes'])):.1f}"
        for row in sampled_rows
    )
    simulated_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(row['bytes'])):.1f}"
        for row in simulated_rows
    )
    y_ticks = [0.0, y_max / 4, y_max / 2, y_max * 3 / 4, y_max]
    grid = []
    for tick in y_ticks:
        y = top + (1.0 - tick / y_max) * plot_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#42526e">{tick:.1f}</text>'
        )
    phase_markers = []
    for row in simulated_rows:
        if row.get("event") != "phase_boundary" and row.get("event") != "phase_start":
            continue
        x = x_pos(float(row["time_us"]))
        phase_markers.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#c1c7d0" stroke-width="1" stroke-dasharray="2 5"/>'
        )
    title = f"{record.get('task_name')} / {record.get('selected_plan_id')} event timeline"
    subtitle = (
        f"event RMSE={_mib(fit.get('rmse_bytes')):.2f} MiB, "
        f"peak error={_mib(fit.get('peak_error_bytes')):.2f} MiB"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="30" font-size="20" font-weight="700" fill="#172b4d">{escape(title)}</text>
  <text x="{left}" y="50" font-size="13" fill="#5e6c84">{escape(subtitle)}</text>
  <g>{"".join(grid)}</g>
  <g>{"".join(phase_markers)}</g>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#172b4d" stroke-width="1.2"/>
  <polyline points="{allocator_peak_points}" fill="none" stroke="#9467bd" stroke-width="2.2" stroke-dasharray="3 4"/>
  <polyline points="{actual_points}" fill="none" stroke="#1f77b4" stroke-width="2.8"/>
  <polyline points="{simulated_points}" fill="none" stroke="#d62728" stroke-width="2.4"/>
  <text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#42526e">Memory (MiB)</text>
  <text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#42526e">Time (us, actual sampled vs costmodel event trace)</text>
  <line x1="{width - 382}" y1="30" x2="{width - 342}" y2="30" stroke="#1f77b4" stroke-width="2.8"/>
  <text x="{width - 334}" y="34" font-size="13" fill="#172b4d">Actual allocated</text>
  <line x1="{width - 232}" y1="30" x2="{width - 192}" y2="30" stroke="#9467bd" stroke-width="2.2" stroke-dasharray="3 4"/>
  <text x="{width - 184}" y="34" font-size="13" fill="#172b4d">Allocator peak</text>
  <line x1="{width - 102}" y1="30" x2="{width - 62}" y2="30" stroke="#d62728" stroke-width="2.4"/>
  <text x="{width - 54}" y="34" font-size="13" fill="#172b4d">Sim</text>
</svg>
"""


def _render_event_gpu_svg(
    record: dict[str, Any],
    sampled_rows: list[dict[str, Any]],
    simulated_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    fit: dict[str, Any],
) -> str:
    width = 980
    height = 720
    left = 88
    right = 34
    top = 60
    memory_h = 350
    gap = 52
    util_top = top + memory_h + gap
    util_h = 185
    bottom = 50
    plot_w = width - left - right
    max_bytes = max(
        max(max(row["allocated_bytes"], row["allocator_peak_bytes"]) for row in sampled_rows),
        max(int(row["bytes"]) for row in simulated_rows),
    )
    memory_y_max = max(1.0, max_bytes / 1048576.0 * 1.12)
    x_min = 0.0
    x_max = max(
        max(float(row["time_us"]) for row in sampled_rows),
        max(float(row["time_us"]) for row in simulated_rows),
        max(float(row["time_us"]) for row in gpu_rows),
        1.0,
    )

    def x_pos(position: float) -> float:
        return left + (position - x_min) / max(x_max - x_min, 1.0) * plot_w

    def memory_y_pos(value_bytes: float) -> float:
        return top + (1.0 - (value_bytes / 1048576.0) / memory_y_max) * memory_h

    def util_y_pos(value_percent: float) -> float:
        clipped = min(max(value_percent, 0.0), 100.0)
        return util_top + (1.0 - clipped / 100.0) * util_h

    actual_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{memory_y_pos(float(row['allocated_bytes'])):.1f}"
        for row in sampled_rows
    )
    allocator_peak_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{memory_y_pos(float(row['allocator_peak_bytes'])):.1f}"
        for row in sampled_rows
    )
    simulated_points = " ".join(
        f"{x_pos(float(row['time_us'])):.1f},{memory_y_pos(float(row['bytes'])):.1f}"
        for row in simulated_rows
    )
    gpu_points = _percent_polyline_points(gpu_rows, "gpu_util_percent", x_pos, util_y_pos)
    memory_util_points = _percent_polyline_points(gpu_rows, "memory_util_percent", x_pos, util_y_pos)
    memory_grid = []
    for tick in [0.0, memory_y_max / 2, memory_y_max]:
        y = top + (1.0 - tick / memory_y_max) * memory_h
        memory_grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>'
        )
        memory_grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#42526e">{tick:.1f}</text>'
        )
    util_grid = []
    for tick in [0, 50, 100]:
        y = util_y_pos(float(tick))
        util_grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>'
        )
        util_grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#42526e">{tick}</text>'
        )
    phase_markers = []
    for row in simulated_rows:
        if row.get("event") not in {"phase_boundary", "phase_start"}:
            continue
        x = x_pos(float(row["time_us"]))
        phase_markers.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{util_top + util_h}" stroke="#c1c7d0" stroke-width="1" stroke-dasharray="2 5"/>'
        )
    title = f"{record.get('task_name')} / {record.get('selected_plan_id')} memory and GPU utilization"
    subtitle = (
        f"memory RMSE={_mib(fit.get('rmse_bytes')):.2f} MiB, "
        f"peak error={_mib(fit.get('peak_error_bytes')):.2f} MiB, "
        f"GPU samples={len(gpu_rows)}"
    )
    memory_util_polyline = (
        f'<polyline points="{memory_util_points}" fill="none" stroke="#ff7f0e" stroke-width="2.2"/>'
        if memory_util_points
        else ""
    )
    gpu_polyline = (
        f'<polyline points="{gpu_points}" fill="none" stroke="#2ca02c" stroke-width="2.8"/>'
        if gpu_points
        else ""
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="30" font-size="20" font-weight="700" fill="#172b4d">{escape(title)}</text>
  <text x="{left}" y="50" font-size="13" fill="#5e6c84">{escape(subtitle)}</text>
  <g>{"".join(memory_grid)}</g>
  <g>{"".join(util_grid)}</g>
  <g>{"".join(phase_markers)}</g>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + memory_h}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{top + memory_h}" x2="{width - right}" y2="{top + memory_h}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{util_top}" x2="{left}" y2="{util_top + util_h}" stroke="#172b4d" stroke-width="1.2"/>
  <line x1="{left}" y1="{util_top + util_h}" x2="{width - right}" y2="{util_top + util_h}" stroke="#172b4d" stroke-width="1.2"/>
  <polyline points="{allocator_peak_points}" fill="none" stroke="#9467bd" stroke-width="2.0" stroke-dasharray="3 4"/>
  <polyline points="{actual_points}" fill="none" stroke="#1f77b4" stroke-width="2.6"/>
  <polyline points="{simulated_points}" fill="none" stroke="#d62728" stroke-width="2.4"/>
  {gpu_polyline}
  {memory_util_polyline}
  <text x="20" y="{top + memory_h / 2:.1f}" transform="rotate(-90 20 {top + memory_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#42526e">Memory (MiB)</text>
  <text x="20" y="{util_top + util_h / 2:.1f}" transform="rotate(-90 20 {util_top + util_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#42526e">Utilization (%)</text>
  <text x="{left + plot_w / 2:.1f}" y="{height - bottom / 2:.1f}" text-anchor="middle" font-size="13" fill="#42526e">Time (us)</text>
  <line x1="{width - 520}" y1="30" x2="{width - 484}" y2="30" stroke="#1f77b4" stroke-width="2.6"/>
  <text x="{width - 476}" y="34" font-size="12" fill="#172b4d">Actual memory</text>
  <line x1="{width - 386}" y1="30" x2="{width - 350}" y2="30" stroke="#d62728" stroke-width="2.4"/>
  <text x="{width - 342}" y="34" font-size="12" fill="#172b4d">Sim memory</text>
  <line x1="{width - 252}" y1="30" x2="{width - 216}" y2="30" stroke="#2ca02c" stroke-width="2.8"/>
  <text x="{width - 208}" y="34" font-size="12" fill="#172b4d">GPU util</text>
  <line x1="{width - 132}" y1="30" x2="{width - 96}" y2="30" stroke="#ff7f0e" stroke-width="2.2"/>
  <text x="{width - 88}" y="34" font-size="12" fill="#172b4d">Mem util</text>
</svg>
"""


def _percent_polyline_points(
    rows: list[dict[str, Any]],
    field: str,
    x_pos: Any,
    y_pos: Any,
) -> str:
    points = []
    for row in rows:
        value = row.get(field)
        if value == "" or value is None:
            continue
        points.append(f"{x_pos(float(row['time_us'])):.1f},{y_pos(float(value)):.1f}")
    return " ".join(points)


def _mib(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value) / 1048576.0


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _x_axis_label(x_axis: str) -> str:
    if x_axis == "time":
        return "Time (us, measured actual vs costmodel-aligned simulated)"
    return "Phase anchor"


if __name__ == "__main__":
    main()
