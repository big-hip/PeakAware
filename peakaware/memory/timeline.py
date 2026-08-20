from __future__ import annotations

import math
from typing import Any

PHASE_TIMELINE_SPEC: tuple[tuple[str, float], ...] = (
    ("start", 0.0),
    ("fw_peak", 1.0),
    ("after_fw", 2.0),
    ("bw_peak", 3.0),
    ("optimizer_peak", 4.0),
    ("overall_peak", 5.0),
)


def build_simulated_memory_timeline(
    *,
    fw_peak_bytes: int | None,
    after_fw_retained_bytes: int | None,
    bw_peak_bytes: int | None,
    optimizer_peak_bytes: int | None,
    estimated_peak_bytes: int | None,
) -> tuple[dict[str, Any], ...]:
    values = {
        "start": 0,
        "fw_peak": fw_peak_bytes,
        "after_fw": after_fw_retained_bytes,
        "bw_peak": bw_peak_bytes,
        "optimizer_peak": optimizer_peak_bytes,
        "overall_peak": estimated_peak_bytes,
    }
    return _timeline_from_values(values)


def build_actual_memory_timeline(
    phase_metrics: dict[str, Any],
    *,
    measured_peak_bytes: int | None = None,
) -> tuple[dict[str, Any], ...]:
    values = {
        "start": 0,
        "fw_peak": phase_metrics.get("fw_peak_bytes"),
        "after_fw": phase_metrics.get("after_fw_allocated_bytes"),
        "bw_peak": phase_metrics.get("bw_peak_bytes"),
        "optimizer_peak": phase_metrics.get("optimizer_peak_bytes"),
        "overall_peak": measured_peak_bytes if measured_peak_bytes is not None else phase_metrics.get("overall_peak_bytes"),
    }
    return _timeline_from_values(values)


def fit_memory_timelines(
    actual: tuple[dict[str, Any], ...],
    simulated: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    actual_by_phase = {str(point["phase"]): int(point["bytes"]) for point in actual}
    simulated_by_phase = {str(point["phase"]): int(point["bytes"]) for point in simulated}
    phases = [phase for phase, _ in PHASE_TIMELINE_SPEC if phase in actual_by_phase and phase in simulated_by_phase]
    pairs = [(actual_by_phase[phase], simulated_by_phase[phase]) for phase in phases]
    if not pairs:
        return {
            "point_count": 0,
            "phases": (),
            "mae_bytes": None,
            "rmse_bytes": None,
            "max_abs_error_bytes": None,
            "peak_error_bytes": None,
            "normalized_rmse": None,
            "pearson_correlation": None,
        }

    errors = [actual_value - simulated_value for actual_value, simulated_value in pairs]
    abs_errors = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    offset_errors = [error for phase, error in zip(phases, errors, strict=True) if phase != "start"] or errors
    offset = sum(offset_errors) / len(offset_errors)
    fitted_errors = [
        actual_value - (_fitted_value(phase, actual_value, simulated_value, offset))
        for phase, (actual_value, simulated_value) in zip(phases, pairs, strict=True)
    ]
    fitted_abs_errors = [abs(error) for error in fitted_errors]
    fitted_squared = [error * error for error in fitted_errors]
    actual_peak = max(actual_value for actual_value, _ in pairs)
    simulated_peak = max(simulated_value for _, simulated_value in pairs)
    rmse = math.sqrt(sum(squared) / len(squared))
    fitted_rmse = math.sqrt(sum(fitted_squared) / len(fitted_squared))
    return {
        "point_count": len(pairs),
        "phases": tuple(phases),
        "mean_error_bytes": sum(errors) / len(errors),
        "mae_bytes": sum(abs_errors) / len(abs_errors),
        "rmse_bytes": rmse,
        "max_abs_error_bytes": max(abs_errors),
        "peak_error_bytes": actual_peak - simulated_peak,
        "normalized_rmse": None if actual_peak == 0 else rmse / actual_peak,
        "offset_fitted_simulated_bytes": offset,
        "offset_fitted_mae_bytes": sum(fitted_abs_errors) / len(fitted_abs_errors),
        "offset_fitted_rmse_bytes": fitted_rmse,
        "offset_fitted_normalized_rmse": None if actual_peak == 0 else fitted_rmse / actual_peak,
        "pearson_correlation": _pearson([actual_value for actual_value, _ in pairs], [sim_value for _, sim_value in pairs]),
    }


def merge_timeline_rows(
    *,
    record: dict[str, Any],
    actual: tuple[dict[str, Any], ...],
    simulated: tuple[dict[str, Any], ...],
    offset_fitted_simulated_bytes: float = 0.0,
    x_axis: str = "phase",
) -> list[dict[str, Any]]:
    metadata = {
        "task_name": record.get("task_name"),
        "variant_name": record.get("variant_name"),
        "microbatch_size": record.get("microbatch_size"),
        "budget_bytes": record.get("budget_bytes"),
        "selected_plan_id": record.get("selected_plan_id"),
        "matrix_pass_index": record.get("matrix_pass_index"),
    }
    actual_by_phase = {str(point["phase"]): point for point in actual}
    simulated_by_phase = {str(point["phase"]): point for point in simulated}
    rows = []
    for phase, position in PHASE_TIMELINE_SPEC:
        if phase not in actual_by_phase or phase not in simulated_by_phase:
            continue
        actual_bytes = int(actual_by_phase[phase]["bytes"])
        simulated_bytes = int(simulated_by_phase[phase]["bytes"])
        actual_x = _actual_x_value(record, phase, position, x_axis)
        simulated_x = _simulated_x_value(record, phase, position, x_axis)
        rows.append(
            {
                **metadata,
                "phase": phase,
                "position": position,
                "actual_x": actual_x,
                "simulated_x": simulated_x,
                "actual_bytes": actual_bytes,
                "simulated_bytes": simulated_bytes,
                "offset_fitted_simulated_bytes": actual_bytes
                if phase == "start"
                else simulated_bytes + offset_fitted_simulated_bytes,
                "error_bytes": actual_bytes - simulated_bytes,
            }
        )
    return rows


def _timeline_from_values(values: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = []
    for phase, position in PHASE_TIMELINE_SPEC:
        value = values.get(phase)
        if value is None:
            continue
        rows.append({"phase": phase, "position": position, "bytes": int(value)})
    return tuple(rows)


def _pearson(left: list[int], right: list[int]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_delta, right_delta, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_delta))
    right_norm = math.sqrt(sum(value * value for value in right_delta))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _fitted_value(phase: str, actual_value: int, simulated_value: int, offset: float) -> float:
    if phase == "start":
        return float(actual_value)
    return simulated_value + offset


def _actual_x_value(record: dict[str, Any], phase: str, position: float, x_axis: str) -> float:
    if x_axis != "time":
        return position
    by_phase = {
        str(point.get("phase")): point
        for point in record.get("actual_memory_trace", ())
        if isinstance(point, dict)
    }
    point = by_phase.get(phase)
    if point is not None and point.get("time_us") is not None:
        return float(point["time_us"])
    fw_us = float(record.get("measured_fw_us") or 0.0)
    bw_us = float(record.get("measured_bw_us") or 0.0)
    optimizer_us = float(record.get("measured_optimizer_us") or 0.0)
    if phase == "start":
        return 0.0
    if phase in {"fw_peak", "after_fw"}:
        return fw_us
    if phase == "bw_peak":
        return fw_us + bw_us
    if phase in {"optimizer_peak", "overall_peak"}:
        return fw_us + bw_us + optimizer_us
    return position


def _simulated_x_value(record: dict[str, Any], phase: str, position: float, x_axis: str) -> float:
    if x_axis != "time":
        return position
    if record.get("time_align") == "matched-total":
        measured_step_us = _record_measured_step_us(record)
        if measured_step_us is not None and measured_step_us > 0:
            if phase == "start":
                return 0.0
            return measured_step_us * (position / PHASE_TIMELINE_SPEC[-1][1])
    estimated_step_us = float(_record_estimated_step_us(record) or 0.0)
    if estimated_step_us <= 0:
        return position
    if phase == "start":
        return 0.0
    return estimated_step_us * (position / PHASE_TIMELINE_SPEC[-1][1])


def _record_estimated_step_us(record: dict[str, Any]) -> float | None:
    direct = record.get("selected_estimated_step_us") or record.get("estimated_step_us")
    if direct is not None:
        return float(direct)
    selected_plan_id = record.get("selected_plan_id")
    for row in record.get("measured_plan_results", ()):
        if isinstance(row, dict) and row.get("plan_id") == selected_plan_id and row.get("estimated_step_us") is not None:
            return float(row["estimated_step_us"])
    return None


def _record_measured_step_us(record: dict[str, Any]) -> float | None:
    direct = record.get("measured_step_us")
    if direct is not None:
        return float(direct)
    actual_trace = record.get("actual_memory_trace", ())
    times = [
        float(point["time_us"])
        for point in actual_trace
        if isinstance(point, dict) and point.get("time_us") is not None
    ]
    return max(times) if times else None
