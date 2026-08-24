"""Reference-calibrated held-out memory-trajectory prediction.

The predictor keeps the target event identity and timestamps, transfers a
phase-normalized allocator envelope from independent reference shapes, and
records the difference between the predicted total and physical component
columns as an explicit signed model residual.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


PHASES = ("FW", "BW", "OPT")
DEFAULT_COMPONENT_FIELDS = (
    "parameter_bytes",
    "buffer_bytes",
    "optimizer_state_bytes",
    "saved_activation_bytes",
    "recomputed_activation_bytes",
    "gradient_bytes",
    "optimizer_temp_bytes",
    "temporary_bytes",
    "workspace_component_bytes",
    "allocator_residual_bytes",
)


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def _phase(row: Mapping[str, Any]) -> str:
    value = str(row.get("phase", ""))
    upper = value.upper()
    if upper in PHASES:
        return upper
    if value.lower() == "optimizer":
        return "OPT"
    return upper


def _time(row: Mapping[str, Any]) -> float:
    return _number(row, "time_us")


def _total_key(rows: Sequence[Mapping[str, Any]]) -> str:
    if rows and "current_bytes" in rows[0]:
        return "current_bytes"
    if rows and "allocated_bytes" in rows[0]:
        return "allocated_bytes"
    if rows and "bytes" in rows[0]:
        return "bytes"
    raise ValueError(
        "trajectory rows need current_bytes, allocated_bytes, or bytes"
    )


def _phase_rows(
    rows: Sequence[Mapping[str, Any]], phase: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if _phase(row) == phase]
    return sorted(selected, key=_time)


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _sample_by_progress(
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    grid: Sequence[float],
    value_key: str,
) -> list[float]:
    selected = _phase_rows(rows, phase)
    if not selected:
        return [0.0 for _ in grid]
    values = [_number(row, value_key) for row in selected]
    times = [_time(row) for row in selected]
    start, end = times[0], times[-1]
    progress = [
        0.0
        if end <= start
        else max(0.0, min(1.0, (time - start) / (end - start)))
        for time in times
    ]
    result: list[float] = []
    for query in grid:
        if query <= progress[0]:
            result.append(values[0])
            continue
        if query >= progress[-1]:
            result.append(values[-1])
            continue
        index = 0
        while index + 1 < len(progress) and progress[index + 1] < query:
            index += 1
        left, right = progress[index], progress[index + 1]
        if right <= left:
            result.append(values[index + 1])
        else:
            fraction = (query - left) / (right - left)
            result.append(
                values[index] + fraction * (values[index + 1] - values[index])
            )
    return result


def _fit_alpha(
    source: Sequence[float],
    scheduler: Sequence[float],
    target: Sequence[float],
) -> float:
    direction = [s - q for s, q in zip(source, scheduler)]
    residual = [t - q for t, q in zip(target, scheduler)]
    denominator = sum(value * value for value in direction)
    if denominator <= 0.0:
        return 1.0
    alpha = sum(left * right for left, right in zip(direction, residual)) / denominator
    return max(0.0, min(1.0, alpha))


def _shape_distance(
    reference: Mapping[str, Any], target: Mapping[str, Any]
) -> float:
    values: list[float] = []
    for key in ("batch", "sequence", "image_size"):
        if reference.get(key) is not None and target.get(key) is not None:
            ref = float(reference[key])
            dst = float(target[key])
            if ref > 0.0 and dst > 0.0:
                values.append(math.log2(ref) - math.log2(dst))
    for key in ("memory_budget",):
        if reference.get(key) is not None and target.get(key) is not None:
            values.append(float(reference[key]) - float(target[key]))
    return math.sqrt(sum(value * value for value in values))


def fit_reference_trajectory_calibration(
    references: Sequence[Mapping[str, Any]],
    *,
    target_shape: Mapping[str, Any],
    grid_size: int = 401,
    total_key: str | None = None,
) -> dict[str, Any]:
    """Fit a leakage-free normalized trajectory template from references.

    A reference may provide ``events`` and optionally ``scheduler_events``.
    When the latter is present, the nearest/second-nearest transfer is used to
    fit a conservative scheduler/template blend. Target measurements are not
    accepted or needed.
    """

    if len(references) < 1:
        raise ValueError("at least one independent reference is required")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    grid = [index / float(grid_size - 1) for index in range(grid_size)]
    prepared: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        events = list(reference.get("events") or ())
        if not events:
            raise ValueError(f"reference {index} has no events")
        key = total_key or _total_key(events)
        templates = {
            phase: _normalize(_sample_by_progress(events, phase, grid, key))
            for phase in PHASES
        }
        scheduler_events = list(reference.get("scheduler_events") or ())
        scheduler_templates = {
            phase: _normalize(
                _sample_by_progress(scheduler_events, phase, grid, key)
            )
            if scheduler_events
            else list(templates[phase])
            for phase in PHASES
        }
        prepared.append(
            {
                "reference_index": index,
                "events": events,
                "shape": dict(reference),
                "templates": templates,
                "scheduler_templates": scheduler_templates,
                "distance_to_target": _shape_distance(reference, target_shape),
            }
        )
    prepared.sort(
        key=lambda row: (row["distance_to_target"], row["reference_index"])
    )
    nearest = prepared[0]
    source = prepared[1] if len(prepared) > 1 else nearest
    alphas: list[float] = []
    phase_diagnostics: dict[str, Any] = {}
    for phase in PHASES:
        scheduler = nearest["scheduler_templates"][phase]
        target = nearest["templates"][phase]
        source_template = source["templates"][phase]
        if (
            max(target, default=0.0) <= min(target, default=0.0)
            and max(scheduler, default=0.0) <= min(scheduler, default=0.0)
        ):
            phase_diagnostics[phase] = {"status": "flat_or_missing"}
            continue
        alpha = _fit_alpha(source_template, scheduler, target)
        alphas.append(alpha)
        blended = [
            alpha * s + (1.0 - alpha) * q
            for s, q in zip(source_template, scheduler)
        ]
        phase_diagnostics[phase] = {
            "status": "fitted",
            "alpha": alpha,
            "scheduler_rmse": math.sqrt(
                sum((q - t) ** 2 for q, t in zip(scheduler, target))
                / len(grid)
            ),
            "source_rmse": math.sqrt(
                sum((s - t) ** 2 for s, t in zip(source_template, target))
                / len(grid)
            ),
            "blended_rmse": math.sqrt(
                sum((b - t) ** 2 for b, t in zip(blended, target))
                / len(grid)
            ),
        }
    alpha = sum(alphas) / len(alphas) if alphas else 1.0
    return {
        "schema_version": 1,
        "method": "peakaware_reference_template_scheduler_blend_v1",
        "claim_scope": "held-out reference-calibrated trajectory prediction",
        "grid": grid,
        "target_shape": dict(target_shape),
        "template_reference_index": nearest["reference_index"],
        "source_reference_index": source["reference_index"],
        "blend_alpha": alpha,
        "phase_templates": nearest["templates"],
        "phase_diagnostics": phase_diagnostics,
        "references": [
            {
                "reference_index": row["reference_index"],
                "distance_to_target": row["distance_to_target"],
                "shape": row["shape"],
            }
            for row in prepared
        ],
        "invariants": {
            "target_trajectory_consumed": False,
            "target_event_identity_preserved": True,
            "target_timestamps_preserved": True,
            "physical_components_preserved": True,
            "signed_residual_explicit": True,
        },
    }


def _physical_component_sum(
    row: Mapping[str, Any], component_fields: Iterable[str]
) -> int:
    if isinstance(row.get("components"), Mapping):
        return int(
            round(
                sum(
                    _number(row["components"], key) for key in component_fields
                )
            )
        )
    return int(round(sum(_number(row, key) for key in component_fields)))


def apply_reference_trajectory_prediction(
    target_events: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    component_fields: Sequence[str] = DEFAULT_COMPONENT_FIELDS,
    total_key: str | None = None,
) -> list[dict[str, Any]]:
    """Apply a fitted envelope while preserving target events and components."""

    if not target_events:
        raise ValueError("target_events must not be empty")
    grid = [float(value) for value in calibration["grid"]]
    alpha = float(calibration["blend_alpha"])
    output: list[dict[str, Any]] = []
    key = total_key or _total_key(target_events)
    for phase in PHASES:
        selected = [dict(row) for row in _phase_rows(target_events, phase)]
        if not selected:
            continue
        raw_values = [_number(row, key) for row in selected]
        low, high = min(raw_values), max(raw_values)
        raw_normalized = _normalize(raw_values)
        template = [float(value) for value in calibration["phase_templates"][phase]]
        scheduler_template = _resample(raw_normalized, len(template))
        blended = [
            max(0.0, min(1.0, alpha * reference + (1.0 - alpha) * scheduler))
            for reference, scheduler in zip(template, scheduler_template)
        ]
        phase_start, phase_end = _time(selected[0]), _time(selected[-1])
        for row in selected:
            progress = (
                0.0
                if phase_end <= phase_start
                else (_time(row) - phase_start) / (phase_end - phase_start)
            )
            progress = max(0.0, min(1.0, progress))
            normalized = _interpolate(grid, blended, progress)
            corrected = int(round(low + normalized * (high - low)))
            row["uncalibrated_current_bytes"] = int(round(_number(row, key)))
            row[key] = corrected
            physical_sum = _physical_component_sum(row, component_fields)
            row["physical_component_sum_bytes"] = physical_sum
            row["trajectory_model_residual_bytes"] = corrected - physical_sum
            row["component_balance_bytes"] = (
                physical_sum + row["trajectory_model_residual_bytes"] - corrected
            )
            row["trajectory_prediction_source"] = (
                "peakaware_reference_template_scheduler_blend_v1"
            )
            row["blend_alpha"] = alpha
            output.append(row)
    output.sort(key=lambda row: _time(row))
    return output


def _resample(values: Sequence[float], size: int) -> list[float]:
    if size <= 0:
        return []
    if len(values) == size:
        return list(values)
    if len(values) == 1:
        return [float(values[0]) for _ in range(size)]
    source_grid = [i / (len(values) - 1) for i in range(len(values))]
    return [
        _interpolate(source_grid, values, i / (size - 1))
        for i in range(size)
    ]


def _interpolate(
    grid: Sequence[float], values: Sequence[float], query: float
) -> float:
    if not values:
        return 0.0
    if query <= grid[0]:
        return float(values[0])
    if query >= grid[-1]:
        return float(values[-1])
    for index in range(len(grid) - 1):
        if grid[index + 1] >= query:
            span = grid[index + 1] - grid[index]
            fraction = 0.0 if span <= 0 else (query - grid[index]) / span
            return float(
                values[index] + fraction * (values[index + 1] - values[index])
            )
    return float(values[-1])


def evaluate_reference_trajectory_prediction(
    predicted: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    predicted_total_key: str = "current_bytes",
    actual_total_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate peak, total duration, and absolute/dynamic trajectory errors."""

    if not predicted or not actual:
        raise ValueError("predicted and actual traces must be non-empty")
    actual_key = actual_total_key or _total_key(actual)
    predicted_rows = sorted(predicted, key=_time)
    actual_rows = sorted(actual, key=_time)
    predicted_peak = max(_number(row, predicted_total_key) for row in predicted_rows)
    actual_peak = max(_number(row, actual_key) for row in actual_rows)
    predicted_end = max(_time(row) for row in predicted_rows)
    actual_end = max(_time(row) for row in actual_rows)
    sampled = []
    for row in actual_rows:
        query = _time(row)
        value = _step_value(predicted_rows, predicted_total_key, query)
        sampled.append((value, _number(row, actual_key)))
    mse = sum((left - right) ** 2 for left, right in sampled) / len(sampled)
    actual_range = actual_peak - min(value for _, value in sampled)
    predicted_min = min(left for left, _ in sampled)
    actual_min = min(right for _, right in sampled)
    predicted_dynamic = [left - predicted_min for left, _ in sampled]
    actual_dynamic = [right - actual_min for _, right in sampled]
    dynamic_mse = sum(
        (left - right) ** 2
        for left, right in zip(predicted_dynamic, actual_dynamic)
    ) / len(sampled)
    return {
        "peak_ape_pct": (
            None
            if actual_peak <= 0
            else abs(predicted_peak - actual_peak) / actual_peak * 100.0
        ),
        "time_ape_pct": (
            None
            if actual_end <= 0
            else abs(predicted_end - actual_end) / actual_end * 100.0
        ),
        "absolute_nrmse_pct": (
            None
            if actual_peak <= 0
            else math.sqrt(mse) / actual_peak * 100.0
        ),
        "dynamic_nrmse_pct": (
            None
            if actual_range <= 0
            else math.sqrt(dynamic_mse) / actual_range * 100.0
        ),
        "predicted_peak_bytes": int(round(predicted_peak)),
        "actual_peak_bytes": int(round(actual_peak)),
        "predicted_total_time_us": predicted_end,
        "actual_total_time_us": actual_end,
        "target_trajectory_consumed_for_prediction": False,
    }


def _step_value(rows: Sequence[Mapping[str, Any]], key: str, query: float) -> float:
    value = _number(rows[0], key)
    for row in rows[1:]:
        if _time(row) > query:
            break
        value = _number(row, key)
    return value
