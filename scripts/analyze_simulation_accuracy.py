from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PeakAware simulation accuracy from measured experiment records without importing torch. "
            "This focuses on simulator precision, not Top-K or plan selection quality."
        )
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-outliers", type=int, default=20)
    args = parser.parse_args()

    records = _load_records(args.records_json)
    payload = analyze_simulation_accuracy(records, top_outliers=args.top_outliers)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "candidate_count": payload["candidate_count"],
                "raw_memory_mape": payload["memory"]["raw"]["mape"],
                "calibrated_memory_mape": payload["memory"]["calibrated"]["mape"],
                "raw_time_mape": payload["time"]["raw_costmodel"]["mape"],
                "calibrated_time_mape": payload["time"]["all_save_scale_calibrated"]["mape"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def analyze_simulation_accuracy(records: list[dict[str, Any]], *, top_outliers: int = 20) -> dict[str, Any]:
    rows = _candidate_rows(records)
    raw_memory = [row for row in rows if row.get("raw_memory_abs_pct_error") is not None]
    calibrated_memory = [row for row in rows if row.get("calibrated_memory_abs_pct_error") is not None]
    raw_time = [row for row in rows if row.get("raw_time_abs_pct_error") is not None]
    calibrated_time = [row for row in rows if row.get("calibrated_time_abs_pct_error") is not None]
    return {
        "schema_version": "0.1",
        "record_count": len(records),
        "ok_record_count": sum(1 for record in records if record.get("status") == "ok"),
        "candidate_count": len(rows),
        "candidate_with_measured_memory_count": len(raw_memory),
        "candidate_with_measured_time_count": len(raw_time),
        "calibration_notes": {
            "memory_calibration": (
                "Uses calibrated_estimated_peak_bytes when present. In current records this is usually all-save "
                "residual/phase calibration, not a pure static liveness estimate."
            ),
            "time_calibration": (
                "Uses per-record all-save measured/estimated step-time scale when an all_save measured candidate "
                "exists. This is a coarse phase-agnostic calibration for simulator diagnosis."
            ),
            "denominator": "percentage errors use measured value as denominator",
        },
        "memory": {
            "raw": _metric_summary(raw_memory, "raw_memory_pct_error", "raw_memory_abs_pct_error"),
            "calibrated": _metric_summary(
                calibrated_memory,
                "calibrated_memory_pct_error",
                "calibrated_memory_abs_pct_error",
            ),
            "raw_underestimate_count": sum(1 for row in raw_memory if row["raw_memory_error_bytes"] > 0),
            "calibrated_underestimate_count": sum(
                1 for row in calibrated_memory if row["calibrated_memory_error_bytes"] > 0
            ),
            "raw_unsafe_false_positive_count": sum(1 for row in raw_memory if row["raw_memory_unsafe_false_positive"]),
            "calibrated_unsafe_false_positive_count": sum(
                1 for row in calibrated_memory if row["calibrated_memory_unsafe_false_positive"]
            ),
            "phase_match_rate": _rate(
                sum(1 for row in raw_memory if row.get("memory_phase_match") is True),
                sum(1 for row in raw_memory if row.get("memory_phase_match") is not None),
            ),
        },
        "time": {
            "raw_costmodel": _metric_summary(raw_time, "raw_time_pct_error", "raw_time_abs_pct_error"),
            "all_save_scale_calibrated": _metric_summary(
                calibrated_time,
                "calibrated_time_pct_error",
                "calibrated_time_abs_pct_error",
            ),
            "raw_rank": _rank_summary(raw_time, "estimated_step_us", "measured_step_us"),
            "calibrated_rank": _rank_summary(calibrated_time, "calibrated_estimated_step_us", "measured_step_us"),
        },
        "by_task": _grouped(rows, "task_name"),
        "by_plan": _grouped(rows, "plan_id"),
        "outliers": {
            "raw_memory": _outliers(raw_memory, "raw_memory_abs_pct_error", top_outliers),
            "calibrated_memory": _outliers(calibrated_memory, "calibrated_memory_abs_pct_error", top_outliers),
            "raw_time": _outliers(raw_time, "raw_time_abs_pct_error", top_outliers),
            "calibrated_time": _outliers(calibrated_time, "calibrated_time_abs_pct_error", top_outliers),
        },
        "rows": rows,
    }


def _candidate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if record.get("status") != "ok":
            continue
        time_scale = _all_save_time_scale(record)
        for candidate_index, candidate in enumerate(record.get("measured_plan_results", ())):
            if not isinstance(candidate, dict):
                continue
            row = _candidate_row(record, candidate, record_index, candidate_index, time_scale)
            if row is not None:
                rows.append(row)
    return rows


def _candidate_row(
    record: dict[str, Any],
    candidate: dict[str, Any],
    record_index: int,
    candidate_index: int,
    time_scale: float | None,
) -> dict[str, Any] | None:
    measured_peak = _optional_float(candidate.get("measured_peak_bytes"))
    estimated_peak = _optional_float(candidate.get("estimated_peak_bytes"))
    calibrated_peak = _optional_float(candidate.get("calibrated_estimated_peak_bytes"))
    measured_step = _optional_float(candidate.get("measured_step_us"))
    estimated_step = _optional_float(candidate.get("estimated_step_us"))
    if candidate.get("plan_id") == record.get("selected_plan_id"):
        trace_peak, trace_step = _selected_event_trace_estimate(record)
        if estimated_peak is None:
            estimated_peak = trace_peak
        if estimated_step is None:
            estimated_step = trace_step
    calibrated_step = None if estimated_step is None or time_scale is None else estimated_step * time_scale
    if measured_peak is None and measured_step is None:
        return None

    budget = _optional_float(record.get("budget_bytes"))
    raw_memory_error = None if measured_peak is None or estimated_peak is None else measured_peak - estimated_peak
    calibrated_memory_error = None if measured_peak is None or calibrated_peak is None else measured_peak - calibrated_peak
    raw_time_error = None if measured_step is None or estimated_step is None else measured_step - estimated_step
    calibrated_time_error = None if measured_step is None or calibrated_step is None else measured_step - calibrated_step
    estimated_phase = candidate.get("estimated_peak_phase") or candidate.get("all_save_phase_calibrated_estimated_peak_phase")
    measured_phase = candidate.get("measured_peak_phase")
    return {
        "record_index": record_index,
        "candidate_index": candidate_index,
        "task_name": record.get("task_name"),
        "variant_name": record.get("variant_name"),
        "microbatch_size": record.get("microbatch_size"),
        "budget_bytes": record.get("budget_bytes"),
        "matrix_pass_index": record.get("matrix_pass_index"),
        "plan_id": candidate.get("plan_id"),
        "strategy_source": _strategy_source(candidate),
        "measured_peak_bytes": _optional_int(candidate.get("measured_peak_bytes")),
        "estimated_peak_bytes": _optional_int(estimated_peak),
        "calibrated_estimated_peak_bytes": _optional_int(candidate.get("calibrated_estimated_peak_bytes")),
        "raw_memory_error_bytes": _optional_int(raw_memory_error),
        "calibrated_memory_error_bytes": _optional_int(calibrated_memory_error),
        "raw_memory_pct_error": _pct_error(raw_memory_error, measured_peak),
        "raw_memory_abs_pct_error": _abs_pct_error(raw_memory_error, measured_peak),
        "calibrated_memory_pct_error": _pct_error(calibrated_memory_error, measured_peak),
        "calibrated_memory_abs_pct_error": _abs_pct_error(calibrated_memory_error, measured_peak),
        "raw_memory_unsafe_false_positive": _unsafe_false_positive(estimated_peak, measured_peak, budget),
        "calibrated_memory_unsafe_false_positive": _unsafe_false_positive(calibrated_peak, measured_peak, budget),
        "estimated_peak_phase": estimated_phase,
        "measured_peak_phase": measured_phase,
        "memory_phase_match": None
        if estimated_phase is None or measured_phase is None
        else str(estimated_phase) == str(measured_phase),
        "measured_step_us": measured_step,
        "estimated_step_us": estimated_step,
        "time_scale_from_all_save": time_scale,
        "calibrated_estimated_step_us": calibrated_step,
        "raw_time_error_us": raw_time_error,
        "calibrated_time_error_us": calibrated_time_error,
        "raw_time_pct_error": _pct_error(raw_time_error, measured_step),
        "raw_time_abs_pct_error": _abs_pct_error(raw_time_error, measured_step),
        "calibrated_time_pct_error": _pct_error(calibrated_time_error, measured_step),
        "calibrated_time_abs_pct_error": _abs_pct_error(calibrated_time_error, measured_step),
    }


def _selected_event_trace_estimate(
    record: dict[str, Any],
) -> tuple[float | None, float | None]:
    trace = (
        record.get("selected_simulated_memory_event_trace")
        or record.get("selected_lowered_fx_l2_simulated_memory_event_trace")
        or ()
    )
    rows = tuple(
        row
        for row in trace
        if isinstance(row, dict)
        and row.get("bytes") is not None
        and row.get("time_us") is not None
    )
    if not rows:
        return None, None
    times = tuple(float(row["time_us"]) for row in rows)
    return (
        max(float(row["bytes"]) for row in rows),
        max(times) - min(times),
    )


def _all_save_time_scale(record: dict[str, Any]) -> float | None:
    for candidate in record.get("measured_plan_results", ()):
        if not isinstance(candidate, dict) or candidate.get("plan_id") != "all_save":
            continue
        measured = _optional_float(candidate.get("measured_step_us"))
        estimated = _optional_float(candidate.get("estimated_step_us"))
        if measured is not None and estimated is not None and estimated > 0.0:
            return measured / estimated
    return None


def _metric_summary(rows: list[dict[str, Any]], signed_key: str, abs_key: str) -> dict[str, Any]:
    signed = [float(row[signed_key]) for row in rows if row.get(signed_key) is not None]
    absolute = [float(row[abs_key]) for row in rows if row.get(abs_key) is not None]
    return {
        "count": len(absolute),
        "bias": _mean(signed),
        "mape": _mean(absolute),
        "p50_ape": _percentile(absolute, 0.50),
        "p90_ape": _percentile(absolute, 0.90),
        "p95_ape": _percentile(absolute, 0.95),
        "max_ape": None if not absolute else max(absolute),
        "within_10_percent_rate": _rate(sum(1 for value in absolute if value <= 0.10), len(absolute)),
        "within_20_percent_rate": _rate(sum(1 for value in absolute if value <= 0.20), len(absolute)),
    }


def _rank_summary(rows: list[dict[str, Any]], estimated_key: str, measured_key: str) -> dict[str, Any]:
    pairs = [
        (float(row[estimated_key]), float(row[measured_key]))
        for row in rows
        if row.get(estimated_key) is not None and row.get(measured_key) is not None
    ]
    if len(pairs) < 2:
        return {"count": len(pairs), "spearman": None, "kendall": None, "kendall_pair_count": 0}
    estimated = [item[0] for item in pairs]
    measured = [item[1] for item in pairs]
    kendall, pair_count = _kendall_tau_a(estimated, measured)
    return {
        "count": len(pairs),
        "spearman": _pearson(_average_ranks(estimated), _average_ranks(measured)),
        "kendall": kendall,
        "kendall_pair_count": pair_count,
    }


def _grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return {
        name: {
            "candidate_count": len(items),
            "raw_memory_mape": _metric_summary(
                [row for row in items if row.get("raw_memory_abs_pct_error") is not None],
                "raw_memory_pct_error",
                "raw_memory_abs_pct_error",
            )["mape"],
            "calibrated_memory_mape": _metric_summary(
                [row for row in items if row.get("calibrated_memory_abs_pct_error") is not None],
                "calibrated_memory_pct_error",
                "calibrated_memory_abs_pct_error",
            )["mape"],
            "raw_time_mape": _metric_summary(
                [row for row in items if row.get("raw_time_abs_pct_error") is not None],
                "raw_time_pct_error",
                "raw_time_abs_pct_error",
            )["mape"],
            "calibrated_time_mape": _metric_summary(
                [row for row in items if row.get("calibrated_time_abs_pct_error") is not None],
                "calibrated_time_pct_error",
                "calibrated_time_abs_pct_error",
            )["mape"],
        }
        for name, items in sorted(grouped.items())
    }


def _outliers(rows: list[dict[str, Any]], key: str, top_k: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=True)[: max(top_k, 0)]
    keep = (
        "task_name",
        "variant_name",
        "budget_bytes",
        "matrix_pass_index",
        "plan_id",
        "strategy_source",
        "measured_peak_bytes",
        "estimated_peak_bytes",
        "calibrated_estimated_peak_bytes",
        "measured_step_us",
        "estimated_step_us",
        "calibrated_estimated_step_us",
        "raw_memory_abs_pct_error",
        "calibrated_memory_abs_pct_error",
        "raw_time_abs_pct_error",
        "calibrated_time_abs_pct_error",
    )
    return [{field: row.get(field) for field in keep} for row in ordered]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _strategy_source(candidate: dict[str, Any]) -> str:
    provenance = candidate.get("strategy_provenance")
    if isinstance(provenance, dict):
        return str(provenance.get("source", "unknown"))
    return "unknown"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _pct_error(error: float | None, measured: float | None) -> float | None:
    if error is None or measured is None or measured == 0.0:
        return None
    return error / measured


def _abs_pct_error(error: float | None, measured: float | None) -> float | None:
    value = _pct_error(error, measured)
    return None if value is None else abs(value)


def _unsafe_false_positive(estimated: float | None, measured: float | None, budget: float | None) -> bool:
    return estimated is not None and measured is not None and budget is not None and estimated <= budget < measured


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        for original, _value in indexed[index:end]:
            ranks[original] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_denominator = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_denominator * right_denominator
    return None if denominator == 0.0 else numerator / denominator


def _kendall_tau_a(left: list[float], right: list[float]) -> tuple[float | None, int]:
    concordant = 0
    discordant = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            left_delta = left[i] - left[j]
            right_delta = right[i] - right[j]
            if left_delta == 0.0 or right_delta == 0.0:
                continue
            if left_delta * right_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    pair_count = concordant + discordant
    if pair_count == 0:
        return None, 0
    return (concordant - discordant) / pair_count, pair_count


if __name__ == "__main__":
    main()
