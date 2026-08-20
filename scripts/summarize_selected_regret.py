from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive selected step-time regret from PeakAware records without importing torch."
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records = _load_records(args.records_json)
    payload = summarize_selected_regret(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "row_count": payload["row_count"],
                "regret_observation_count": payload["regret_observation_count"],
                "mean_selected_regret_ratio": payload["mean_selected_regret_ratio"],
                "selected_best_feasible_rate": payload["selected_best_feasible_rate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def summarize_selected_regret(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok" or record.get("selected_plan_id") is None:
            continue
        budget_bytes = int(record["budget_bytes"])
        measured_plan_results = [
            dict(row) for row in record.get("measured_plan_results", ()) if isinstance(row, dict)
        ]
        selected_plan_id = str(record["selected_plan_id"])
        candidates = [
            row
            for row in measured_plan_results
            if row.get("measured_peak_bytes") is not None
            and row.get("measured_step_us") is not None
            and int(row["measured_peak_bytes"]) <= budget_bytes
        ]
        selected_row = next((row for row in measured_plan_results if row.get("plan_id") == selected_plan_id), None)
        if selected_row is None or selected_row.get("measured_step_us") is None:
            continue
        selected_step_us = float(selected_row["measured_step_us"])
        selected_peak_bytes = _optional_int(selected_row.get("measured_peak_bytes"))
        selected_feasible = selected_peak_bytes is not None and selected_peak_bytes <= budget_bytes
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    float(row["measured_step_us"]),
                    int(row["measured_peak_bytes"]),
                    str(row.get("plan_id")),
                ),
            )
            best_step_us = float(best["measured_step_us"])
            best_plan_id = str(best.get("plan_id"))
            best_peak_bytes = int(best["measured_peak_bytes"])
            if selected_feasible:
                regret_us = selected_step_us - best_step_us
                regret_ratio = regret_us / best_step_us if best_step_us > 0.0 else None
            else:
                regret_us = None
                regret_ratio = None
        else:
            best_step_us = None
            best_plan_id = None
            best_peak_bytes = None
            regret_us = None
            regret_ratio = None
        rows.append(
            {
                "variant_name": record.get("variant_name"),
                "task_name": record.get("task_name"),
                "microbatch_size": record.get("microbatch_size"),
                "budget_bytes": budget_bytes,
                "selected_plan_id": selected_plan_id,
                "selected_measured_peak_bytes": selected_peak_bytes,
                "selected_measured_step_us": selected_step_us,
                "selected_measured_feasible": selected_feasible,
                "best_feasible_plan_id": best_plan_id,
                "best_feasible_peak_bytes": best_peak_bytes,
                "best_feasible_step_us": best_step_us,
                "candidate_feasible_count": len(candidates),
                "candidate_measured_count": len(measured_plan_results),
                "selected_regret_us": regret_us,
                "selected_regret_ratio": regret_ratio,
                "selected_is_best_feasible": None
                if best_plan_id is None
                else selected_plan_id == best_plan_id,
                "matrix_pass_index": record.get("matrix_pass_index"),
                "matrix_pass_count": record.get("matrix_pass_count"),
            }
        )
    regret_values = [float(row["selected_regret_us"]) for row in rows if row["selected_regret_us"] is not None]
    regret_ratios = [
        float(row["selected_regret_ratio"])
        for row in rows
        if row["selected_regret_ratio"] is not None
    ]
    best_matches = [
        bool(row["selected_is_best_feasible"])
        for row in rows
        if row["selected_is_best_feasible"] is not None
    ]
    feasible_selected = [bool(row["selected_measured_feasible"]) for row in rows]
    return {
        "row_count": len(rows),
        "regret_observation_count": len(regret_values),
        "mean_selected_regret_us": _mean(regret_values),
        "p50_selected_regret_us": _percentile(regret_values, 0.50),
        "p90_selected_regret_us": _percentile(regret_values, 0.90),
        "max_selected_regret_us": None if not regret_values else max(regret_values),
        "mean_selected_regret_ratio": _mean(regret_ratios),
        "p50_selected_regret_ratio": _percentile(regret_ratios, 0.50),
        "p90_selected_regret_ratio": _percentile(regret_ratios, 0.90),
        "selected_best_feasible_rate": None
        if not best_matches
        else sum(1 for value in best_matches if value) / len(best_matches),
        "selected_measured_feasible_rate": None
        if not feasible_selected
        else sum(1 for value in feasible_selected if value) / len(feasible_selected),
        "rows": rows,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _mean(values: list[float | int]) -> float | None:
    return None if not values else sum(float(value) for value in values) / len(values)


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


if __name__ == "__main__":
    main()
