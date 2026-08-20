from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_timeline_readiness import audit_records
from scripts.summarize_selected_regret import summarize_selected_regret


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate current main-experiment readiness from PeakAware records without importing torch."
    )
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selected-regret-json", type=Path, default=None)
    parser.add_argument("--failure-taxonomy-json", type=Path, default=None)
    parser.add_argument("--require-gpu-util", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records_json)
    selected_regret = (
        json.loads(args.selected_regret_json.read_text(encoding="utf-8"))
        if args.selected_regret_json is not None and args.selected_regret_json.is_file()
        else summarize_selected_regret(records)
    )
    payload = evaluate_main_experiment_status(
        records,
        selected_regret=selected_regret,
        failure_taxonomy=(
            json.loads(args.failure_taxonomy_json.read_text(encoding="utf-8"))
            if args.failure_taxonomy_json is not None and args.failure_taxonomy_json.is_file()
            else None
        ),
        require_gpu_util=args.require_gpu_util,
    )
    payload["records_json"] = str(args.records_json)
    payload["selected_regret_json"] = (
        str(args.selected_regret_json) if args.selected_regret_json is not None else None
    )
    payload["failure_taxonomy_json"] = (
        str(args.failure_taxonomy_json) if args.failure_taxonomy_json is not None else None
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "main_question_status": payload["main_question_status"],
                "ok_record_count": payload["record_summary"]["ok_record_count"],
                "budget_violation_count": payload["record_summary"]["budget_violation_count"],
                "selected_best_feasible_rate": payload["selected_regret_summary"]["selected_best_feasible_rate"],
                "timeline_ready": payload["timeline_readiness"]["timeline_ready"],
                "gpu_ready": payload["timeline_readiness"]["gpu_ready"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def evaluate_main_experiment_status(
    records: list[dict[str, Any]],
    *,
    selected_regret: dict[str, Any],
    failure_taxonomy: dict[str, Any] | None = None,
    require_gpu_util: bool = False,
) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    failed_records = [record for record in records if record.get("status") != "ok"]
    budget_violations = [
        record
        for record in ok_records
        if record.get("measured_peak_bytes") is not None
        and int(record["measured_peak_bytes"]) > int(record["budget_bytes"])
    ]
    timeline = audit_records(records, require_gpu_util=require_gpu_util)
    baseline = _baseline_provenance_summary(ok_records)
    measurement = _measurement_protocol_summary(ok_records)
    selected_regret_summary = {
        key: selected_regret.get(key)
        for key in (
            "row_count",
            "regret_observation_count",
            "mean_selected_regret_us",
            "p90_selected_regret_us",
            "max_selected_regret_us",
            "mean_selected_regret_ratio",
            "p90_selected_regret_ratio",
            "selected_best_feasible_rate",
            "selected_measured_feasible_rate",
        )
    }
    blockers = []
    if baseline["proxy_plan_row_count"] > 0 and baseline["real_external_baseline_row_count"] == 0:
        blockers.append("baseline_provenance_proxy_only")
    if not timeline["timeline_ready"]:
        blockers.append("missing_continuous_actual_vs_simulated_timeline")
    if require_gpu_util and not timeline["gpu_ready"]:
        blockers.append("missing_gpu_utilization_trace")
    if measurement["steady_state_record_count"] < len(ok_records):
        blockers.append("measurement_protocol_not_uniformly_steady_state")
    if failed_records and failure_taxonomy is None:
        blockers.append("failed_records_require_taxonomy")
    strengths = []
    if ok_records and not budget_violations:
        strengths.append("no_budget_violation_among_ok_records")
    if selected_regret_summary["selected_measured_feasible_rate"] == 1.0:
        strengths.append("selected_candidates_measured_budget_feasible")
    if (
        selected_regret_summary["selected_best_feasible_rate"] is not None
        and float(selected_regret_summary["selected_best_feasible_rate"]) >= 0.95
    ):
        strengths.append("selected_candidates_near_best_measured_feasible_time")
    if failed_records and failure_taxonomy is not None:
        strengths.append("failed_records_have_taxonomy")
    main_status = "ready_for_final_main_experiment_claim" if not blockers else "partial_evidence_needs_rerun"
    return {
        "schema_version": "0.1",
        "main_question": (
            "Given a PyTorch model and memory budget, does PeakAware more reliably choose a full-step "
            "budget-feasible recomputation plan than saved-memory/min-cut style methods while preserving time?"
        ),
        "main_question_status": main_status,
        "strengths": strengths,
        "blockers": blockers,
        "record_summary": {
            "record_count": len(records),
            "ok_record_count": len(ok_records),
            "failed_record_count": len(failed_records),
            "budget_violation_count": len(budget_violations),
            "budget_violation_rate_among_ok": None
            if not ok_records
            else len(budget_violations) / len(ok_records),
            "task_count": len({record.get("task_name") for record in ok_records}),
            "variant_count": len({record.get("variant_name") for record in ok_records}),
        },
        "measurement_protocol": measurement,
        "baseline_provenance": baseline,
        "selected_regret_summary": selected_regret_summary,
        "failure_taxonomy_summary": None
        if failure_taxonomy is None
        else {
            "failed_record_count": failure_taxonomy.get("failed_record_count"),
            "category_counts": failure_taxonomy.get("category_counts"),
            "error_type_counts": failure_taxonomy.get("error_type_counts"),
        },
        "timeline_readiness": {
            key: timeline.get(key)
            for key in (
                "timeline_ready",
                "timeline_ready_record_count",
                "timeline_ready_rate",
                "gpu_ready",
                "gpu_ready_record_count",
                "gpu_ready_rate",
                "field_counts",
                "gpu_field_counts",
            )
        },
    }


def _baseline_provenance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    plan_counts: dict[str, int] = {}
    proxy_rows = 0
    real_rows = 0
    for record in records:
        for row in record.get("measured_plan_results", ()):
            if not isinstance(row, dict):
                continue
            plan_id = str(row.get("plan_id"))
            plan_counts[plan_id] = plan_counts.get(plan_id, 0) + 1
            provenance = row.get("strategy_provenance") or {}
            source = str(provenance.get("source", "unknown")) if isinstance(provenance, dict) else "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1
            if "proxy" in source:
                proxy_rows += 1
            if source.startswith("real_") or source in {"pytorch_min_cut", "pytorch_sac", "block_ac"}:
                real_rows += 1
    return {
        "plan_counts": dict(sorted(plan_counts.items())),
        "strategy_source_counts": dict(sorted(source_counts.items())),
        "proxy_plan_row_count": proxy_rows,
        "real_external_baseline_row_count": real_rows,
    }


def _measurement_protocol_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    repeats = [int(record["measurement_repeats"]) for record in records if record.get("measurement_repeats") is not None]
    warmups = [
        int(record["measurement_warmup_steps"])
        for record in records
        if record.get("measurement_warmup_steps") is not None
    ]
    steady_state_count = sum(
        1
        for record in records
        if int(record.get("measurement_repeats") or 0) >= 20
        and int(record.get("measurement_warmup_steps") or 0) >= 5
    )
    return {
        "min_measurement_repeats": None if not repeats else min(repeats),
        "min_measurement_warmup_steps": None if not warmups else min(warmups),
        "steady_state_record_count": steady_state_count,
        "steady_state_definition": "measurement_repeats>=20 and measurement_warmup_steps>=5",
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


if __name__ == "__main__":
    main()
