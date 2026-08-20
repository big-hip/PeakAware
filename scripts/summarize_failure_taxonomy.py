from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize failed PeakAware experiment records without importing torch.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records = _load_records(args.records_json)
    payload = summarize_failure_taxonomy(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "failed_record_count": payload["failed_record_count"],
                "category_counts": payload["category_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def summarize_failure_taxonomy(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        if record.get("status") == "ok":
            continue
        error_type = str(record.get("error_type") or "unknown")
        error_message = str(record.get("error_message") or "")
        category = _failure_category(error_type, error_message)
        rows.append(
            {
                "variant_name": record.get("variant_name"),
                "task_name": record.get("task_name"),
                "microbatch_size": record.get("microbatch_size"),
                "budget_bytes": record.get("budget_bytes"),
                "matrix_pass_index": record.get("matrix_pass_index"),
                "matrix_pass_count": record.get("matrix_pass_count"),
                "status": record.get("status"),
                "error_type": error_type,
                "failure_category": category,
                "error_message": error_message,
            }
        )
    return {
        "record_count": len(records),
        "failed_record_count": len(rows),
        "category_counts": _counts(row["failure_category"] for row in rows),
        "error_type_counts": _counts(row["error_type"] for row in rows),
        "task_category_counts": _task_category_counts(rows),
        "rows": rows,
    }


def _failure_category(error_type: str, error_message: str) -> str:
    text = f"{error_type}: {error_message}".lower()
    if "fixed lower bound" in text or "infeasible_by_activation" in text:
        return "fixed_frontier_budget_infeasible"
    if "no top-k candidate passed" in text:
        return "topk_no_candidate_passed_measurement"
    if "out of memory" in text or "cuda oom" in text or "cudaerroroutofmemory" in text:
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "correctness" in text or "gradients_match" in text or "outputs_match" in text:
        return "correctness_failure"
    if "unsupported" in text or "notimplemented" in text:
        return "unsupported"
    if "infeasiblebudgeterror" in text:
        return "budget_or_candidate_infeasible"
    return "other"


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _task_category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_name")), []).append(str(row["failure_category"]))
    return {task: _counts(categories) for task, categories in sorted(grouped.items())}


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("rows", payload))
    if not isinstance(payload, list):
        raise SystemExit("records-json must contain a list or an object with records/rows")
    return [dict(item) for item in payload if isinstance(item, dict)]


if __name__ == "__main__":
    main()
