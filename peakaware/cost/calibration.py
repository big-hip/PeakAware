from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResidualRule:
    key: tuple[str, ...]
    residual_bytes: int
    sample_count: int
    mean_raw_ape: float | None
    mean_calibrated_ape: float | None


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(round(statistics.median(values)))


def _ape(estimated: int, measured: int) -> float | None:
    if measured <= 0:
        return None
    return abs(estimated - measured) / measured


def _record_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "unknown")) for field in fields)


def _iter_plan_rows(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        for row in record.get("measured_plan_results") or ():
            estimated = row.get("estimated_peak_bytes")
            measured = row.get("measured_peak_bytes")
            if estimated is None or measured is None:
                continue
            rows.append((record, row))
    return rows


def fit_residual_rules(
    records: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...] = ("task_name",),
) -> tuple[ResidualRule, ...]:
    grouped: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for record, row in _iter_plan_rows(records):
        key = _record_key(record, key_fields)
        estimated = int(row["estimated_peak_bytes"])
        measured = int(row["measured_peak_bytes"])
        grouped.setdefault(key, []).append((estimated, measured))

    rules: list[ResidualRule] = []
    for key, pairs in sorted(grouped.items()):
        residual = _median_int([measured - estimated for estimated, measured in pairs])
        raw_apes = [_ape(estimated, measured) for estimated, measured in pairs]
        calibrated_apes = [_ape(estimated + residual, measured) for estimated, measured in pairs]
        raw = [item for item in raw_apes if item is not None]
        calibrated = [item for item in calibrated_apes if item is not None]
        rules.append(
            ResidualRule(
                key=key,
                residual_bytes=residual,
                sample_count=len(pairs),
                mean_raw_ape=None if not raw else statistics.fmean(raw),
                mean_calibrated_ape=None if not calibrated else statistics.fmean(calibrated),
            )
        )
    return tuple(rules)


def evaluate_residual_rules(
    records: list[dict[str, Any]],
    rules: tuple[ResidualRule, ...],
    *,
    key_fields: tuple[str, ...] = ("task_name",),
) -> dict[str, Any]:
    rule_by_key = {rule.key: rule for rule in rules}
    rows = _iter_plan_rows(records)
    raw_apes: list[float] = []
    calibrated_apes: list[float] = []
    improved = 0
    worsened = 0
    covered = 0
    for record, row in rows:
        estimated = int(row["estimated_peak_bytes"])
        measured = int(row["measured_peak_bytes"])
        raw = _ape(estimated, measured)
        if raw is None:
            continue
        raw_apes.append(raw)
        rule = rule_by_key.get(_record_key(record, key_fields))
        calibrated_estimated = estimated if rule is None else estimated + rule.residual_bytes
        if rule is not None:
            covered += 1
        calibrated = _ape(calibrated_estimated, measured)
        if calibrated is None:
            continue
        calibrated_apes.append(calibrated)
        if calibrated < raw:
            improved += 1
        elif calibrated > raw:
            worsened += 1
    return {
        "row_count": len(rows),
        "covered_row_count": covered,
        "mean_raw_ape": None if not raw_apes else statistics.fmean(raw_apes),
        "mean_calibrated_ape": None if not calibrated_apes else statistics.fmean(calibrated_apes),
        "p90_raw_ape": _percentile(raw_apes, 0.90),
        "p90_calibrated_ape": _percentile(calibrated_apes, 0.90),
        "within_10_raw_rate": None if not raw_apes else sum(1 for item in raw_apes if item <= 0.10) / len(raw_apes),
        "within_10_calibrated_rate": None
        if not calibrated_apes
        else sum(1 for item in calibrated_apes if item <= 0.10) / len(calibrated_apes),
        "improved_row_count": improved,
        "worsened_row_count": worsened,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def build_residual_calibration_report(
    records: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...] = ("task_name",),
    holdout_field: str | None = None,
    holdout_value: str | None = None,
) -> dict[str, Any]:
    train_records = records
    eval_records = records
    split: dict[str, Any] | None = None
    if holdout_field is not None and holdout_value is not None:
        train_records = [record for record in records if str(record.get(holdout_field)) != str(holdout_value)]
        eval_records = [record for record in records if str(record.get(holdout_field)) == str(holdout_value)]
        split = {
            "holdout_field": holdout_field,
            "holdout_value": holdout_value,
            "train_record_count": len(train_records),
            "eval_record_count": len(eval_records),
        }
    rules = fit_residual_rules(train_records, key_fields=key_fields)
    evaluation = evaluate_residual_rules(eval_records, rules, key_fields=key_fields)
    return {
        "schema_version": "0.1",
        "calibration_kind": "median_peak_residual",
        "key_fields": list(key_fields),
        "split": split,
        "rule_count": len(rules),
        "rules": [
            {
                "key": list(rule.key),
                "residual_bytes": rule.residual_bytes,
                "sample_count": rule.sample_count,
                "mean_raw_ape": rule.mean_raw_ape,
                "mean_calibrated_ape": rule.mean_calibrated_ape,
            }
            for rule in rules
        ],
        "evaluation": evaluation,
    }


def write_residual_calibration_report(
    records_json: Path,
    output_json: Path,
    *,
    key_fields: tuple[str, ...] = ("task_name",),
    holdout_field: str | None = None,
    holdout_value: str | None = None,
) -> dict[str, Any]:
    records = json.loads(records_json.read_text(encoding="utf-8"))
    report = build_residual_calibration_report(
        records,
        key_fields=key_fields,
        holdout_field=holdout_field,
        holdout_value=holdout_value,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
