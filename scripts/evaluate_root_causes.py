from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.diagnostics import (
    RootCauseGroundTruth,
    RootCausePrediction,
    evaluate_root_cause_predictions,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("plan_diagnostics"), list):
        return payload["plan_diagnostics"]
    if isinstance(payload, dict) and isinstance(payload.get("diagnostic"), dict):
        diagnostic = payload["diagnostic"]
        return [
            {
                "plan_id": payload.get("selected_plan_id", diagnostic.get("plan_id", "selected")),
                "primary_cause": diagnostic["primary_cause"],
                "root_causes": (diagnostic["primary_cause"], *diagnostic.get("secondary_causes", ())),
            }
        ]
    raise ValueError("predictions must be a list, a PeakAware summary, or a dict with plan_diagnostics")


def _label_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        return payload["labels"]
    raise ValueError("labels must be a list or a dict with a labels list")


def _prediction_from_row(row: dict[str, Any]) -> RootCausePrediction:
    root_causes = tuple(row.get("root_causes") or (row["primary_cause"], *row.get("secondary_causes", ())))
    return RootCausePrediction(
        plan_id=row["plan_id"],
        primary_cause=row["primary_cause"],
        root_causes=root_causes,
    )


def _label_from_row(row: dict[str, Any]) -> RootCauseGroundTruth:
    return RootCauseGroundTruth(
        plan_id=row["plan_id"],
        primary_cause=row["primary_cause"],
        root_causes=tuple(row["root_causes"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PeakAware root-cause predictions against labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    predictions = tuple(_prediction_from_row(row) for row in _prediction_rows(_load_json(args.predictions)))
    labels = tuple(_label_from_row(row) for row in _label_rows(_load_json(args.labels)))
    evaluation = evaluate_root_cause_predictions(predictions, labels)
    text = json.dumps(asdict(evaluation), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
