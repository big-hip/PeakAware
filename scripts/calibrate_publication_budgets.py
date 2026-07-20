from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import experiment_records_from_dicts
from peakaware.publication.budget_calibration import build_budget_plan_from_records


def _ratios(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("ratio list must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a publication physical-budget manifest from all-save reference records."
    )
    parser.add_argument("--from-records-json", type=Path, required=True)
    parser.add_argument("--ratios", type=_ratios, default=(0.50, 0.65, 0.80, 0.95, 1.00))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-status", choices=("draft", "provisional", "frozen"), default="provisional")
    parser.add_argument("--min-reference-count", type=int, default=5)
    args = parser.parse_args()

    records_payload = json.loads(args.from_records_json.read_text(encoding="utf-8"))
    records = experiment_records_from_dicts(records_payload)
    manifest = build_budget_plan_from_records(
        records,
        ratios=args.ratios,
        evidence_status=args.evidence_status,
        min_reference_count=args.min_reference_count,
    )
    manifest["source_records_json"] = str(args.from_records_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evidence_status": manifest["evidence_status"],
                "complete": manifest["complete"],
                "cell_count": manifest["cell_count"],
                "warning_count": manifest["warning_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
