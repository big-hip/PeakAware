from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.cost.calibration import write_residual_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a residual peak calibration report from PeakAware records.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--key-fields",
        default="task_name",
        help="Comma-separated record fields used as residual calibration key.",
    )
    parser.add_argument("--holdout-field", default=None)
    parser.add_argument("--holdout-value", default=None)
    args = parser.parse_args()
    key_fields = tuple(field.strip() for field in args.key_fields.split(",") if field.strip())
    report = write_residual_calibration_report(
        args.records_json,
        args.output_json,
        key_fields=key_fields,
        holdout_field=args.holdout_field,
        holdout_value=args.holdout_value,
    )
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
