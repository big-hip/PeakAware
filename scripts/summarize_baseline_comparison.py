from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import experiment_records_from_dicts, write_experiment_baseline_comparison_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PeakAware baseline comparisons from saved records.")
    parser.add_argument("records_json", type=Path)
    parser.add_argument("--sac-baseline-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records_payload = json.loads(args.records_json.read_text(encoding="utf-8"))
    records = experiment_records_from_dicts(records_payload)
    sac_baseline = None
    if args.sac_baseline_json is not None:
        sac_baseline = json.loads(args.sac_baseline_json.read_text(encoding="utf-8"))
    write_experiment_baseline_comparison_json(records, args.output_json, sac_baseline=sac_baseline)
    print(args.output_json)


if __name__ == "__main__":
    main()
