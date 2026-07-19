from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import experiment_records_from_dicts, write_experiment_effect_acceptance_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PeakAware effect-oriented acceptance metrics.")
    parser.add_argument("records_json", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.records_json:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    records = experiment_records_from_dicts(rows)
    write_experiment_effect_acceptance_json(records, args.output_json)
    print(json.dumps({"record_count": len(records), "output_json": str(args.output_json)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
