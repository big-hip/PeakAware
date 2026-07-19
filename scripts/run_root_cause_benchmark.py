from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.root_cause_benchmark import (
    run_synthetic_root_cause_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PeakAware synthetic root-cause accuracy benchmark.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = run_synthetic_root_cause_benchmark()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
