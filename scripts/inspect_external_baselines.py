from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.external_baselines import (
    summarize_external_baseline_capabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect external PyTorch baseline availability for PeakAware.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = summarize_external_baseline_capabilities()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
