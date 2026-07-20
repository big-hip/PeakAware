from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.budget_calibration import validate_budget_plan_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PeakAware publication physical-budget manifest.")
    parser.add_argument("budget_manifest", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()

    payload = validate_budget_plan_manifest(args.budget_manifest, require_frozen=args.require_frozen)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
