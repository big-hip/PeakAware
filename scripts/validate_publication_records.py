from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.records import validate_publication_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PeakAware publication experiment records.")
    parser.add_argument("records_json", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--require-runtime-identity", action="store_true")
    args = parser.parse_args()

    payload = validate_publication_records(
        args.records_json,
        require_frozen=args.require_frozen,
        require_runtime_identity=args.require_runtime_identity,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
