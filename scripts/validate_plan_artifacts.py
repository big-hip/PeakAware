from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.reporting import load_plan_artifact_json, validate_plan_artifact_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PeakAware plan artifact identity keys.")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in args.artifacts:
        result = validate_plan_artifact_identity(load_plan_artifact_json(path))
        rows.append({"path": str(path), **result})
    payload = {
        "artifact_count": len(rows),
        "valid_count": sum(1 for row in rows if row["valid"]),
        "invalid_count": sum(1 for row in rows if not row["valid"]),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["invalid_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
