from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.figures import validate_publication_figure_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PeakAware publication figure artifact directories.")
    parser.add_argument("figure_root", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()

    payload = validate_publication_figure_artifacts(args.figure_root, require_frozen=args.require_frozen)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
