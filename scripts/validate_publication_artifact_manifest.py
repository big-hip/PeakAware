from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.artifact import validate_publication_artifact_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PeakAware publication manifest and checksum file.")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--checksums", type=Path, default=None)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()

    payload = validate_publication_artifact_manifest(
        args.artifact_root,
        manifest_json=args.manifest_json,
        checksums_path=args.checksums,
        require_frozen=args.require_frozen,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
