from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.artifact import build_publication_artifact_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PeakAware publication artifact manifest and checksums.")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=("smoke", "full", "release", "draft"), default="draft")
    parser.add_argument(
        "--evidence-status",
        choices=("smoke", "draft", "provisional", "frozen", "invalid"),
        default="draft",
    )
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--checksums", type=Path, default=None)
    parser.add_argument("--known-limitation", action="append", default=[])
    args = parser.parse_args()

    manifest = build_publication_artifact_manifest(
        args.artifact_root,
        run_id=args.run_id,
        scope=args.scope,
        evidence_status=args.evidence_status,
        output_manifest=args.manifest_json,
        output_checksums=args.checksums,
        known_limitations=tuple(args.known_limitation),
    )
    print(
        json.dumps(
            {
                "artifact_root": str(args.artifact_root),
                "run_id": manifest["run_id"],
                "scope": manifest["scope"],
                "evidence_status": manifest["evidence_status"],
                "file_count": len(manifest["files"]),
                "manifest_json": str(args.manifest_json or args.artifact_root / "manifest.json"),
                "checksums": str(args.checksums or args.artifact_root / "checksums.sha256"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
