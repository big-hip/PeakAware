from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import experiment_records_from_dicts
from peakaware.publication.tables import build_publication_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PeakAware publication table artifact directories.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status", choices=("draft", "provisional", "frozen"), default="draft")
    args = parser.parse_args()

    records = experiment_records_from_dicts(json.loads(args.records_json.read_text(encoding="utf-8")))
    artifacts = build_publication_tables(
        records,
        args.output_root,
        status=args.status,
        source_paths=(args.records_json,),
    )
    print(
        json.dumps(
            {
                "table_count": len(artifacts),
                "tables": [
                    {
                        "table_id": artifact.table_id,
                        "output_dir": str(artifact.output_dir),
                        "row_count": artifact.row_count,
                        "status": artifact.status,
                    }
                    for artifact in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
