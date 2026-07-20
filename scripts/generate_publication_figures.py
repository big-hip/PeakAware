from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import experiment_records_from_dicts
from peakaware.publication.figures import build_publication_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PeakAware publication figure artifact directories.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status", choices=("draft", "provisional", "frozen"), default="draft")
    parser.add_argument("--ev-id", action="append", default=[])
    args = parser.parse_args()

    records_payload = json.loads(args.records_json.read_text(encoding="utf-8"))
    records = experiment_records_from_dicts(records_payload)
    command = " ".join(sys.argv)
    artifacts = build_publication_figures(
        records,
        args.output_root,
        status=args.status,
        ev_ids=tuple(args.ev_id),
        source_paths=(args.records_json,),
        command=command,
    )
    print(
        json.dumps(
            {
                "figure_count": len(artifacts),
                "figures": [
                    {
                        "figure_id": artifact.figure_id,
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
