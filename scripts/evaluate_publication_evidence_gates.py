from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.evidence import evaluate_publication_evidence_gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PeakAware thesis evidence gates for an artifact root.")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--records-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--workload-manifest", type=Path, default=None)
    parser.add_argument("--budget-manifest", type=Path, default=None)
    parser.add_argument("--qualification-summary", type=Path, default=None)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--fail-on-unmet", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = evaluate_publication_evidence_gates(
        args.artifact_root,
        records_json=args.records_json,
        summary_json=args.summary_json,
        workload_manifest=args.workload_manifest,
        budget_manifest=args.budget_manifest,
        qualification_summary=args.qualification_summary,
        require_frozen=args.require_frozen,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_unmet and not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
