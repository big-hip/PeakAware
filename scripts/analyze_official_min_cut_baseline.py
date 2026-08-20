from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.min_cut_analysis import (
    analyze_official_min_cut,
    load_jsonl_records,
    write_analysis_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze qualified official PyTorch AOT min-cut Memory Budget records."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Qualification artifact directories or records.jsonl files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-ratio", type=float, default=1.0)
    args = parser.parse_args()

    records = load_jsonl_records(args.inputs)
    payload = analyze_official_min_cut(records, baseline_ratio=args.baseline_ratio)
    outputs = write_analysis_outputs(payload, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
