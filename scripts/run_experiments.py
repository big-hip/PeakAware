from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.config import PeakAwareConfig
from peakaware.experiments import (
    experiment_records_to_dicts,
    run_experiment_matrix,
    write_experiment_csv,
    write_experiment_json,
)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item)


def _parse_csv_text(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PeakAware experiment matrix.")
    parser.add_argument("--tasks", default="tiny_residual_w8")
    parser.add_argument("--budget-mib", default="256")
    parser.add_argument("--microbatches", default="1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    config = PeakAwareConfig(
        safety_margin_bytes=0,
        safety_margin_ratio=0.0,
        top_k=args.top_k,
        isolate_candidate_measurement=args.isolate,
        profile_db_path=args.profile_db,
    )
    records = run_experiment_matrix(
        task_names=_parse_csv_text(args.tasks),
        microbatch_sizes=_parse_csv_ints(args.microbatches),
        budget_bytes=tuple(value << 20 for value in _parse_csv_ints(args.budget_mib)),
        config=config,
    )
    if args.output_json is not None:
        write_experiment_json(records, args.output_json)
    if args.output_csv is not None:
        write_experiment_csv(records, args.output_csv)
    print(json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
