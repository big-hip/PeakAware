from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.publication.matrix import run_publication_matrix_from_budget_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PeakAware publication matrix from a budget manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Budget manifest JSON.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-hints", choices=("on", "off", "both"), default="both")
    parser.add_argument("--matrix-passes", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--measurement-warmup-steps", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument(
        "--selection-objective",
        choices=("min_peak_then_time", "min_time_then_peak"),
        default="min_peak_then_time",
    )
    parser.add_argument("--require-frozen-budget", action="store_true")
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--limit-budgets", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override device from the budget manifest.")
    parser.add_argument("--plan-artifact-dir", type=Path, default=None)
    parser.add_argument("--figure-status", choices=("draft", "provisional", "frozen"), default="draft")
    parser.add_argument("--ev-id", action="append", default=[])
    args = parser.parse_args()

    result = run_publication_matrix_from_budget_plan(
        args.manifest,
        args.output_root,
        diagnostic_hints=args.diagnostic_hints,
        matrix_passes=args.matrix_passes,
        top_k=args.top_k,
        measurement_warmup_steps=args.measurement_warmup_steps,
        measurement_repeats=args.measurement_repeats,
        selection_objective=args.selection_objective,
        require_frozen_budget=args.require_frozen_budget,
        limit_cells=args.limit_cells,
        limit_budgets=args.limit_budgets,
        device_override=args.device,
        plan_artifact_dir=args.plan_artifact_dir,
        figure_status=args.figure_status,
        figure_ev_ids=tuple(args.ev_id),
    )
    status_counts: dict[str, int] = {}
    for record in result.records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "cell_count": len(result.cells),
                "record_count": len(result.records),
                "status_counts": status_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
