from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.experiments import (
    experiment_records_from_dicts,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
    write_experiment_baseline_comparison_json,
    write_experiment_cache_reuse_json,
    write_experiment_hint_ablation_json,
    write_experiment_layered_accuracy_json,
    write_experiment_simulation_error_json,
    write_experiment_steady_state_json,
    write_experiment_summary_json,
    write_experiment_variant_summary_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PeakAware publication derived summaries from records.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    records_payload = json.loads(args.records_json.read_text(encoding="utf-8"))
    records = experiment_records_from_dicts(records_payload)
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": args.output_root / "summary.json",
        "variant_summary": args.output_root / "variant_summary.json",
        "baseline_comparison": args.output_root / "baseline_comparison.json",
        "hint_ablation": args.output_root / "hint_ablation.json",
        "cache_reuse": args.output_root / "cache_reuse.json",
        "layered_accuracy": args.output_root / "layered_accuracy.json",
        "simulation_error": args.output_root / "simulation_error.json",
        "steady_state": args.output_root / "steady_state.json",
    }
    write_experiment_summary_json(summarize_experiment_records(records), outputs["summary"])
    write_experiment_variant_summary_json(summarize_experiment_records_by_variant(records), outputs["variant_summary"])
    write_experiment_baseline_comparison_json(records, outputs["baseline_comparison"])
    write_experiment_hint_ablation_json(records, outputs["hint_ablation"])
    write_experiment_cache_reuse_json(records, outputs["cache_reuse"])
    write_experiment_layered_accuracy_json(records, outputs["layered_accuracy"])
    write_experiment_simulation_error_json(records, outputs["simulation_error"])
    write_experiment_steady_state_json(records, outputs["steady_state"])
    print(
        json.dumps(
            {
                "record_count": len(records),
                "output_root": str(args.output_root),
                "written_paths": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
