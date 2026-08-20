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
    write_experiment_selected_regret_json,
    write_experiment_simulation_error_json,
    write_experiment_steady_state_json,
    write_experiment_summary_json,
    write_experiment_variant_summary_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate PeakAware summary artifacts from saved records.")
    parser.add_argument("records_json", type=Path)
    parser.add_argument("--sac-baseline-json", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument("--output-variant-summary-json", type=Path, default=None)
    parser.add_argument("--output-hint-ablation-json", type=Path, default=None)
    parser.add_argument("--output-cache-reuse-json", type=Path, default=None)
    parser.add_argument("--output-baseline-comparison-json", type=Path, default=None)
    parser.add_argument("--output-layered-accuracy-json", type=Path, default=None)
    parser.add_argument("--output-simulation-error-json", type=Path, default=None)
    parser.add_argument("--output-selected-regret-json", type=Path, default=None)
    parser.add_argument("--output-steady-state-json", type=Path, default=None)
    args = parser.parse_args()

    records_payload = json.loads(args.records_json.read_text(encoding="utf-8"))
    records = experiment_records_from_dicts(records_payload)
    written_paths: list[str] = []

    if args.output_summary_json is not None:
        write_experiment_summary_json(summarize_experiment_records(records), args.output_summary_json)
        written_paths.append(str(args.output_summary_json))
    if args.output_variant_summary_json is not None:
        write_experiment_variant_summary_json(
            summarize_experiment_records_by_variant(records),
            args.output_variant_summary_json,
        )
        written_paths.append(str(args.output_variant_summary_json))
    if args.output_hint_ablation_json is not None:
        write_experiment_hint_ablation_json(records, args.output_hint_ablation_json)
        written_paths.append(str(args.output_hint_ablation_json))
    if args.output_cache_reuse_json is not None:
        write_experiment_cache_reuse_json(records, args.output_cache_reuse_json)
        written_paths.append(str(args.output_cache_reuse_json))
    if args.output_baseline_comparison_json is not None:
        sac_baseline = None
        if args.sac_baseline_json is not None:
            sac_baseline = json.loads(args.sac_baseline_json.read_text(encoding="utf-8"))
        write_experiment_baseline_comparison_json(
            records,
            args.output_baseline_comparison_json,
            sac_baseline=sac_baseline,
        )
        written_paths.append(str(args.output_baseline_comparison_json))
    if args.output_layered_accuracy_json is not None:
        write_experiment_layered_accuracy_json(records, args.output_layered_accuracy_json)
        written_paths.append(str(args.output_layered_accuracy_json))
    if args.output_simulation_error_json is not None:
        write_experiment_simulation_error_json(records, args.output_simulation_error_json)
        written_paths.append(str(args.output_simulation_error_json))
    if args.output_selected_regret_json is not None:
        write_experiment_selected_regret_json(records, args.output_selected_regret_json)
        written_paths.append(str(args.output_selected_regret_json))
    if args.output_steady_state_json is not None:
        write_experiment_steady_state_json(records, args.output_steady_state_json)
        written_paths.append(str(args.output_steady_state_json))

    print(json.dumps({"record_count": len(records), "written_paths": written_paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
