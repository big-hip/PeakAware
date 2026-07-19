from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.config import PeakAwareConfig
from peakaware.experiments import (
    experiment_records_to_dicts,
    run_experiment_matrix,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
    write_experiment_csv,
    write_experiment_baseline_comparison_json,
    write_experiment_hint_ablation_json,
    write_experiment_json,
    write_experiment_layered_accuracy_json,
    write_experiment_summary_json,
    write_experiment_variant_summary_json,
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--diagnostic-hints", choices=("on", "off", "both"), default="on")
    parser.add_argument("--selection-objective", choices=("min_peak_then_time", "min_time_then_peak"), default="min_peak_then_time")
    parser.add_argument("--measurement-warmup-steps", type=int, default=0)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--exact-small-graph", action="store_true")
    parser.add_argument("--exact-max-candidates", type=int, default=12)
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument("--output-variant-summary-json", type=Path, default=None)
    parser.add_argument("--output-hint-ablation-json", type=Path, default=None)
    parser.add_argument("--output-baseline-comparison-json", type=Path, default=None)
    parser.add_argument("--sac-baseline-json", type=Path, default=None)
    parser.add_argument("--output-layered-accuracy-json", type=Path, default=None)
    args = parser.parse_args()

    base_config = PeakAwareConfig(
        safety_margin_bytes=0,
        safety_margin_ratio=0.0,
        top_k=args.top_k,
        isolate_candidate_measurement=args.isolate,
        profile_db_path=args.profile_db,
        cache_root=args.cache_root,
        measurement_warmup_steps=args.measurement_warmup_steps,
        measurement_repeats=args.measurement_repeats,
        selection_objective=args.selection_objective,
    )
    if args.diagnostic_hints == "both":
        variants = (("diagnostic_hints_on", True), ("diagnostic_hints_off", False))
    else:
        enabled = args.diagnostic_hints == "on"
        variants = ((f"diagnostic_hints_{args.diagnostic_hints}", enabled),)
    records = tuple(
        record
        for variant_name, enabled in variants
        for record in run_experiment_matrix(
            task_names=_parse_csv_text(args.tasks),
            microbatch_sizes=_parse_csv_ints(args.microbatches),
            budget_bytes=tuple(value << 20 for value in _parse_csv_ints(args.budget_mib)),
            config=replace(base_config, enable_diagnostic_hints=enabled),
            include_exact_baseline=args.exact_small_graph,
            exact_max_candidate_count=args.exact_max_candidates,
            variant_name=variant_name,
            device=args.device,
        )
    )
    if args.output_json is not None:
        write_experiment_json(records, args.output_json)
    if args.output_csv is not None:
        write_experiment_csv(records, args.output_csv)
    summary = summarize_experiment_records(records)
    if args.output_summary_json is not None:
        write_experiment_summary_json(summary, args.output_summary_json)
    if args.output_variant_summary_json is not None:
        write_experiment_variant_summary_json(
            summarize_experiment_records_by_variant(records),
            args.output_variant_summary_json,
        )
    if args.output_hint_ablation_json is not None:
        write_experiment_hint_ablation_json(records, args.output_hint_ablation_json)
    if args.output_baseline_comparison_json is not None:
        sac_baseline = None
        if args.sac_baseline_json is not None:
            sac_baseline = json.loads(args.sac_baseline_json.read_text(encoding="utf-8"))
        write_experiment_baseline_comparison_json(
            records,
            args.output_baseline_comparison_json,
            sac_baseline=sac_baseline,
        )
    if args.output_layered_accuracy_json is not None:
        write_experiment_layered_accuracy_json(records, args.output_layered_accuracy_json)
    print(json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
