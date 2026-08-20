from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.search_efficiency import analyze_search_efficiency, compare_actual_measurement_runs


def _parse_csv_values(text: str | None) -> frozenset[str] | None:
    if text is None:
        return None
    values = frozenset(item.strip() for item in text.split(",") if item.strip())
    return values or None


def _parse_top_k_values(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top-k-values must contain positive integers")
    return values


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for field, value in row.items()
                }
            )


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"


def _speedup(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}×"


def _seconds_from_gpu_hours(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 3600.0:.2f} s"


def _regret_vs_random(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if numeric >= 0:
        return f"{numeric * 100.0:.2f}% lower regret"
    return f"{-numeric * 100.0:.2f}% higher regret"


def _write_markdown(path: Path, payload: dict[str, Any], source: Path) -> None:
    aggregate = payload["aggregate"]
    actual = payload.get("actual_paired_resource")
    zero_validation = payload.get("grouped_zero_validation_selection") or {}
    lines = [
        "# PeakAware Search-Efficiency Experiment",
        "",
        f"Source records: `{source}`",
        "",
        f"Evidence mode: `{payload['evidence_mode']}`. The Top-k branch is replayed from candidates that were all measured in the source artifact.",
        "",
    ]
    if actual is not None:
        grouped = actual.get("grouped_final_selection") or {}
        guided_policy = actual.get("guided_selection_policy", "ranked")
        if guided_policy == "simulation_only_ranked":
            guided_description = (
                "the guided branch uses simulation to choose and materialize one executable without "
                "candidate warmup, timing, memory measurement, or correctness dry-run."
            )
        else:
            guided_description = (
                f"the guided branch uses the `{guided_policy}` simulation policy to choose "
                "the strategies that are actually validated."
            )
        lines.extend(
            [
                "## Primary result: paired all-strategy actual execution",
                "",
                (
                    "Both branches were executed independently on the GPU. The baseline validates every "
                    f"generated strategy; {guided_description}"
                ),
                "",
                "| Search mode | End-to-end time | Candidate attempts | Speedup vs all-strategy actual run | Workload-level runtime regret |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| All strategies actual run | {_seconds_from_gpu_hours(actual['full_total_search_gpu_hours'])} | "
                    f"{actual['full_candidate_attempt_count']} | 1.00× | 0.00% |"
                ),
                (
                    f"| Simulation-guided actual run ({guided_policy}) | "
                    f"{_seconds_from_gpu_hours(actual['peakaware_total_search_gpu_hours'])} | "
                    f"{actual['peakaware_candidate_attempt_count']} | "
                    f"{_speedup(actual.get('all_strategy_actual_run_speedup'))} | "
                    f"{_percent(grouped.get('mean_time_regret_ratio'))} mean / "
                    f"{_percent(grouped.get('max_time_regret_ratio'))} max |"
                ),
                "",
                (
                    f"Across {actual['paired_record_count']} paired runs, the ratio of summed wall times is "
                    f"{_speedup(actual.get('all_strategy_actual_run_speedup'))}; the median per-pair speedup is "
                    f"{_speedup(actual.get('paired_total_search_speedup_p50'))} "
                    f"(P10–P90: {_speedup(actual.get('paired_total_search_speedup_p10'))}–"
                    f"{_speedup(actual.get('paired_total_search_speedup_p90'))})."
                ),
                "",
                "| Task | Paired runs | All-strategy time | Guided time | End-to-end speedup | Median paired speedup | Mean pass-level regret |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task_name, summary in actual.get("by_task", {}).items():
            lines.append(
                f"| {task_name} | {summary['paired_record_count']} | "
                f"{_seconds_from_gpu_hours(summary['full_total_search_gpu_hours'])} | "
                f"{_seconds_from_gpu_hours(summary['peakaware_total_search_gpu_hours'])} | "
                f"{_speedup(summary['all_strategy_actual_run_speedup'])} | "
                f"{_speedup(summary['paired_total_search_speedup_p50'])} | "
                f"{_percent(summary['guided_selection_mean_time_regret_ratio'])} |"
            )
        lines.extend(
            [
                "",
                "### Actual validation and selection audit",
                "",
                "| Metric | All-strategy run | Guided run | Reduction / quality |",
                "|---|---:|---:|---:|",
                (
                    f"| Candidate-validation GPU-hours | {actual['full_measurement_gpu_hours']:.6f} | "
                    f"{actual['peakaware_measurement_gpu_hours']:.6f} | "
                    f"{_percent(actual['gpu_hours_reduction_rate'])} "
                    f"({'eliminated' if actual['peakaware_measurement_gpu_hours'] == 0 else _speedup(actual.get('candidate_validation_speedup'))}) |"
                ),
                (
                    f"| End-to-end search GPU-hours | {actual['full_total_search_gpu_hours']:.6f} | "
                    f"{actual['peakaware_total_search_gpu_hours']:.6f} | "
                    f"{_percent(actual['total_search_gpu_hours_reduction_rate'])} "
                    f"({_speedup(actual.get('total_search_speedup'))}) |"
                ),
                (
                    f"| Exhaustive-oracle selection rate | 100.00% | "
                    f"{_percent(grouped.get('guided_selected_oracle_rate'))} | "
                    f"{grouped.get('selection_reference_count', 0)} referenced workloads |"
                ),
                "",
                (
                    f"Input audit: exhaustive failed={actual['exhaustive_failed_record_count']}, "
                    f"guided failed={actual['guided_failed_record_count']}, "
                    f"exhaustive unmatched={actual['exhaustive_unmatched_ok_record_count']}, "
                    f"guided unmatched={actual['guided_unmatched_ok_record_count']}, "
                    f"guided candidates covered by exhaustive measurements="
                    f"{_percent(actual.get('guided_candidate_coverage_rate'))}."
                ),
                "",
            ]
        )
        shortlist = actual.get("grouped_shortlist_replay") or {}
        shortlist_mean = actual.get("guided_shortlist_replay_mean_time_regret_ratio")
        if shortlist and shortlist_mean is not None:
            random_mean = aggregate.get("random_topk_mean_time_regret_ratio")
            relative_to_random = (
                None
                if random_mean in (None, 0)
                else 1.0 - float(shortlist_mean) / float(random_mean)
            )
            lines.extend(
                [
                    "### Structural-shortlist replay on exhaustive measurements",
                    "",
                    (
                        "The independently executed guided run and exhaustive run have substantial timing noise. "
                        "To isolate shortlist membership from cross-run selection noise, this replay maps the exact "
                        "five candidates actually chosen by `structural_diverse` onto the matching exhaustive records "
                        "and selects the fastest measured member."
                    ),
                    "",
                    "| View | Oracle coverage | Mean runtime regret | P90 / maximum regret |",
                    "|---|---:|---:|---:|",
                    (
                        f"| Per-pass exhaustive replay | "
                        f"{_percent(actual.get('guided_shortlist_oracle_coverage_rate'))} | "
                        f"{_percent(shortlist_mean)} | "
                        f"{_percent(actual.get('guided_shortlist_replay_p90_time_regret_ratio'))} P90 / "
                        f"{_percent(actual.get('guided_shortlist_replay_max_time_regret_ratio'))} max |"
                    ),
                    (
                        f"| Repeated-run median by workload | "
                        f"{_percent(shortlist.get('oracle_coverage_rate'))} | "
                        f"{_percent(shortlist.get('mean_time_regret_ratio'))} | "
                        f"{_percent(shortlist.get('max_time_regret_ratio'))} max |"
                    ),
                    "",
                    (
                        "Against the exact uniform-random Top-k expectation, the structural shortlist has "
                        f"{_regret_vs_random(relative_to_random)}. This is a replay of measured candidates, "
                        "not a new independent guided GPU run."
                    ),
                    "",
                ]
            )
    counterfactual_relation = (
        "It is separate from the structurally diversified actual guided run. "
        if actual is not None
        and actual.get("guided_selection_policy") == "structural_diverse"
        else "It complements the independently executed guided result above. "
    )
    lines.extend(
        [
            "## Counterfactual ranked Top-k operating-point analysis",
        "",
        (
            "This replay always uses the pure simulator-ranked prefix over the exhaustive candidate records. "
            + counterfactual_relation
            + "All modes retain the same capture, IR construction, executor construction, and simulation-analysis cost."
        ),
        "",
        (
            f"Using repeated-run medians per workload, zero-validation selection is actually feasible in "
            f"{_percent(zero_validation.get('actual_feasible_rate'))} of groups, hits the exhaustive oracle in "
            f"{_percent(zero_validation.get('oracle_hit_rate'))}, and has "
            f"{_percent(zero_validation.get('mean_time_regret_ratio'))} mean / "
            f"{_percent(zero_validation.get('max_time_regret_ratio'))} maximum runtime regret."
        ),
        "",
        "| Search mode | End-to-end GPU-hours | Speedup vs all-strategy actual run | Actual validations | Mean runtime regret | P90 runtime regret |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| All strategies actual run | {aggregate['full_total_search_gpu_hours']:.6f} | 1.00× | "
            f"{aggregate['full_measurement_candidate_count']} | 0.00% | 0.00% |"
        ),
        (
            f"| Simulation-only plan selection (0 candidate benchmarks) | "
            f"{aggregate['simulation_only_total_search_gpu_hours']:.6f} | "
            f"{_speedup(aggregate['simulation_only_end_to_end_speedup'])} | 0 | "
            f"{_percent(aggregate['simulation_only_mean_time_regret_ratio'])} | "
            f"{_percent(aggregate['simulation_only_p90_time_regret_ratio'])} |"
        ),
        (
            f"| Simulation + ranked Top-{payload['selected_top_k']} confirmation | "
            f"{aggregate['counterfactual_topk_total_search_gpu_hours']:.6f} | "
            f"{_speedup(aggregate['counterfactual_topk_end_to_end_speedup'])} | "
            f"{aggregate['peakaware_measured_candidate_count']} | "
            f"{_percent(aggregate['mean_time_regret_ratio'])} | "
            f"{_percent(aggregate['p90_time_regret_ratio'])} |"
        ),
        "",
        "### Counterfactual ranked Top-k replay",
        "",
        "| Metric | Full measurement | PeakAware Top-k | Reduction / quality |",
        "|---|---:|---:|---:|",
        (
            f"| Candidate measurements | {aggregate['full_measurement_candidate_count']} | "
            f"{aggregate['peakaware_measured_candidate_count']} | "
            f"{_percent(aggregate['candidate_measurement_reduction_rate'])} |"
        ),
        (
            f"| Replay-estimated validation GPU-hours | {aggregate['full_measurement_gpu_hours']:.6f} | "
            f"{aggregate['peakaware_measurement_gpu_hours']:.6f} | "
            f"{_percent(aggregate['gpu_hours_reduction_rate'])} |"
        ),
        f"| Best-time Hit@k | - | - | {_percent(aggregate['best_time_hit_rate'])} |",
        f"| Mean time regret | - | - | {_percent(aggregate['mean_time_regret_ratio'])} |",
        f"| P90 time regret | - | - | {_percent(aggregate['p90_time_regret_ratio'])} |",
        f"| Mean gap to minimum feasible peak | - | - | {_percent(aggregate['mean_min_peak_gap_ratio'])} |",
        f"| P90 gap to minimum feasible peak | - | - | {_percent(aggregate['p90_min_peak_gap_ratio'])} |",
        (
            f"| Mean time regret vs uniform random Top-k | "
            f"{_percent(aggregate['random_topk_mean_time_regret_ratio'])} random | "
            f"{_percent(aggregate['mean_time_regret_ratio'])} PeakAware | "
            f"{_regret_vs_random(aggregate['time_regret_reduction_vs_random_rate'])} |"
        ),
        "",
        "### Top-k sweep",
        "",
        "| k | Measured candidates | End-to-end speedup | Hit@k | Random Hit@k | Mean regret | Random mean regret | Regret reduction vs random |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for top_k, summary in sorted(payload["by_k"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {top_k} | {summary['peakaware_measured_candidate_count']} | "
            f"{_speedup(summary['counterfactual_topk_end_to_end_speedup'])} | "
            f"{_percent(summary['best_time_hit_rate'])} | "
            f"{_percent(summary['random_topk_best_time_hit_rate'])} | "
            f"{_percent(summary['mean_time_regret_ratio'])} | "
            f"{_percent(summary['random_topk_mean_time_regret_ratio'])} | "
            f"{_regret_vs_random(summary['time_regret_reduction_vs_random_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Per-task replay result",
            "",
            "| Task | Records | Full candidates | Top-k candidates | Hit@k | Mean time regret | FW-only false gains excluded | Budget violations excluded | OOM excluded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name, summary in payload["by_task"].items():
        lines.append(
            f"| {task_name} | {summary['record_count']} | {summary['full_measurement_candidate_count']} | "
            f"{summary['peakaware_measured_candidate_count']} | {_percent(summary['best_time_hit_rate'])} | "
            f"{_percent(summary['mean_time_regret_ratio'])} | "
            f"{summary['excluded_fw_only_no_global_gain_count']} | "
            f"{summary['excluded_actual_budget_violation_count']} | "
            f"{summary['excluded_actual_oom_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Panels derived from `by_k` are counterfactual replays over the exhaustive measured candidate set.",
            "- Simulation-only and counterfactual Top-k wall times remove or reduce candidate validation from the observed exhaustive run while retaining its shared fixed search cost.",
            "- Simulation-only is an end-to-end plan-selection counterfactual. The current runtime still needs to realize an executable before training; that later one-plan realization/dry-run cost is not separable in these records and is not included in the 0-validation value.",
            "- The paired resource section uses actual exhaustive/guided runs; its grouped final-selection regret maps the guided-selected plan back to exhaustive measurements of the same plan.",
            "- Random Top-k values are exact finite-set expectations for these exhaustively measured candidate pools. They are descriptive comparisons, not a population-level significance claim.",
            "- GPU-hours are exact only when per-candidate validation elapsed fields exist. Otherwise the source is reported in `gpu_hours_cost_source_counts` and must be described as prorated or a lower bound.",
            "- An OOM count of zero means no excluded measured row was labeled OOM; it does not prove that unrecorded failed candidate attempts were absent.",
            (
                "- A publication claim about actual saved GPU-hours requires paired exhaustive/guided GPU runs; "
                "the `actual_paired_resource` section is authoritative when present."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(path: Path, payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = payload["aggregate"]
    by_k = sorted(payload["by_k"].items(), key=lambda item: int(item[0]))
    selected_ks = [item for item in by_k if int(item[0]) in {1, 3}]
    actual = payload.get("actual_paired_resource") or {}
    labels = ["All-strategy\nactual"]
    speedups = [1.0]
    colors = ["#6B7280"]
    if actual.get("all_strategy_actual_run_speedup") is not None:
        labels.append("Guided\nactual")
        speedups.append(float(actual["all_strategy_actual_run_speedup"]))
        colors.append("#DC2626")
    labels.extend(["Zero-validation\n(replay)", *[f"Top-{top_k}\n(replay)" for top_k, _ in selected_ks]])
    speedups.extend(
        [
            float(aggregate["simulation_only_end_to_end_speedup"]),
            *[
                float(summary["counterfactual_topk_end_to_end_speedup"])
                for _, summary in selected_ks
            ],
        ]
    )
    colors.extend(["#2563EB", *["#60A5FA", "#34D399"][: len(selected_ks)]])

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    bars = axes[0].bar(labels, speedups, color=colors)
    axes[0].axhline(1.0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("End-to-end speedup (×)")
    axes[0].set_title("Search speedup vs all-strategy GPU execution")
    axes[0].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, speedups):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    sweep_x = [float(summary["mean_time_regret_ratio"]) * 100.0 for _, summary in by_k]
    sweep_y = [float(summary["counterfactual_topk_end_to_end_speedup"]) for _, summary in by_k]
    axes[1].plot(sweep_x, sweep_y, marker="o", color="#2563EB", linewidth=1.5)
    random_x = [
        float(summary["random_topk_mean_time_regret_ratio"]) * 100.0
        for _, summary in by_k
    ]
    random_y = [float(summary["random_topk_end_to_end_speedup"]) for _, summary in by_k]
    axes[1].plot(
        random_x,
        random_y,
        marker="o",
        color="#9CA3AF",
        linewidth=1.2,
        linestyle="--",
        label="Uniform random Top-k",
    )
    for (top_k, _), x_value, y_value in zip(by_k, sweep_x, sweep_y):
        axes[1].annotate(f"k={top_k}", (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    grouped = actual.get("grouped_final_selection") or {}
    if actual.get("total_search_speedup") is not None and grouped.get("mean_time_regret_ratio") is not None:
        axes[1].scatter(
            [float(grouped["mean_time_regret_ratio"]) * 100.0],
            [float(actual["total_search_speedup"])],
            marker="*",
            s=120,
            color="#DC2626",
            label="Actual paired Top-k",
            zorder=3,
        )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Mean measured runtime regret (%)")
    axes[1].set_ylabel("End-to-end speedup (×)")
    axes[1].set_title("Speed–quality trade-off")
    axes[1].grid(alpha=0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay PeakAware Top-k screening over exhaustive candidate measurements.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--guided-records-json", type=Path, default=None)
    parser.add_argument(
        "--guided-selection-policy",
        choices=("ranked", "structural_diverse", "simulation_only_ranked"),
        default="ranked",
    )
    parser.add_argument("--selected-top-k", type=int, default=2)
    parser.add_argument("--top-k-values", default="1,2,3,4")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, default=None)
    args = parser.parse_args()

    records = json.loads(args.records_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("records JSON must contain a list")
    payload = analyze_search_efficiency(
        records,
        selected_top_k=args.selected_top_k,
        top_k_values=_parse_top_k_values(args.top_k_values),
        variants=_parse_csv_values(args.variants),
        tasks=_parse_csv_values(args.tasks),
    )
    if args.guided_records_json is not None:
        guided_records = json.loads(args.guided_records_json.read_text(encoding="utf-8"))
        if not isinstance(guided_records, list):
            raise ValueError("guided records JSON must contain a list")
        payload["actual_paired_resource"] = compare_actual_measurement_runs(
            records,
            guided_records,
            guided_selection_policy=args.guided_selection_policy,
        )
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, payload["rows"])
    _write_markdown(args.output_md, payload, args.records_json)
    if args.output_figure is not None:
        _write_figure(args.output_figure, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
