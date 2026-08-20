from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.contracts import FixedTimeline, JointTrainingIR, OpInfo, StorageInfo, ValueInfo
from peakaware.cost.base import OpCost, OpSignature
from peakaware.search.beam import solve_exact_candidate_sets, solve_peak_aware_beam


class _SyntheticCostProvider:
    cache_safe = True
    source = "synthetic_search_oracle"

    def __init__(self, costs_us: tuple[float, ...]) -> None:
        self.costs_us = costs_us

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def estimate(self, signature: OpSignature) -> OpCost:
        try:
            index = int(signature.target.rsplit(".", 1)[-1])
            estimated_us = self.costs_us[index]
        except (ValueError, IndexError):
            estimated_us = 0.0
        return OpCost(
            estimated_us=float(estimated_us),
            memory_bytes=0,
            source=self.source,
            confidence=1.0,
        )


def _synthetic_branching_ir(
    seed: int,
    *,
    candidate_count: int,
) -> tuple[JointTrainingIR, _SyntheticCostProvider]:
    rng = random.Random(seed)
    sizes = tuple(rng.randrange(32, 513) * 1024 for _ in range(candidate_count))
    costs = tuple(rng.uniform(1.0, 120.0) for _ in range(candidate_count))
    inputs: list[tuple[int, ...]] = []
    for index in range(candidate_count):
        dependencies = [] if index == 0 else [index - 1]
        if index > 2 and rng.random() < 0.7:
            dependencies.append(rng.randrange(0, index - 1))
        inputs.append(tuple(sorted(set(dependencies))))
    consumers: dict[int, list[int]] = {index: [] for index in range(candidate_count)}
    for index, dependencies in enumerate(inputs):
        for dependency in dependencies:
            consumers[dependency].append(index)
    for index in range(candidate_count):
        consumers[index].append(candidate_count + index)
    ir = JointTrainingIR(
        ops=(
            tuple(
                OpInfo(
                    id=index,
                    name=f"fw{index}",
                    target=f"aten.synthetic.{index}",
                    phase="fw",
                    input_value_ids=inputs[index],
                    output_value_ids=(index,),
                    recomputable=True,
                    mandatory_save_reason=None,
                )
                for index in range(candidate_count)
            )
            + tuple(
                OpInfo(
                    id=candidate_count + index,
                    name=f"bw{index}",
                    target=f"aten.synthetic_bw.{index}",
                    phase="bw",
                    input_value_ids=(index,),
                    output_value_ids=(),
                    recomputable=False,
                    mandatory_save_reason="synthetic_backward_consumer",
                )
                for index in reversed(range(candidate_count))
            )
        ),
        values=tuple(
            ValueInfo(
                id=index,
                producer_id=index,
                consumer_ids=tuple(consumers[index]),
                storage_id=index,
                logical_nbytes=sizes[index],
                phase="fw",
                crosses_fw_bw=True,
                recomputable=True,
                mandatory_save_reason=None,
                name=f"v{index}",
            )
            for index in range(candidate_count)
        ),
        storages=tuple(
            StorageInfo(
                id=index,
                value_ids=(index,),
                physical_nbytes=sizes[index],
                is_external=False,
            )
            for index in range(candidate_count)
        ),
        regions=(),
        graph_key=f"synthetic-search-oracle-{seed}",
    )
    return ir, _SyntheticCostProvider(costs)


def benchmark_search_oracle(
    *,
    seeds: int = 20,
    candidate_count: int = 10,
    beam_widths: tuple[int, ...] = (4, 8, 16),
    budget_quantiles: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
) -> dict[str, Any]:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if candidate_count <= 0 or candidate_count > 16:
        raise ValueError("candidate_count must be in [1, 16] for exact enumeration")
    if not beam_widths or any(width <= 0 for width in beam_widths):
        raise ValueError("beam_widths must contain positive values")
    if not budget_quantiles or any(not 0.0 <= value <= 1.0 for value in budget_quantiles):
        raise ValueError("budget_quantiles must be in [0, 1]")

    fixed = FixedTimeline(
        parameter_bytes=2 << 20,
        buffer_bytes=0,
        gradient_bytes=2 << 20,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=0,
    )
    methods = {
        "pareto_beam": ("pareto_lexicographic", 16, "error"),
        "lagrangian_beam": ("lagrangian", 16, "error"),
        "lagrangian_sweep_beam": ("lagrangian_sweep", 16, "error"),
        "lagrangian_coarse8": ("lagrangian", 8, "coarsen_tail"),
        "lagrangian_coarse4": ("lagrangian", 4, "coarsen_tail"),
        "objective_beam": ("objective", 16, "error"),
        "peak_only_beam": ("peak_only", 16, "error"),
    }
    accumulators: dict[tuple[int, str], dict[str, Any]] = {
        (width, method): {
            "case_count": 0,
            "exact_hit_count": 0,
            "feasible_miss_count": 0,
            "time_regrets": [],
            "evaluated_plan_counts": [],
            "effective_candidate_counts": [],
            "coarsened_count": 0,
            "exact_plan_counts": [],
        }
        for width in beam_widths
        for method in methods
    }

    for seed in range(seeds):
        ir, provider = _synthetic_branching_ir(seed, candidate_count=candidate_count)
        loose = solve_exact_candidate_sets(
            ir,
            fixed,
            budget_bytes=1 << 60,
            cost_provider=provider,
            event_trace_materialization="none",
        )
        peaks = sorted({item.simulation.estimated_peak_bytes for item in loose.evaluated})
        for quantile in budget_quantiles:
            budget_index = min(len(peaks) - 1, int((len(peaks) - 1) * quantile))
            budget_bytes = peaks[budget_index]
            exact = solve_exact_candidate_sets(
                ir,
                fixed,
                budget_bytes=budget_bytes,
                cost_provider=provider,
                event_trace_materialization="none",
            )
            for width in beam_widths:
                for method, (
                    pruning_strategy,
                    max_candidate_count,
                    candidate_overflow_policy,
                ) in methods.items():
                    approximate = solve_peak_aware_beam(
                        ir,
                        fixed,
                        budget_bytes=budget_bytes,
                        beam_width=width,
                        max_candidate_count=max_candidate_count,
                        candidate_overflow_policy=candidate_overflow_policy,
                        cost_provider=provider,
                        pruning_strategy=pruning_strategy,
                        event_trace_materialization="none",
                    )
                    accumulator = accumulators[(width, method)]
                    accumulator["case_count"] += 1
                    accumulator["exact_hit_count"] += int(
                        approximate.best.plan.saved_value_ids
                        == exact.best.plan.saved_value_ids
                    )
                    if exact.best.feasible and not approximate.best.feasible:
                        accumulator["feasible_miss_count"] += 1
                    if exact.best.feasible and approximate.best.feasible:
                        exact_us = exact.best.simulation.estimated_step_us
                        approximate_us = approximate.best.simulation.estimated_step_us
                        accumulator["time_regrets"].append(
                            (approximate_us - exact_us) / exact_us
                            if exact_us > 0.0
                            else 0.0
                        )
                    accumulator["evaluated_plan_counts"].append(
                        approximate.evaluated_plan_count
                    )
                    accumulator["effective_candidate_counts"].append(
                        approximate.candidate_count
                    )
                    accumulator["coarsened_count"] += int(
                        approximate.candidate_coarsened
                    )
                    accumulator["exact_plan_counts"].append(exact.evaluated_plan_count)

    rows: list[dict[str, Any]] = []
    for (width, method), accumulator in sorted(accumulators.items()):
        case_count = accumulator["case_count"]
        regrets = accumulator["time_regrets"]
        evaluated_counts = accumulator["evaluated_plan_counts"]
        exact_counts = accumulator["exact_plan_counts"]
        effective_candidate_counts = accumulator["effective_candidate_counts"]
        rows.append(
            {
                "beam_width": width,
                "method": method,
                "case_count": case_count,
                "exact_hit_rate": accumulator["exact_hit_count"] / case_count,
                "feasible_miss_rate": accumulator["feasible_miss_count"] / case_count,
                "mean_time_regret_ratio": statistics.mean(regrets) if regrets else None,
                "max_time_regret_ratio": max(regrets) if regrets else None,
                "mean_evaluated_plan_count": statistics.mean(evaluated_counts),
                "mean_effective_candidate_count": statistics.mean(
                    effective_candidate_counts
                ),
                "candidate_coarsened_rate": accumulator["coarsened_count"]
                / case_count,
                "mean_exact_plan_count": statistics.mean(exact_counts),
                "plan_evaluation_reduction_rate": 1.0
                - statistics.mean(evaluated_counts) / statistics.mean(exact_counts),
            }
        )
    return {
        "evidence_scope": "synthetic_small_graph_simulation_oracle",
        "candidate_measurements_used": 0,
        "seeds": seeds,
        "candidate_count": candidate_count,
        "budget_quantiles": list(budget_quantiles),
        "beam_widths": list(beam_widths),
        "rows": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Search Planner Exact-Oracle Benchmark",
        "",
        "This is a synthetic small-graph planner benchmark. It uses the PeakAware simulator only; no candidate GPU measurement is consumed.",
        "",
        "| Planner | Width | Decisions | Coarsened | Exact-plan hit | Feasible miss | Mean time regret | Max time regret | Simulated plans | Reduction vs exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['method']} | {row['beam_width']} | "
            f"{row['mean_effective_candidate_count']:.1f} | "
            f"{row['candidate_coarsened_rate'] * 100.0:.2f}% | "
            f"{row['exact_hit_rate'] * 100.0:.2f}% | "
            f"{row['feasible_miss_rate'] * 100.0:.2f}% | "
            f"{row['mean_time_regret_ratio'] * 100.0:.2f}% | "
            f"{row['max_time_regret_ratio'] * 100.0:.2f}% | "
            f"{row['mean_evaluated_plan_count']:.1f} | "
            f"{row['plan_evaluation_reduction_rate'] * 100.0:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: exact enumeration is an oracle only for these generated graphs and the current simulator objective. It does not establish real-model or GPU-runtime superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare approximate planners with an exact simulator oracle.")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--beam-widths", default="4,8,16")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = benchmark_search_oracle(
        seeds=args.seeds,
        candidate_count=args.candidate_count,
        beam_widths=_parse_int_tuple(args.beam_widths),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "search_oracle_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "SEARCH_ORACLE_BENCHMARK.md", payload)


if __name__ == "__main__":
    main()
