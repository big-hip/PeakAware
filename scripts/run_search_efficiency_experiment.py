from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.config import PeakAwareConfig
from peakaware.experiments import (
    experiment_records_to_dicts,
    run_experiment_matrix,
    summarize_experiment_records,
    write_experiment_csv,
    write_experiment_json,
    write_experiment_summary_json,
)
from peakaware.search_efficiency import analyze_search_efficiency, compare_actual_measurement_runs

from analyze_search_efficiency import _write_csv, _write_figure, _write_json, _write_markdown


def _csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _csv_text(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one value")
    return values


def _task_budget_map(text: str | None) -> dict[str, int]:
    if text is None:
        return {}
    result: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        task_name, separator, budget = item.partition("=")
        if not separator or not task_name.strip() or not budget.strip():
            raise ValueError("--task-budget-mib entries must use task=MiB")
        result[task_name.strip()] = int(budget.strip()) << 20
    return result


def _stable_order_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(
        ":".join((str(base), *(str(part) for part in parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _run_matrix_with_fresh_compiler_cache(*, config: PeakAwareConfig, **kwargs):
    if not config.enable_compile:
        return run_experiment_matrix(config=config, **kwargs)
    torch.compiler.reset()
    candidate_limit = config.max_greedy_candidates + 4
    recompile_limit = max(int(torch._dynamo.config.recompile_limit), candidate_limit)
    accumulated_limit = max(
        int(torch._dynamo.config.accumulated_recompile_limit),
        recompile_limit,
    )
    with torch._dynamo.config.patch(
        recompile_limit=recompile_limit,
        accumulated_recompile_limit=accumulated_limit,
    ):
        return run_experiment_matrix(config=config, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired exhaustive and simulation-guided PeakAware searches.")
    parser.add_argument("--tasks", default="bert_base,gpt2,resnet50,vit_b_16")
    parser.add_argument("--budget-mib", default="256")
    parser.add_argument(
        "--safety-margin-mib",
        type=int,
        default=0,
        help="Conservative headroom calibrated offline from all-save peak residuals.",
    )
    parser.add_argument(
        "--task-budget-mib",
        default=None,
        help="Optional per-task physical budgets, for example bert_base=172,gpt2=337.",
    )
    parser.add_argument("--microbatches", default="1")
    parser.add_argument("--candidate-pool-top-k", type=int, default=12)
    parser.add_argument("--compiler-refinement-top-k", type=int, default=0)
    parser.add_argument(
        "--search-algorithm",
        choices=(
            "greedy",
            "pareto_beam",
            "lagrangian_beam",
            "lagrangian_sweep_beam",
        ),
        default="greedy",
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-beam-candidates", type=int, default=128)
    parser.add_argument(
        "--beam-candidate-overflow-policy",
        choices=("error", "coarsen_tail"),
        default="error",
    )
    parser.add_argument("--validation-top-k", type=int, default=5)
    parser.add_argument(
        "--validation-selection-policy",
        choices=("ranked", "structural_diverse"),
        default="ranked",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capture-backend", choices=("auto", "aot", "fx"), default="aot")
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--enable-inductor", action="store_true")
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--measurement-warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-repeats", type=int, default=20)
    parser.add_argument(
        "--candidate-measurement-protocol",
        choices=("legacy_phase", "publication_overall"),
        default="publication_overall",
        help=(
            "Use independent overall-step CUDA Event timing for paper-quality candidate "
            "runtime truth, or retain the legacy sum of phase wall clocks."
        ),
    )
    parser.add_argument("--matrix-passes", type=int, default=5)
    parser.add_argument(
        "--run-mode",
        choices=(
            "paired",
            "guided-only",
            "exhaustive-only",
            "simulation-only",
            "selected-validation",
        ),
        default="paired",
        help=(
            "Run both branches, one measured branch, a standalone zero-benchmark "
            "simulation-only branch, or validate only the simulator's top-ranked plan."
        ),
    )
    parser.add_argument("--reference-exhaustive-json", type=Path, default=None)
    parser.add_argument("--candidate-order-seed-base", type=int, default=None)
    parser.add_argument(
        "--reset-compiler-before-candidate-measurement",
        action="store_true",
    )
    parser.add_argument("--profile-db", type=Path, default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional persistent capture/analysis cache for repeated models or budgets.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.validation_top_k > args.candidate_pool_top_k + 3:
        raise ValueError("validation_top_k cannot exceed the maximum retained candidate pool")
    if args.matrix_passes <= 0:
        raise ValueError("matrix_passes must be positive")
    if args.safety_margin_mib < 0:
        raise ValueError("safety-margin-mib must be non-negative")
    if args.run_mode == "guided-only" and args.reference_exhaustive_json is None:
        raise ValueError("--reference-exhaustive-json is required for --run-mode guided-only")
    if args.run_mode == "simulation-only" and args.validation_top_k != 0:
        raise ValueError("--run-mode simulation-only requires --validation-top-k 0")
    if args.run_mode == "selected-validation" and args.validation_top_k != 1:
        raise ValueError("--run-mode selected-validation requires --validation-top-k 1")
    if args.run_mode == "selected-validation" and not (args.enable_compile or args.enable_inductor):
        raise ValueError("--run-mode selected-validation requires --enable-compile or --enable-inductor")

    tasks = _csv_text(args.tasks)
    budgets = tuple(value << 20 for value in _csv_ints(args.budget_mib))
    task_budgets = _task_budget_map(args.task_budget_mib)
    unknown_budget_tasks = sorted(set(task_budgets) - set(tasks))
    if unknown_budget_tasks:
        raise ValueError(f"task budgets provided for unselected tasks: {unknown_budget_tasks}")
    missing_budget_tasks = sorted(set(tasks) - set(task_budgets)) if task_budgets else []
    if missing_budget_tasks:
        raise ValueError(f"task budgets missing for selected tasks: {missing_budget_tasks}")
    microbatches = _csv_ints(args.microbatches)
    base_config = PeakAwareConfig(
        safety_margin_bytes=args.safety_margin_mib << 20,
        safety_margin_ratio=0.0,
        top_k=args.candidate_pool_top_k,
        max_greedy_candidates=args.candidate_pool_top_k,
        compiler_refinement_top_k=args.compiler_refinement_top_k,
        search_algorithm=args.search_algorithm,
        beam_width=args.beam_width,
        max_beam_candidates=args.max_beam_candidates,
        beam_candidate_overflow_policy=args.beam_candidate_overflow_policy,
        enable_compile=args.enable_compile or args.enable_inductor,
        enable_inductor=args.enable_inductor,
        capture_backend=args.capture_backend,
        isolate_candidate_measurement=args.isolate,
        profile_db_path=args.profile_db,
        cache_root=args.cache_root,
        measurement_warmup_steps=args.measurement_warmup_steps,
        measurement_repeats=args.measurement_repeats,
        candidate_measurement_protocol=args.candidate_measurement_protocol,
        selection_objective="min_time_then_peak",
        validation_selection_policy=args.validation_selection_policy,
        reset_compiler_before_candidate_measurement=(
            args.reset_compiler_before_candidate_measurement
        ),
        enable_diagnostic_hints=True,
    )
    exhaustive_records = []
    guided_records = []
    task_budget_cases = (
        tuple(((task_name,), (task_budgets[task_name],)) for task_name in tasks)
        if task_budgets
        else ((tasks, budgets),)
    )
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    total_pair_count = args.matrix_passes * len(task_budget_cases)
    experiment_start = time.perf_counter()
    _write_json(
        root / "progress.json",
        {
            "status": "running",
            "completed_pair_count": 0,
            "total_pair_count": total_pair_count,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
        },
    )
    for pass_index in range(args.matrix_passes):
        for case_index, (case_tasks, case_budgets) in enumerate(task_budget_cases):
            pair_start = time.perf_counter()
            common = {
                "task_names": case_tasks,
                "microbatch_sizes": microbatches,
                "budget_bytes": case_budgets,
                "variant_name": "diagnostic_hints_on",
                "device": args.device,
                "matrix_pass_index": pass_index,
                "matrix_pass_count": args.matrix_passes,
            }
            exhaustive_seed = (
                None
                if args.candidate_order_seed_base is None
                else _stable_order_seed(
                    args.candidate_order_seed_base,
                    "exhaustive",
                    pass_index,
                    case_index,
                    *case_tasks,
                )
            )
            guided_seed = (
                None
                if args.candidate_order_seed_base is None
                else _stable_order_seed(
                    args.candidate_order_seed_base,
                    "guided",
                    pass_index,
                    case_index,
                    *case_tasks,
                )
            )
            exhaustive_config = replace(
                base_config,
                validation_top_k=None,
                candidate_measurement_order_seed=exhaustive_seed,
            )
            guided_config = replace(
                base_config,
                validation_top_k=args.validation_top_k,
                candidate_measurement_order_seed=guided_seed,
            )
            guided_first = (pass_index + case_index) % 2 == 1
            if args.run_mode == "exhaustive-only":
                exhaustive_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=exhaustive_config,
                    )
                )
            elif args.run_mode in {"guided-only", "simulation-only", "selected-validation"}:
                guided_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=guided_config,
                    )
                )
            elif guided_first:
                guided_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=guided_config,
                    )
                )
                exhaustive_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=exhaustive_config,
                    )
                )
            else:
                exhaustive_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=exhaustive_config,
                    )
                )
                guided_records.extend(
                    _run_matrix_with_fresh_compiler_cache(
                        **common,
                        config=guided_config,
                    )
                )
            completed_pair_count = pass_index * len(task_budget_cases) + case_index + 1
            elapsed_seconds = time.perf_counter() - experiment_start
            mean_pair_seconds = elapsed_seconds / completed_pair_count
            estimated_remaining_seconds = mean_pair_seconds * (
                total_pair_count - completed_pair_count
            )
            progress = {
                "status": "running",
                "completed_pair_count": completed_pair_count,
                "total_pair_count": total_pair_count,
                "elapsed_seconds": elapsed_seconds,
                "mean_pair_seconds": mean_pair_seconds,
                "estimated_remaining_seconds": estimated_remaining_seconds,
                "last_pair_seconds": time.perf_counter() - pair_start,
                "last_pass_index": pass_index,
                "last_case_index": case_index,
                "last_tasks": case_tasks,
                "last_budget_bytes": case_budgets,
                "exhaustive_record_count": len(exhaustive_records),
                "guided_record_count": len(guided_records),
            }
            _write_json(root / "progress.json", progress)
            print(
                json.dumps(
                    {
                        "progress": f"{completed_pair_count}/{total_pair_count}",
                        "last_tasks": case_tasks,
                        "last_pair_seconds": progress["last_pair_seconds"],
                        "estimated_remaining_seconds": estimated_remaining_seconds,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

    guided_tuple = tuple(guided_records)
    guided_json = root / f"guided_top{args.validation_top_k}_records.json"
    if args.run_mode == "simulation-only":
        write_experiment_json(guided_tuple, guided_json)
        write_experiment_csv(
            guided_tuple,
            root / f"guided_top{args.validation_top_k}_records.csv",
        )
        simulation_summary = summarize_experiment_records(guided_tuple)
        write_experiment_summary_json(
            simulation_summary,
            root / f"guided_top{args.validation_top_k}_summary.json",
        )
        _write_json(
            root / "run_config.json",
            {
                "tasks": tasks,
                "budget_bytes": budgets,
                "task_budget_bytes": task_budgets,
                "safety_margin_bytes": args.safety_margin_mib << 20,
                "microbatches": microbatches,
                "candidate_pool_top_k": args.candidate_pool_top_k,
                "compiler_refinement_top_k": args.compiler_refinement_top_k,
                "search_algorithm": args.search_algorithm,
                "beam_width": args.beam_width,
                "max_beam_candidates": args.max_beam_candidates,
                "beam_candidate_overflow_policy": args.beam_candidate_overflow_policy,
                "validation_top_k": args.validation_top_k,
                "validation_selection_policy": "simulation_only_ranked",
                "device": args.device,
                "capture_backend": args.capture_backend,
                "enable_compile": args.enable_compile or args.enable_inductor,
                "enable_inductor": args.enable_inductor,
                "measurement_warmup_steps": args.measurement_warmup_steps,
                "measurement_repeats": args.measurement_repeats,
                "candidate_measurement_protocol": args.candidate_measurement_protocol,
                "matrix_passes": args.matrix_passes,
                "run_mode": args.run_mode,
                "candidate_order_seed_base": args.candidate_order_seed_base,
                "reset_compiler_before_candidate_measurement": False,
                "candidate_benchmark_count": 0,
                "cache_root": None if args.cache_root is None else str(args.cache_root),
            },
        )
        _write_json(
            root / "progress.json",
            {
                "status": "complete",
                "completed_pair_count": total_pair_count,
                "total_pair_count": total_pair_count,
                "elapsed_seconds": time.perf_counter() - experiment_start,
                "exhaustive_record_count": 0,
                "guided_record_count": len(guided_tuple),
                "candidate_benchmark_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "mode": "simulation-only",
                    "record_count": len(guided_tuple),
                    "candidate_benchmark_count": 0,
                },
                indent=2,
            )
        )
        return
    if args.run_mode == "selected-validation":
        write_experiment_json(guided_tuple, guided_json)
        write_experiment_csv(
            guided_tuple,
            root / f"guided_top{args.validation_top_k}_records.csv",
        )
        write_experiment_summary_json(
            summarize_experiment_records(guided_tuple),
            root / f"guided_top{args.validation_top_k}_summary.json",
        )
        candidate_benchmark_count = sum(
            len(record.candidate_attempts) for record in guided_tuple
        )
        candidate_measurement_count = sum(
            int(record.measured_candidate_count) for record in guided_tuple
        )
        failed_records = [record for record in guided_tuple if record.status != "ok"]
        if (
            failed_records
            or candidate_benchmark_count != len(guided_tuple)
            or candidate_measurement_count != len(guided_tuple)
        ):
            raise RuntimeError(
                "selected-validation must produce exactly one successful final-plan "
                f"measurement per record; failures={len(failed_records)}, "
                f"attempts={candidate_benchmark_count}, measurements={candidate_measurement_count}, "
                f"records={len(guided_tuple)}"
            )
        _write_json(
            root / "run_config.json",
            {
                "tasks": tasks,
                "budget_bytes": budgets,
                "task_budget_bytes": task_budgets,
                "safety_margin_bytes": args.safety_margin_mib << 20,
                "microbatches": microbatches,
                "candidate_pool_top_k": args.candidate_pool_top_k,
                "compiler_refinement_top_k": args.compiler_refinement_top_k,
                "search_algorithm": args.search_algorithm,
                "beam_width": args.beam_width,
                "max_beam_candidates": args.max_beam_candidates,
                "beam_candidate_overflow_policy": args.beam_candidate_overflow_policy,
                "validation_top_k": 1,
                "validation_selection_policy": "simulator_top1_final_validation",
                "device": args.device,
                "capture_backend": args.capture_backend,
                "enable_compile": args.enable_compile or args.enable_inductor,
                "enable_inductor": args.enable_inductor,
                "measurement_warmup_steps": args.measurement_warmup_steps,
                "measurement_repeats": args.measurement_repeats,
                "candidate_measurement_protocol": args.candidate_measurement_protocol,
                "matrix_passes": args.matrix_passes,
                "run_mode": args.run_mode,
                "candidate_order_seed_base": args.candidate_order_seed_base,
                "reset_compiler_before_candidate_measurement": (
                    args.reset_compiler_before_candidate_measurement
                ),
                "candidate_benchmark_count": candidate_benchmark_count,
                "candidate_measurement_count": candidate_measurement_count,
                "profile_db": None if args.profile_db is None else str(args.profile_db),
                "cache_root": None if args.cache_root is None else str(args.cache_root),
            },
        )
        _write_json(
            root / "progress.json",
            {
                "status": "complete",
                "completed_pair_count": total_pair_count,
                "total_pair_count": total_pair_count,
                "elapsed_seconds": time.perf_counter() - experiment_start,
                "exhaustive_record_count": 0,
                "guided_record_count": len(guided_tuple),
                "candidate_benchmark_count": candidate_benchmark_count,
                "candidate_measurement_count": candidate_measurement_count,
            },
        )
        print(
            json.dumps(
                {
                    "mode": "selected-validation",
                    "record_count": len(guided_tuple),
                    "candidate_benchmark_count": candidate_benchmark_count,
                    "candidate_measurement_count": candidate_measurement_count,
                },
                indent=2,
            )
        )
        return
    if args.run_mode in {"paired", "exhaustive-only"}:
        exhaustive_tuple = tuple(exhaustive_records)
        exhaustive_json = root / "exhaustive_records.json"
        write_experiment_json(exhaustive_tuple, exhaustive_json)
        write_experiment_csv(exhaustive_tuple, root / "exhaustive_records.csv")
        write_experiment_summary_json(
            summarize_experiment_records(exhaustive_tuple),
            root / "exhaustive_summary.json",
        )
        exhaustive_dicts = experiment_records_to_dicts(exhaustive_tuple)
    else:
        exhaustive_json = args.reference_exhaustive_json
        assert exhaustive_json is not None
        loaded_exhaustive = json.loads(exhaustive_json.read_text(encoding="utf-8"))
        if not isinstance(loaded_exhaustive, list):
            raise ValueError("reference exhaustive JSON must contain a list")
        exhaustive_dicts = loaded_exhaustive
    if args.run_mode == "exhaustive-only":
        _write_json(
            root / "run_config.json",
            {
                "tasks": tasks,
                "budget_bytes": budgets,
                "task_budget_bytes": task_budgets,
                "safety_margin_bytes": args.safety_margin_mib << 20,
                "microbatches": microbatches,
                "candidate_pool_top_k": args.candidate_pool_top_k,
                "compiler_refinement_top_k": args.compiler_refinement_top_k,
                "search_algorithm": args.search_algorithm,
                "beam_width": args.beam_width,
                "max_beam_candidates": args.max_beam_candidates,
                "beam_candidate_overflow_policy": args.beam_candidate_overflow_policy,
                "validation_top_k": None,
                "device": args.device,
                "capture_backend": args.capture_backend,
                "enable_compile": args.enable_compile or args.enable_inductor,
                "enable_inductor": args.enable_inductor,
                "measurement_warmup_steps": args.measurement_warmup_steps,
                "measurement_repeats": args.measurement_repeats,
                "candidate_measurement_protocol": args.candidate_measurement_protocol,
                "matrix_passes": args.matrix_passes,
                "run_mode": args.run_mode,
                "candidate_order_seed_base": args.candidate_order_seed_base,
                "reset_compiler_before_candidate_measurement": (
                    args.reset_compiler_before_candidate_measurement
                ),
                "compiler_cache_policy": (
                    "reset_before_each_record_and_each_candidate"
                    if args.reset_compiler_before_candidate_measurement
                    else "reset_before_each_record"
                ),
                "cache_root": None if args.cache_root is None else str(args.cache_root),
            },
        )
        _write_json(
            root / "progress.json",
            {
                "status": "complete",
                "completed_pair_count": total_pair_count,
                "total_pair_count": total_pair_count,
                "elapsed_seconds": time.perf_counter() - experiment_start,
                "exhaustive_record_count": len(exhaustive_dicts),
                "guided_record_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "mode": "exhaustive-only",
                    "record_count": len(exhaustive_dicts),
                },
                indent=2,
            )
        )
        return
    write_experiment_json(guided_tuple, guided_json)
    write_experiment_csv(guided_tuple, root / f"guided_top{args.validation_top_k}_records.csv")
    write_experiment_summary_json(
        summarize_experiment_records(guided_tuple),
        root / f"guided_top{args.validation_top_k}_summary.json",
    )

    guided_dicts = experiment_records_to_dicts(guided_tuple)
    replay_top_k = max(args.validation_top_k, 1)
    payload = analyze_search_efficiency(
        exhaustive_dicts,
        selected_top_k=replay_top_k,
        top_k_values=tuple(range(1, args.candidate_pool_top_k + 4)),
        variants=frozenset({"diagnostic_hints_on"}),
    )
    payload["actual_validation_top_k"] = args.validation_top_k
    payload["actual_paired_resource"] = compare_actual_measurement_runs(
        exhaustive_dicts,
        guided_dicts,
        guided_selection_policy=(
            "simulation_only_ranked"
            if args.validation_top_k == 0
            else args.validation_selection_policy
        ),
    )
    _write_json(root / "search_efficiency_summary.json", payload)
    _write_csv(root / "search_efficiency_rows.csv", payload["rows"])
    _write_csv(
        root / "actual_paired_rows.csv",
        payload["actual_paired_resource"]["rows"],
    )
    _write_markdown(root / "SEARCH_EFFICIENCY_REPORT.md", payload, exhaustive_json)
    _write_figure(root / "search_efficiency.png", payload)
    _write_json(
        root / "run_config.json",
        {
            "tasks": tasks,
            "budget_bytes": budgets,
            "task_budget_bytes": task_budgets,
            "safety_margin_bytes": args.safety_margin_mib << 20,
            "microbatches": microbatches,
            "candidate_pool_top_k": args.candidate_pool_top_k,
            "compiler_refinement_top_k": args.compiler_refinement_top_k,
            "search_algorithm": args.search_algorithm,
            "beam_width": args.beam_width,
            "max_beam_candidates": args.max_beam_candidates,
            "beam_candidate_overflow_policy": args.beam_candidate_overflow_policy,
            "validation_top_k": args.validation_top_k,
            "validation_selection_policy": args.validation_selection_policy,
            "device": args.device,
            "capture_backend": args.capture_backend,
            "enable_compile": args.enable_compile or args.enable_inductor,
            "enable_inductor": args.enable_inductor,
            "measurement_warmup_steps": args.measurement_warmup_steps,
            "measurement_repeats": args.measurement_repeats,
            "candidate_measurement_protocol": args.candidate_measurement_protocol,
            "matrix_passes": args.matrix_passes,
            "run_mode": args.run_mode,
            "reference_exhaustive_json": None
            if args.reference_exhaustive_json is None
            else str(args.reference_exhaustive_json),
            "candidate_order_seed_base": args.candidate_order_seed_base,
            "reset_compiler_before_candidate_measurement": (
                args.reset_compiler_before_candidate_measurement
            ),
            "paired_order": "alternating_by_pass_and_task",
            "compiler_cache_policy": (
                "reset_before_each_record_and_each_candidate"
                if args.reset_compiler_before_candidate_measurement
                else "reset_before_each_record_and_raise_recompile_limit_to_candidate_count"
                if args.enable_compile or args.enable_inductor
                else "not_applicable"
            ),
            "profile_db": None if args.profile_db is None else str(args.profile_db),
            "cache_root": None if args.cache_root is None else str(args.cache_root),
        },
    )
    _write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed_pair_count": total_pair_count,
            "total_pair_count": total_pair_count,
            "elapsed_seconds": time.perf_counter() - experiment_start,
            "exhaustive_record_count": len(exhaustive_dicts),
            "guided_record_count": len(guided_records),
        },
    )
    print(json.dumps({"aggregate": payload["aggregate"], "actual": payload["actual_paired_resource"]}, indent=2))


if __name__ == "__main__":
    main()
