from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "peakaware-search-efficiency-v3"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0.0 else numerator / denominator


def _candidate_plan_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("plan_id") or "unknown")


def _estimated_feasible(candidate: Mapping[str, Any], budget_bytes: int) -> bool:
    if candidate.get("estimated_feasible") is not None:
        return bool(candidate["estimated_feasible"])
    estimated_peak = _optional_int(candidate.get("estimated_peak_bytes"))
    return estimated_peak is not None and estimated_peak <= budget_bytes


def _actual_feasible(candidate: Mapping[str, Any], budget_bytes: int) -> bool:
    if not bool(candidate.get("correctness_passed", True)):
        return False
    measured_peak = _optional_int(candidate.get("measured_peak_bytes"))
    return measured_peak is not None and measured_peak <= budget_bytes


def _is_oom(candidate: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(candidate.get(field) or "")
        for field in ("status", "error_type", "error_message", "failure_reason")
    ).lower()
    return (
        "outofmemory" in text
        or "out of memory" in text
        or "cuda oom" in text
        or "cuda_oom" in text
        or text.strip() == "oom"
    )


def _rank_candidates(candidates: Sequence[Mapping[str, Any]], budget_bytes: int) -> list[Mapping[str, Any]]:
    def key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        estimated_step = _optional_float(candidate.get("estimated_step_us"))
        estimated_peak = _optional_int(candidate.get("estimated_peak_bytes"))
        risk_score = _optional_float(candidate.get("risk_score"))
        confidence = _optional_float(candidate.get("confidence"))
        return (
            not _estimated_feasible(candidate, budget_bytes),
            math.inf if estimated_step is None else estimated_step,
            math.inf if estimated_peak is None else estimated_peak,
            math.inf if risk_score is None else risk_score,
            math.inf if confidence is None else -confidence,
            _candidate_plan_id(candidate),
        )

    return sorted(candidates, key=key)


def _protocol_candidate_us(candidate: Mapping[str, Any], record: Mapping[str, Any]) -> float:
    attempt_elapsed = _optional_float(candidate.get("validation_elapsed_us"))
    if attempt_elapsed is not None:
        return max(attempt_elapsed, 0.0)
    phase_metrics = candidate.get("phase_metrics") or {}
    elapsed = _optional_float(phase_metrics.get("candidate_validation_elapsed_us"))
    if elapsed is not None:
        return max(elapsed, 0.0)
    step_us = _optional_float(candidate.get("measured_step_us")) or 0.0
    repeats = _optional_int(phase_metrics.get("measurement_repeats"))
    if repeats is None:
        repeats = _optional_int(record.get("measurement_repeats"))
    warmups = _optional_int(phase_metrics.get("measurement_warmup_steps"))
    if warmups is None:
        warmups = _optional_int(record.get("measurement_warmup_steps"))
    return max(step_us, 0.0) * max((repeats or 1) + (warmups or 0), 1)


def _measurement_cost_us(
    record: Mapping[str, Any],
    all_candidates: Sequence[Mapping[str, Any]],
    selected_candidates: Sequence[Mapping[str, Any]],
) -> tuple[float, float, str]:
    all_elapsed = []
    for candidate in all_candidates:
        elapsed = _optional_float(candidate.get("validation_elapsed_us"))
        if elapsed is None:
            elapsed = _optional_float(
                (candidate.get("phase_metrics") or {}).get("candidate_validation_elapsed_us")
            )
        all_elapsed.append(elapsed)
    if all_candidates and all(value is not None for value in all_elapsed):
        elapsed_by_id = {
            _candidate_plan_id(candidate): float(elapsed)
            for candidate, elapsed in zip(all_candidates, all_elapsed)
            if elapsed is not None
        }
        return (
            sum(elapsed_by_id.values()),
            sum(elapsed_by_id.get(_candidate_plan_id(candidate), 0.0) for candidate in selected_candidates),
            "per_candidate_validation_elapsed",
        )

    all_protocol = sum(_protocol_candidate_us(candidate, record) for candidate in all_candidates)
    selected_protocol = sum(_protocol_candidate_us(candidate, record) for candidate in selected_candidates)
    observed_total = _optional_float(record.get("optimization_candidate_validation_measurement_us"))
    if observed_total is not None and all_protocol > 0.0:
        return (
            max(observed_total, 0.0),
            max(observed_total, 0.0) * selected_protocol / all_protocol,
            "observed_total_prorated_by_protocol_time",
        )
    return all_protocol, selected_protocol, "protocol_step_time_lower_bound"


def _shared_search_cost_us(
    record: Mapping[str, Any],
    full_measurement_cost_us: float,
) -> tuple[float, float, str]:
    """Return exhaustive total and the fixed cost shared by all search modes.

    Candidate validation is the only cost removed by simulation-only screening or
    reduced by Top-k confirmation. Capture, IR construction, executor construction,
    and candidate analysis remain part of every end-to-end search mode.
    """

    observed_total = _optional_float(record.get("optimization_total_us"))
    observed_validation = _optional_float(
        record.get("optimization_candidate_validation_measurement_us")
    )
    if observed_total is not None and observed_validation is not None:
        fixed = max(observed_total - observed_validation, 0.0)
        return max(observed_total, fixed), fixed, "observed_total_minus_observed_validation"
    if observed_total is not None:
        fixed = max(observed_total - full_measurement_cost_us, 0.0)
        return max(observed_total, fixed), fixed, "observed_total_minus_reconstructed_validation"

    component_fields = (
        "optimization_capture_us",
        "optimization_ir_build_us",
        "optimization_executor_build_us",
        "optimization_analysis_us",
    )
    components = [_optional_float(record.get(field)) for field in component_fields]
    if any(value is not None for value in components):
        fixed = sum(max(float(value), 0.0) for value in components if value is not None)
        return fixed + full_measurement_cost_us, fixed, "summed_fixed_components"
    return full_measurement_cost_us, 0.0, "validation_only_lower_bound"


def _random_topk_expectation(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget_bytes: int,
    top_k: int,
    oracle_time: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    """Exact expectation for uniformly sampling k candidates without replacement."""

    candidate_count = len(candidates)
    sample_count = min(top_k, candidate_count)
    if candidate_count == 0 or sample_count == 0:
        return {
            "any_feasible_probability": None,
            "best_time_hit_probability": None,
            "expected_time_regret_ratio": None,
            "expected_peak_regret_ratio": None,
        }
    feasible = sorted(
        (candidate for candidate in candidates if _actual_feasible(candidate, budget_bytes)),
        key=lambda candidate: (
            float(candidate["measured_step_us"]),
            int(candidate["measured_peak_bytes"]),
            _candidate_plan_id(candidate),
        ),
    )
    if not feasible or oracle_time is None:
        return {
            "any_feasible_probability": 0.0,
            "best_time_hit_probability": 0.0,
            "expected_time_regret_ratio": None,
            "expected_peak_regret_ratio": None,
        }
    total_subsets = math.comb(candidate_count, sample_count)
    infeasible_count = candidate_count - len(feasible)
    no_feasible_subsets = (
        math.comb(infeasible_count, sample_count)
        if infeasible_count >= sample_count
        else 0
    )
    any_feasible_probability = 1.0 - no_feasible_subsets / total_subsets
    oracle_step_us = float(oracle_time["measured_step_us"])
    oracle_peak_bytes = int(oracle_time["measured_peak_bytes"])
    expected_time_regret = 0.0
    expected_peak_regret = 0.0
    for faster_count, candidate in enumerate(feasible):
        remaining_without_faster = candidate_count - 1 - faster_count
        winning_subsets = (
            math.comb(remaining_without_faster, sample_count - 1)
            if remaining_without_faster >= sample_count - 1
            else 0
        )
        probability = winning_subsets / total_subsets
        expected_time_regret += probability * (
            (float(candidate["measured_step_us"]) - oracle_step_us) / oracle_step_us
        )
        expected_peak_regret += probability * (
            (int(candidate["measured_peak_bytes"]) - oracle_peak_bytes) / oracle_peak_bytes
        )
    return {
        "any_feasible_probability": any_feasible_probability,
        "best_time_hit_probability": sample_count / candidate_count,
        "expected_time_regret_ratio": expected_time_regret / any_feasible_probability,
        "expected_peak_regret_ratio": expected_peak_regret / any_feasible_probability,
    }


def _fw_only_no_global_gain(
    candidate: Mapping[str, Any],
    all_save: Mapping[str, Any] | None,
) -> bool:
    if all_save is None or _candidate_plan_id(candidate) == "all_save":
        return False
    candidate_metrics = candidate.get("phase_metrics") or {}
    all_save_metrics = all_save.get("phase_metrics") or {}
    candidate_fw = _optional_int(candidate_metrics.get("fw_peak_bytes"))
    all_save_fw = _optional_int(all_save_metrics.get("fw_peak_bytes"))
    candidate_overall = _optional_int(candidate_metrics.get("overall_peak_bytes"))
    if candidate_overall is None:
        candidate_overall = _optional_int(candidate.get("measured_peak_bytes"))
    all_save_overall = _optional_int(all_save_metrics.get("overall_peak_bytes"))
    if all_save_overall is None:
        all_save_overall = _optional_int(all_save.get("measured_peak_bytes"))
    if None in (candidate_fw, all_save_fw, candidate_overall, all_save_overall):
        return False
    return candidate_fw < all_save_fw and candidate_overall >= all_save_overall


def _record_key(record: Mapping[str, Any], index: int) -> str:
    return ":".join(
        (
            str(record.get("task_name") or "unknown"),
            str(record.get("variant_name") or "unknown"),
            str(record.get("microbatch_size") or "unknown"),
            str(record.get("budget_bytes") or "unknown"),
            str(record.get("matrix_pass_index") if record.get("matrix_pass_index") is not None else index),
        )
    )


def replay_record(record: Mapping[str, Any], *, top_k: int, index: int = 0) -> dict[str, Any] | None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if record.get("status") != "ok":
        return None
    measured_candidates = [
        candidate
        for candidate in (record.get("measured_plan_results") or ())
        if _optional_int(candidate.get("measured_peak_bytes")) is not None
        and _optional_float(candidate.get("measured_step_us")) is not None
    ]
    measured_ids = {_candidate_plan_id(candidate) for candidate in measured_candidates}
    failed_attempts = [
        attempt
        for attempt in (record.get("candidate_attempts") or ())
        if _candidate_plan_id(attempt) not in measured_ids
    ]
    candidates = [*measured_candidates, *failed_attempts]
    if not candidates:
        return None
    budget_bytes = int(record["budget_bytes"])
    ranked = _rank_candidates(candidates, budget_bytes)
    measured_topk = ranked[: min(top_k, len(ranked))]
    excluded = ranked[len(measured_topk) :]
    feasible = [candidate for candidate in candidates if _actual_feasible(candidate, budget_bytes)]
    feasible_topk = [candidate for candidate in measured_topk if _actual_feasible(candidate, budget_bytes)]
    oracle_time = min(
        feasible,
        key=lambda candidate: (
            float(candidate["measured_step_us"]),
            int(candidate["measured_peak_bytes"]),
            _candidate_plan_id(candidate),
        ),
        default=None,
    )
    oracle_peak = min(
        feasible,
        key=lambda candidate: (
            int(candidate["measured_peak_bytes"]),
            float(candidate["measured_step_us"]),
            _candidate_plan_id(candidate),
        ),
        default=None,
    )
    selected = min(
        feasible_topk,
        key=lambda candidate: (
            float(candidate["measured_step_us"]),
            int(candidate["measured_peak_bytes"]),
            _candidate_plan_id(candidate),
        ),
        default=None,
    )
    all_save = next((candidate for candidate in candidates if _candidate_plan_id(candidate) == "all_save"), None)
    full_cost_us, topk_cost_us, cost_source = _measurement_cost_us(record, candidates, measured_topk)
    full_total_us, shared_fixed_us, total_cost_source = _shared_search_cost_us(record, full_cost_us)
    topk_total_us = (
        full_total_us
        if len(measured_topk) == len(candidates)
        else shared_fixed_us + topk_cost_us
    )
    simulation_only = ranked[0]

    oracle_time_us = None if oracle_time is None else float(oracle_time["measured_step_us"])
    oracle_time_peak_bytes = None if oracle_time is None else int(oracle_time["measured_peak_bytes"])
    oracle_peak_bytes = None if oracle_peak is None else int(oracle_peak["measured_peak_bytes"])
    selected_time_us = None if selected is None else float(selected["measured_step_us"])
    selected_peak_bytes = None if selected is None else int(selected["measured_peak_bytes"])
    time_regret_us = None
    time_regret_ratio = None
    peak_regret_bytes = None
    peak_regret_ratio = None
    min_peak_gap_bytes = None
    min_peak_gap_ratio = None
    if selected_time_us is not None and oracle_time_us is not None:
        time_regret_us = selected_time_us - oracle_time_us
        time_regret_ratio = _safe_ratio(time_regret_us, oracle_time_us)
    if selected_peak_bytes is not None and oracle_time_peak_bytes is not None:
        peak_regret_bytes = selected_peak_bytes - oracle_time_peak_bytes
        peak_regret_ratio = _safe_ratio(float(peak_regret_bytes), float(oracle_time_peak_bytes))
    if selected_peak_bytes is not None and oracle_peak_bytes is not None:
        min_peak_gap_bytes = selected_peak_bytes - oracle_peak_bytes
        min_peak_gap_ratio = _safe_ratio(float(min_peak_gap_bytes), float(oracle_peak_bytes))

    oracle_time_id = None if oracle_time is None else _candidate_plan_id(oracle_time)
    oracle_peak_id = None if oracle_peak is None else _candidate_plan_id(oracle_peak)
    topk_ids = [_candidate_plan_id(candidate) for candidate in measured_topk]
    excluded_budget_violations = sum(
        1
        for candidate in excluded
        if not _is_oom(candidate)
        and _optional_int(candidate.get("measured_peak_bytes")) is not None
        and int(candidate["measured_peak_bytes"]) > budget_bytes
    )
    simulation_only_feasible = _actual_feasible(simulation_only, budget_bytes)
    simulation_only_time_regret_ratio = None
    simulation_only_peak_regret_ratio = None
    if simulation_only_feasible and oracle_time is not None:
        simulation_step_us = float(simulation_only["measured_step_us"])
        simulation_peak_bytes = int(simulation_only["measured_peak_bytes"])
        simulation_only_time_regret_ratio = _safe_ratio(
            simulation_step_us - float(oracle_time["measured_step_us"]),
            float(oracle_time["measured_step_us"]),
        )
        simulation_only_peak_regret_ratio = _safe_ratio(
            float(simulation_peak_bytes - int(oracle_time["measured_peak_bytes"])),
            float(oracle_time["measured_peak_bytes"]),
        )
    random_topk = _random_topk_expectation(
        candidates,
        budget_bytes=budget_bytes,
        top_k=top_k,
        oracle_time=oracle_time,
    )
    random_measurement_us = full_cost_us * len(measured_topk) / len(candidates)
    random_total_us = (
        full_total_us
        if len(measured_topk) == len(candidates)
        else shared_fixed_us + random_measurement_us
    )
    return {
        "record_key": _record_key(record, index),
        "task_name": str(record.get("task_name") or "unknown"),
        "variant_name": str(record.get("variant_name") or "unknown"),
        "microbatch_size": record.get("microbatch_size"),
        "budget_bytes": budget_bytes,
        "matrix_pass_index": record.get("matrix_pass_index"),
        "top_k": top_k,
        "full_candidate_count": len(candidates),
        "peakaware_measured_candidate_count": len(measured_topk),
        "candidate_measurements_saved": len(candidates) - len(measured_topk),
        "full_measurement_gpu_hours": full_cost_us / 3_600_000_000.0,
        "peakaware_measurement_gpu_hours": topk_cost_us / 3_600_000_000.0,
        "gpu_hours_saved": (full_cost_us - topk_cost_us) / 3_600_000_000.0,
        "gpu_hours_cost_source": cost_source,
        "full_total_search_gpu_hours": full_total_us / 3_600_000_000.0,
        "shared_fixed_search_gpu_hours": shared_fixed_us / 3_600_000_000.0,
        "simulation_only_total_search_gpu_hours": shared_fixed_us / 3_600_000_000.0,
        "counterfactual_topk_total_search_gpu_hours": topk_total_us / 3_600_000_000.0,
        "simulation_only_end_to_end_speedup": _safe_ratio(full_total_us, shared_fixed_us),
        "zero_validation_plan_selection_speedup": _safe_ratio(
            full_total_us,
            shared_fixed_us,
        ),
        "counterfactual_topk_end_to_end_speedup": _safe_ratio(full_total_us, topk_total_us),
        "counterfactual_topk_validation_speedup": _safe_ratio(full_cost_us, topk_cost_us),
        "total_search_cost_source": total_cost_source,
        "ranked_plan_ids": [_candidate_plan_id(candidate) for candidate in ranked],
        "topk_plan_ids": topk_ids,
        "excluded_plan_ids": [_candidate_plan_id(candidate) for candidate in excluded],
        "oracle_fastest_feasible_plan_id": oracle_time_id,
        "oracle_min_peak_feasible_plan_id": oracle_peak_id,
        "selected_plan_id": None if selected is None else _candidate_plan_id(selected),
        "simulation_only_selected_plan_id": _candidate_plan_id(simulation_only),
        "simulation_only_actual_feasible": simulation_only_feasible,
        "simulation_only_selected_is_oracle": (
            oracle_time_id is not None and _candidate_plan_id(simulation_only) == oracle_time_id
        ),
        "simulation_only_time_regret_ratio": simulation_only_time_regret_ratio,
        "simulation_only_peak_regret_ratio": simulation_only_peak_regret_ratio,
        "random_topk_any_feasible_probability": random_topk["any_feasible_probability"],
        "random_topk_best_time_hit_probability": random_topk["best_time_hit_probability"],
        "random_topk_expected_time_regret_ratio": random_topk["expected_time_regret_ratio"],
        "random_topk_expected_peak_regret_ratio": random_topk["expected_peak_regret_ratio"],
        "random_topk_measurement_gpu_hours": random_measurement_us / 3_600_000_000.0,
        "random_topk_total_search_gpu_hours": random_total_us / 3_600_000_000.0,
        "random_topk_end_to_end_speedup": _safe_ratio(full_total_us, random_total_us),
        "any_feasible_hit": bool(feasible_topk),
        "best_time_hit": oracle_time_id is not None and oracle_time_id in topk_ids,
        "best_peak_hit": oracle_peak_id is not None and oracle_peak_id in topk_ids,
        "selected_measured_feasible": selected is not None,
        "oracle_fastest_step_us": oracle_time_us,
        "oracle_fastest_peak_bytes": oracle_time_peak_bytes,
        "selected_step_us": selected_time_us,
        "time_regret_us": time_regret_us,
        "time_regret_ratio": time_regret_ratio,
        "oracle_min_peak_bytes": oracle_peak_bytes,
        "selected_peak_bytes": selected_peak_bytes,
        "peak_regret_bytes": peak_regret_bytes,
        "peak_regret_ratio": peak_regret_ratio,
        "min_peak_gap_bytes": min_peak_gap_bytes,
        "min_peak_gap_ratio": min_peak_gap_ratio,
        "excluded_actual_oom_count": sum(1 for candidate in excluded if _is_oom(candidate)),
        "excluded_actual_budget_violation_count": excluded_budget_violations,
        "excluded_fw_only_no_global_gain_count": sum(
            1
            for candidate in excluded
            if not _is_oom(candidate) and _fw_only_no_global_gain(candidate, all_save)
        ),
    }


def summarize_replay_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "record_count": 0,
            "full_measurement_candidate_count": 0,
            "peakaware_measured_candidate_count": 0,
            "candidate_measurements_saved": 0,
            "candidate_measurement_reduction_rate": None,
            "full_measurement_gpu_hours": 0.0,
            "peakaware_measurement_gpu_hours": 0.0,
            "gpu_hours_saved": 0.0,
            "gpu_hours_reduction_rate": None,
            "full_total_search_gpu_hours": 0.0,
            "shared_fixed_search_gpu_hours": 0.0,
            "simulation_only_total_search_gpu_hours": 0.0,
            "counterfactual_topk_total_search_gpu_hours": 0.0,
            "simulation_only_end_to_end_speedup": None,
            "zero_validation_plan_selection_speedup": None,
            "counterfactual_topk_end_to_end_speedup": None,
            "counterfactual_topk_validation_speedup": None,
            "random_topk_measurement_gpu_hours": 0.0,
            "random_topk_total_search_gpu_hours": 0.0,
            "random_topk_end_to_end_speedup": None,
            "random_topk_feasible_rate": None,
            "random_topk_best_time_hit_rate": None,
            "random_topk_mean_time_regret_ratio": None,
            "random_topk_mean_peak_regret_ratio": None,
            "time_regret_reduction_vs_random_rate": None,
            "mean_min_peak_gap_ratio": None,
            "p50_min_peak_gap_ratio": None,
            "p90_min_peak_gap_ratio": None,
            "max_min_peak_gap_ratio": None,
        }
    full_count = sum(int(row["full_candidate_count"]) for row in rows)
    peakaware_count = sum(int(row["peakaware_measured_candidate_count"]) for row in rows)
    full_gpu_hours = sum(float(row["full_measurement_gpu_hours"]) for row in rows)
    peakaware_gpu_hours = sum(float(row["peakaware_measurement_gpu_hours"]) for row in rows)
    full_total_hours = sum(float(row["full_total_search_gpu_hours"]) for row in rows)
    shared_fixed_hours = sum(float(row["shared_fixed_search_gpu_hours"]) for row in rows)
    topk_total_hours = sum(float(row["counterfactual_topk_total_search_gpu_hours"]) for row in rows)
    random_measurement_hours = sum(float(row["random_topk_measurement_gpu_hours"]) for row in rows)
    random_total_hours = sum(float(row["random_topk_total_search_gpu_hours"]) for row in rows)
    time_regrets = [float(row["time_regret_ratio"]) for row in rows if row.get("time_regret_ratio") is not None]
    peak_regrets = [float(row["peak_regret_ratio"]) for row in rows if row.get("peak_regret_ratio") is not None]
    min_peak_gaps = [
        float(row["min_peak_gap_ratio"])
        for row in rows
        if row.get("min_peak_gap_ratio") is not None
    ]
    simulation_time_regrets = [
        float(row["simulation_only_time_regret_ratio"])
        for row in rows
        if row.get("simulation_only_time_regret_ratio") is not None
    ]
    simulation_peak_regrets = [
        float(row["simulation_only_peak_regret_ratio"])
        for row in rows
        if row.get("simulation_only_peak_regret_ratio") is not None
    ]
    random_time_regrets = [
        float(row["random_topk_expected_time_regret_ratio"])
        for row in rows
        if row.get("random_topk_expected_time_regret_ratio") is not None
    ]
    random_peak_regrets = [
        float(row["random_topk_expected_peak_regret_ratio"])
        for row in rows
        if row.get("random_topk_expected_peak_regret_ratio") is not None
    ]
    random_hit_probabilities = [
        float(row["random_topk_best_time_hit_probability"])
        for row in rows
        if row.get("random_topk_best_time_hit_probability") is not None
    ]
    random_feasible_probabilities = [
        float(row["random_topk_any_feasible_probability"])
        for row in rows
        if row.get("random_topk_any_feasible_probability") is not None
    ]
    mean_time_regret = _mean(time_regrets)
    random_mean_time_regret = _mean(random_time_regrets)
    return {
        "record_count": len(rows),
        "full_measurement_candidate_count": full_count,
        "peakaware_measured_candidate_count": peakaware_count,
        "candidate_measurements_saved": full_count - peakaware_count,
        "candidate_measurement_reduction_rate": _safe_ratio(float(full_count - peakaware_count), float(full_count)),
        "full_measurement_gpu_hours": full_gpu_hours,
        "peakaware_measurement_gpu_hours": peakaware_gpu_hours,
        "gpu_hours_saved": full_gpu_hours - peakaware_gpu_hours,
        "gpu_hours_reduction_rate": _safe_ratio(full_gpu_hours - peakaware_gpu_hours, full_gpu_hours),
        "gpu_hours_cost_source_counts": dict(Counter(str(row["gpu_hours_cost_source"]) for row in rows)),
        "full_total_search_gpu_hours": full_total_hours,
        "shared_fixed_search_gpu_hours": shared_fixed_hours,
        "simulation_only_total_search_gpu_hours": shared_fixed_hours,
        "counterfactual_topk_total_search_gpu_hours": topk_total_hours,
        "simulation_only_end_to_end_speedup": _safe_ratio(full_total_hours, shared_fixed_hours),
        "zero_validation_plan_selection_speedup": _safe_ratio(
            full_total_hours,
            shared_fixed_hours,
        ),
        "counterfactual_topk_end_to_end_speedup": _safe_ratio(full_total_hours, topk_total_hours),
        "counterfactual_topk_validation_speedup": _safe_ratio(full_gpu_hours, peakaware_gpu_hours),
        "random_topk_measurement_gpu_hours": random_measurement_hours,
        "random_topk_total_search_gpu_hours": random_total_hours,
        "random_topk_end_to_end_speedup": _safe_ratio(full_total_hours, random_total_hours),
        "simulation_only_total_search_reduction_rate": _safe_ratio(
            full_total_hours - shared_fixed_hours,
            full_total_hours,
        ),
        "counterfactual_topk_total_search_reduction_rate": _safe_ratio(
            full_total_hours - topk_total_hours,
            full_total_hours,
        ),
        "total_search_cost_source_counts": dict(
            Counter(str(row["total_search_cost_source"]) for row in rows)
        ),
        "simulation_only_actual_feasible_rate": _mean(
            [1.0 if row.get("simulation_only_actual_feasible") else 0.0 for row in rows]
        ),
        "simulation_only_oracle_hit_rate": _mean(
            [1.0 if row.get("simulation_only_selected_is_oracle") else 0.0 for row in rows]
        ),
        "simulation_only_mean_time_regret_ratio": _mean(simulation_time_regrets),
        "simulation_only_p90_time_regret_ratio": _percentile(simulation_time_regrets, 0.90),
        "simulation_only_max_time_regret_ratio": (
            None if not simulation_time_regrets else max(simulation_time_regrets)
        ),
        "simulation_only_mean_peak_regret_ratio": _mean(simulation_peak_regrets),
        "random_topk_feasible_rate": _mean(random_feasible_probabilities),
        "random_topk_best_time_hit_rate": _mean(random_hit_probabilities),
        "random_topk_mean_time_regret_ratio": random_mean_time_regret,
        "random_topk_mean_peak_regret_ratio": _mean(random_peak_regrets),
        "time_regret_reduction_vs_random_rate": (
            None
            if mean_time_regret is None or random_mean_time_regret is None
            else _safe_ratio(random_mean_time_regret - mean_time_regret, random_mean_time_regret)
        ),
        "oracle_feasible_record_count": sum(
            1 for row in rows if row.get("oracle_fastest_feasible_plan_id") is not None
        ),
        "topk_feasible_record_count": sum(1 for row in rows if bool(row.get("any_feasible_hit"))),
        "topk_feasible_rate": _mean([1.0 if row.get("any_feasible_hit") else 0.0 for row in rows]),
        "best_time_hit_rate": _mean([1.0 if row.get("best_time_hit") else 0.0 for row in rows]),
        "best_peak_hit_rate": _mean([1.0 if row.get("best_peak_hit") else 0.0 for row in rows]),
        "mean_time_regret_ratio": mean_time_regret,
        "p50_time_regret_ratio": _percentile(time_regrets, 0.50),
        "p90_time_regret_ratio": _percentile(time_regrets, 0.90),
        "max_time_regret_ratio": None if not time_regrets else max(time_regrets),
        "mean_peak_regret_ratio": _mean(peak_regrets),
        "p50_peak_regret_ratio": _percentile(peak_regrets, 0.50),
        "p90_peak_regret_ratio": _percentile(peak_regrets, 0.90),
        "max_peak_regret_ratio": None if not peak_regrets else max(peak_regrets),
        "mean_min_peak_gap_ratio": _mean(min_peak_gaps),
        "p50_min_peak_gap_ratio": _percentile(min_peak_gaps, 0.50),
        "p90_min_peak_gap_ratio": _percentile(min_peak_gaps, 0.90),
        "max_min_peak_gap_ratio": None if not min_peak_gaps else max(min_peak_gaps),
        "excluded_actual_oom_count": sum(int(row["excluded_actual_oom_count"]) for row in rows),
        "excluded_actual_budget_violation_count": sum(
            int(row["excluded_actual_budget_violation_count"]) for row in rows
        ),
        "excluded_fw_only_no_global_gain_count": sum(
            int(row["excluded_fw_only_no_global_gain_count"]) for row in rows
        ),
    }


def analyze_search_efficiency(
    records: Iterable[Mapping[str, Any]],
    *,
    selected_top_k: int = 2,
    top_k_values: Sequence[int] = (1, 2, 3, 4),
    variants: frozenset[str] | None = None,
    tasks: frozenset[str] | None = None,
) -> dict[str, Any]:
    records = tuple(records)
    filtered = [
        record
        for record in records
        if (variants is None or str(record.get("variant_name")) in variants)
        and (tasks is None or str(record.get("task_name")) in tasks)
    ]
    normalized_top_k = tuple(sorted({value for value in top_k_values if value > 0} | {selected_top_k}))
    rows_by_k = {
        top_k: tuple(
            row
            for index, record in enumerate(filtered)
            if (row := replay_record(record, top_k=top_k, index=index)) is not None
        )
        for top_k in normalized_top_k
    }
    selected_rows = rows_by_k[selected_top_k]
    by_task_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_task_rows[str(row["task_name"])].append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_mode": "counterfactual_topk_replay_over_exhaustively_measured_candidates",
        "selected_top_k": selected_top_k,
        "filters": {
            "variants": None if variants is None else sorted(variants),
            "tasks": None if tasks is None else sorted(tasks),
        },
        "definitions": {
            "full_measurement": "all measured_plan_results candidates in each successful record",
            "peakaware_measurement": "first k candidates ranked by estimated feasibility then estimated step time",
            "time_regret": "selected Top-k measured-feasible step time relative to exhaustive fastest measured-feasible candidate",
            "peak_regret": "selected Top-k peak relative to the same exhaustive fastest measured-feasible oracle used by time regret",
            "min_peak_gap": "selected Top-k peak relative to the exhaustive minimum-peak measured-feasible candidate",
            "fw_only_no_global_gain": "candidate FW peak is lower than all-save while complete-step peak is not lower",
            "gpu_hours": (
                "per-candidate validation elapsed when available; otherwise observed aggregate validation time is "
                "prorated by measured protocol time, or protocol step time is used as a lower bound"
            ),
            "exhaustive_end_to_end": (
                "observed optimization_total_us for the run that validates every generated strategy"
            ),
            "simulation_only_end_to_end": (
                "the exhaustive run's shared capture, IR, executor, and analysis cost with all candidate "
                "validation removed; this is end-to-end plan-selection cost and excludes later realization "
                "or dry-run of the selected executable"
            ),
            "counterfactual_topk_end_to_end": (
                "the same shared fixed cost plus validation elapsed for the first k simulation-ranked strategies"
            ),
            "speedup": "exhaustive all-strategy actual-run wall time divided by the compared search-mode wall time",
            "random_topk": (
                "exact expectation under uniform sampling without replacement from the same generated candidate set"
            ),
        },
        "aggregate": summarize_replay_rows(selected_rows),
        "grouped_zero_validation_selection": _grouped_zero_validation_selection(filtered),
        "by_k": {str(top_k): summarize_replay_rows(rows) for top_k, rows in rows_by_k.items()},
        "by_task": {
            task_name: summarize_replay_rows(task_rows)
            for task_name, task_rows in sorted(by_task_rows.items())
        },
        "rows": list(selected_rows),
    }


def _paired_record_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("task_name"),
        record.get("variant_name"),
        record.get("microbatch_size"),
        record.get("budget_bytes"),
        record.get("matrix_pass_index", 0),
    )


def _selection_group_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("task_name"),
        record.get("variant_name"),
        record.get("microbatch_size"),
        record.get("budget_bytes"),
    )


def _aggregate_group_plans(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"step_us": [], "peak_bytes": [], "correctness": []}
    )
    for record in records:
        for candidate in record.get("measured_plan_results") or ():
            step_us = _optional_float(candidate.get("measured_step_us"))
            peak_bytes = _optional_int(candidate.get("measured_peak_bytes"))
            if step_us is None or peak_bytes is None:
                continue
            plan_id = _candidate_plan_id(candidate)
            observations[plan_id]["step_us"].append(step_us)
            observations[plan_id]["peak_bytes"].append(peak_bytes)
            observations[plan_id]["correctness"].append(bool(candidate.get("correctness_passed", True)))
    return {
        plan_id: {
            "plan_id": plan_id,
            "observation_count": len(values["step_us"]),
            "median_step_us": _percentile(values["step_us"], 0.50),
            "median_peak_bytes": _percentile(values["peak_bytes"], 0.50),
            "max_peak_bytes": max(values["peak_bytes"]),
            "correctness_passed": all(values["correctness"]),
        }
        for plan_id, values in observations.items()
        if values["step_us"]
    }


def _grouped_final_selection(
    exhaustive_records: Sequence[Mapping[str, Any]],
    guided_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exhaustive_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    guided_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in exhaustive_records:
        if record.get("status") == "ok":
            exhaustive_groups[_selection_group_identity(record)].append(record)
    for record in guided_records:
        if record.get("status") == "ok":
            guided_groups[_selection_group_identity(record)].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(exhaustive_groups.keys() & guided_groups.keys(), key=str):
        budget_bytes = int(key[3])
        exhaustive_plans = _aggregate_group_plans(exhaustive_groups[key])
        guided_plans = _aggregate_group_plans(guided_groups[key])
        exhaustive_feasible = [
            plan
            for plan in exhaustive_plans.values()
            if plan["correctness_passed"] and int(plan["max_peak_bytes"]) <= budget_bytes
        ]
        guided_feasible = [
            plan
            for plan in guided_plans.values()
            if plan["correctness_passed"] and int(plan["max_peak_bytes"]) <= budget_bytes
        ]
        oracle = min(
            exhaustive_feasible,
            key=lambda plan: (
                float(plan["median_step_us"]),
                float(plan["median_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        guided_selected = min(
            guided_feasible,
            key=lambda plan: (
                float(plan["median_step_us"]),
                float(plan["median_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        if guided_selected is None and not guided_plans:
            selected_counts = Counter(
                str(record["selected_plan_id"])
                for record in guided_groups[key]
                if record.get("selected_plan_id") is not None
            )
            if selected_counts:
                selected_plan_id = min(
                    selected_counts,
                    key=lambda plan_id: (-selected_counts[plan_id], plan_id),
                )
                guided_selected = {"plan_id": selected_plan_id}
        reference = (
            None
            if guided_selected is None
            else exhaustive_plans.get(str(guided_selected["plan_id"]))
        )
        time_regret_ratio = None
        peak_regret_ratio = None
        if oracle is not None and reference is not None:
            oracle_step = float(oracle["median_step_us"])
            oracle_peak = float(oracle["median_peak_bytes"])
            time_regret_ratio = _safe_ratio(
                float(reference["median_step_us"]) - oracle_step,
                oracle_step,
            )
            peak_regret_ratio = _safe_ratio(
                float(reference["median_peak_bytes"]) - oracle_peak,
                oracle_peak,
            )
        rows.append(
            {
                "task_name": key[0],
                "variant_name": key[1],
                "microbatch_size": key[2],
                "budget_bytes": budget_bytes,
                "exhaustive_pass_count": len(exhaustive_groups[key]),
                "guided_pass_count": len(guided_groups[key]),
                "oracle_plan_id": None if oracle is None else oracle["plan_id"],
                "guided_selected_plan_id": None
                if guided_selected is None
                else guided_selected["plan_id"],
                "guided_selected_reference_available": reference is not None,
                "guided_selected_is_oracle": None
                if oracle is None or guided_selected is None
                else oracle["plan_id"] == guided_selected["plan_id"],
                "time_regret_ratio": time_regret_ratio,
                "peak_regret_ratio": peak_regret_ratio,
                "oracle_median_step_us": None if oracle is None else oracle["median_step_us"],
                "guided_reference_median_step_us": None
                if reference is None
                else reference["median_step_us"],
                "oracle_median_peak_bytes": None if oracle is None else oracle["median_peak_bytes"],
                "guided_reference_median_peak_bytes": None
                if reference is None
                else reference["median_peak_bytes"],
            }
        )
    time_regrets = [
        float(row["time_regret_ratio"])
        for row in rows
        if row["time_regret_ratio"] is not None
    ]
    peak_regrets = [
        float(row["peak_regret_ratio"])
        for row in rows
        if row["peak_regret_ratio"] is not None
    ]
    oracle_flags = [
        bool(row["guided_selected_is_oracle"])
        for row in rows
        if row["guided_selected_is_oracle"] is not None
    ]
    return {
        "group_count": len(rows),
        "selection_reference_count": len(time_regrets),
        "guided_selected_oracle_rate": _mean([1.0 if value else 0.0 for value in oracle_flags]),
        "mean_time_regret_ratio": _mean(time_regrets),
        "p50_time_regret_ratio": _percentile(time_regrets, 0.50),
        "max_time_regret_ratio": None if not time_regrets else max(time_regrets),
        "mean_peak_regret_ratio": _mean(peak_regrets),
        "max_peak_regret_ratio": None if not peak_regrets else max(peak_regrets),
        "rows": rows,
    }


def _grouped_shortlist_replay(
    exhaustive_records: Sequence[Mapping[str, Any]],
    guided_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exhaustive_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    guided_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in exhaustive_records:
        if record.get("status") == "ok":
            exhaustive_groups[_selection_group_identity(record)].append(record)
    for record in guided_records:
        if record.get("status") == "ok":
            guided_groups[_selection_group_identity(record)].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(exhaustive_groups.keys() & guided_groups.keys(), key=str):
        budget_bytes = int(key[3])
        exhaustive_plans = _aggregate_group_plans(exhaustive_groups[key])
        exhaustive_feasible = [
            plan
            for plan in exhaustive_plans.values()
            if plan["correctness_passed"] and int(plan["max_peak_bytes"]) <= budget_bytes
        ]
        oracle = min(
            exhaustive_feasible,
            key=lambda plan: (
                float(plan["median_step_us"]),
                float(plan["median_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        shortlist_counts: Counter[tuple[str, ...]] = Counter()
        for record in guided_groups[key]:
            attempts = record.get("candidate_attempts") or ()
            plan_ids = tuple(sorted({_candidate_plan_id(item) for item in attempts}))
            if not plan_ids and record.get("selected_plan_id") is not None:
                plan_ids = (str(record["selected_plan_id"]),)
            if plan_ids:
                shortlist_counts[plan_ids] += 1
        shortlist_ids = (
            ()
            if not shortlist_counts
            else min(
                shortlist_counts,
                key=lambda plan_ids: (-shortlist_counts[plan_ids], plan_ids),
            )
        )
        shortlist_feasible = [
            plan
            for plan in exhaustive_feasible
            if str(plan["plan_id"]) in shortlist_ids
        ]
        replay_selected = min(
            shortlist_feasible,
            key=lambda plan: (
                float(plan["median_step_us"]),
                float(plan["median_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        regret = None
        if oracle is not None and replay_selected is not None:
            regret = _safe_ratio(
                float(replay_selected["median_step_us"])
                - float(oracle["median_step_us"]),
                float(oracle["median_step_us"]),
            )
        rows.append(
            {
                "task_name": key[0],
                "variant_name": key[1],
                "microbatch_size": key[2],
                "budget_bytes": budget_bytes,
                "shortlist_plan_ids": shortlist_ids,
                "shortlist_observation_count": shortlist_counts.get(shortlist_ids, 0),
                "oracle_plan_id": None if oracle is None else oracle["plan_id"],
                "oracle_in_shortlist": (
                    None if oracle is None else str(oracle["plan_id"]) in shortlist_ids
                ),
                "replay_selected_plan_id": None
                if replay_selected is None
                else replay_selected["plan_id"],
                "time_regret_ratio": regret,
            }
        )
    regrets = [
        float(row["time_regret_ratio"])
        for row in rows
        if row["time_regret_ratio"] is not None
    ]
    coverage = [
        bool(row["oracle_in_shortlist"])
        for row in rows
        if row["oracle_in_shortlist"] is not None
    ]
    return {
        "group_count": len(rows),
        "oracle_coverage_rate": _mean([1.0 if value else 0.0 for value in coverage]),
        "mean_time_regret_ratio": _mean(regrets),
        "p50_time_regret_ratio": _percentile(regrets, 0.50),
        "max_time_regret_ratio": None if not regrets else max(regrets),
        "rows": rows,
    }


def _grouped_zero_validation_selection(
    exhaustive_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in exhaustive_records:
        if record.get("status") == "ok":
            groups[_selection_group_identity(record)].append(record)

    rows: list[dict[str, Any]] = []
    for key, group_records in sorted(groups.items(), key=lambda item: str(item[0])):
        budget_bytes = int(key[3])
        observations: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: {
                "estimated_step_us": [],
                "estimated_peak_bytes": [],
                "measured_step_us": [],
                "measured_peak_bytes": [],
                "correctness": [],
            }
        )
        for record in group_records:
            candidates = record.get("candidate_attempts") or record.get("measured_plan_results") or ()
            for candidate in candidates:
                plan_id = _candidate_plan_id(candidate)
                for field, parser in (
                    ("estimated_step_us", _optional_float),
                    ("estimated_peak_bytes", _optional_int),
                    ("measured_step_us", _optional_float),
                    ("measured_peak_bytes", _optional_int),
                ):
                    value = parser(candidate.get(field))
                    if value is not None:
                        observations[plan_id][field].append(value)
                observations[plan_id]["correctness"].append(
                    bool(candidate.get("correctness_passed", True))
                )

        plans = []
        for plan_id, values in observations.items():
            if not values["estimated_step_us"] or not values["estimated_peak_bytes"]:
                continue
            plans.append(
                {
                    "plan_id": plan_id,
                    "median_estimated_step_us": _percentile(
                        values["estimated_step_us"], 0.50
                    ),
                    "max_estimated_peak_bytes": max(values["estimated_peak_bytes"]),
                    "median_measured_step_us": _percentile(values["measured_step_us"], 0.50),
                    "median_measured_peak_bytes": _percentile(
                        values["measured_peak_bytes"], 0.50
                    ),
                    "max_measured_peak_bytes": None
                    if not values["measured_peak_bytes"]
                    else max(values["measured_peak_bytes"]),
                    "correctness_passed": all(values["correctness"]),
                }
            )

        predicted_feasible = [
            plan
            for plan in plans
            if int(plan["max_estimated_peak_bytes"]) <= budget_bytes
        ]
        actual_feasible = [
            plan
            for plan in plans
            if plan["correctness_passed"]
            and plan["median_measured_step_us"] is not None
            and plan["max_measured_peak_bytes"] is not None
            and int(plan["max_measured_peak_bytes"]) <= budget_bytes
        ]
        selected = min(
            predicted_feasible,
            key=lambda plan: (
                float(plan["median_estimated_step_us"]),
                int(plan["max_estimated_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        oracle = min(
            actual_feasible,
            key=lambda plan: (
                float(plan["median_measured_step_us"]),
                float(plan["median_measured_peak_bytes"]),
                str(plan["plan_id"]),
            ),
            default=None,
        )
        selected_actual_feasible = (
            selected is not None
            and selected["correctness_passed"]
            and selected["median_measured_step_us"] is not None
            and selected["max_measured_peak_bytes"] is not None
            and int(selected["max_measured_peak_bytes"]) <= budget_bytes
        )
        time_regret_ratio = None
        if selected_actual_feasible and oracle is not None:
            oracle_step = float(oracle["median_measured_step_us"])
            time_regret_ratio = _safe_ratio(
                float(selected["median_measured_step_us"]) - oracle_step,
                oracle_step,
            )
        rows.append(
            {
                "task_name": key[0],
                "variant_name": key[1],
                "microbatch_size": key[2],
                "budget_bytes": budget_bytes,
                "pass_count": len(group_records),
                "simulator_selected_plan_id": None if selected is None else selected["plan_id"],
                "oracle_plan_id": None if oracle is None else oracle["plan_id"],
                "simulator_selected_actual_feasible": selected_actual_feasible,
                "simulator_selected_is_oracle": None
                if selected is None or oracle is None
                else selected["plan_id"] == oracle["plan_id"],
                "time_regret_ratio": time_regret_ratio,
                "simulator_median_estimated_step_us": None
                if selected is None
                else selected["median_estimated_step_us"],
                "simulator_reference_median_measured_step_us": None
                if selected is None
                else selected["median_measured_step_us"],
                "oracle_median_measured_step_us": None
                if oracle is None
                else oracle["median_measured_step_us"],
            }
        )

    regrets = [
        float(row["time_regret_ratio"])
        for row in rows
        if row["time_regret_ratio"] is not None
    ]
    oracle_flags = [
        bool(row["simulator_selected_is_oracle"])
        for row in rows
        if row["simulator_selected_is_oracle"] is not None
    ]
    return {
        "group_count": len(rows),
        "selection_reference_count": len(regrets),
        "actual_feasible_rate": _mean(
            [
                1.0 if row["simulator_selected_actual_feasible"] else 0.0
                for row in rows
            ]
        ),
        "oracle_hit_rate": _mean([1.0 if value else 0.0 for value in oracle_flags]),
        "mean_time_regret_ratio": _mean(regrets),
        "p50_time_regret_ratio": _percentile(regrets, 0.50),
        "max_time_regret_ratio": None if not regrets else max(regrets),
        "rows": rows,
    }


def _actual_attempt_count(record: Mapping[str, Any]) -> int:
    attempts = record.get("candidate_attempts") or ()
    if attempts:
        return len(attempts)
    return _actual_measurement_count(record)


def _actual_measurement_count(record: Mapping[str, Any]) -> int:
    return len(record.get("measured_plan_results") or ())


def _actual_validation_us(record: Mapping[str, Any]) -> tuple[float | None, str]:
    observed = _optional_float(record.get("optimization_candidate_validation_measurement_us"))
    if observed is not None:
        return observed, "observed_candidate_validation_wall_time"
    attempts = record.get("candidate_attempts") or ()
    elapsed = [_optional_float(attempt.get("validation_elapsed_us")) for attempt in attempts]
    if attempts and all(value is not None for value in elapsed):
        return sum(float(value) for value in elapsed if value is not None), "summed_candidate_validation_elapsed"
    return None, "unavailable"


def _actual_total_search_us(record: Mapping[str, Any]) -> float | None:
    return _optional_float(record.get("optimization_total_us"))


def compare_actual_measurement_runs(
    exhaustive_records: Iterable[Mapping[str, Any]],
    guided_records: Iterable[Mapping[str, Any]],
    *,
    guided_selection_policy: str = "ranked",
) -> dict[str, Any]:
    if guided_selection_policy not in {
        "ranked",
        "structural_diverse",
        "simulation_only_ranked",
    }:
        raise ValueError(
            "guided_selection_policy must be one of: ranked, structural_diverse, "
            "simulation_only_ranked"
        )
    exhaustive_records = tuple(exhaustive_records)
    guided_records = tuple(guided_records)
    exhaustive_by_key = {
        _paired_record_identity(record): record
        for record in exhaustive_records
        if record.get("status") == "ok"
    }
    guided_by_key = {
        _paired_record_identity(record): record
        for record in guided_records
        if record.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(exhaustive_by_key.keys() & guided_by_key.keys(), key=str):
        exhaustive = exhaustive_by_key[key]
        guided = guided_by_key[key]
        full_count = _actual_measurement_count(exhaustive)
        guided_count = _actual_measurement_count(guided)
        full_attempt_count = _actual_attempt_count(exhaustive)
        guided_attempt_count = _actual_attempt_count(guided)
        full_us, full_source = _actual_validation_us(exhaustive)
        guided_us, guided_source = _actual_validation_us(guided)
        full_total_us = _actual_total_search_us(exhaustive)
        guided_total_us = _actual_total_search_us(guided)
        budget_bytes = int(exhaustive["budget_bytes"])
        exhaustive_candidates = [
            candidate
            for candidate in (exhaustive.get("measured_plan_results") or ())
            if _optional_int(candidate.get("measured_peak_bytes")) is not None
            and _optional_float(candidate.get("measured_step_us")) is not None
        ]
        oracle_candidates = [
            candidate for candidate in exhaustive_candidates if _actual_feasible(candidate, budget_bytes)
        ]
        oracle = min(
            oracle_candidates,
            key=lambda candidate: (
                float(candidate["measured_step_us"]),
                int(candidate["measured_peak_bytes"]),
                _candidate_plan_id(candidate),
            ),
            default=None,
        )
        guided_plan_id = None if guided.get("selected_plan_id") is None else str(guided["selected_plan_id"])
        guided_reference = next(
            (
                candidate
                for candidate in exhaustive_candidates
                if _candidate_plan_id(candidate) == guided_plan_id
            ),
            None,
        )
        guided_reference_feasible = (
            guided_reference is not None and _actual_feasible(guided_reference, budget_bytes)
        )
        guided_time_regret_us = None
        guided_time_regret_ratio = None
        guided_peak_regret_bytes = None
        guided_peak_regret_ratio = None
        if oracle is not None and guided_reference_feasible:
            oracle_step = float(oracle["measured_step_us"])
            oracle_peak = int(oracle["measured_peak_bytes"])
            guided_step = float(guided_reference["measured_step_us"])
            guided_peak = int(guided_reference["measured_peak_bytes"])
            guided_time_regret_us = guided_step - oracle_step
            guided_time_regret_ratio = _safe_ratio(guided_time_regret_us, oracle_step)
            guided_peak_regret_bytes = guided_peak - oracle_peak
            guided_peak_regret_ratio = _safe_ratio(float(guided_peak_regret_bytes), float(oracle_peak))
        ranked = _rank_candidates(exhaustive_candidates, budget_bytes)
        replay_count = (
            1
            if guided_selection_policy == "simulation_only_ranked"
            else guided_attempt_count
        )
        replay_ids = {
            _candidate_plan_id(candidate)
            for candidate in ranked[: min(replay_count, len(ranked))]
        }
        guided_attempts = guided.get("candidate_attempts") or ()
        if guided_selection_policy == "simulation_only_ranked":
            actual_guided_ids = set() if guided_plan_id is None else {guided_plan_id}
        else:
            actual_guided_ids = {
                _candidate_plan_id(candidate)
                for candidate in (
                    guided_attempts
                    if guided_attempts
                    else (guided.get("measured_plan_results") or ())
                )
            }
        shortlist_candidates = [
            candidate
            for candidate in exhaustive_candidates
            if _candidate_plan_id(candidate) in actual_guided_ids
        ]
        shortlist_feasible = [
            candidate
            for candidate in shortlist_candidates
            if _actual_feasible(candidate, budget_bytes)
        ]
        shortlist_selected = min(
            shortlist_feasible,
            key=lambda candidate: (
                float(candidate["measured_step_us"]),
                int(candidate["measured_peak_bytes"]),
                _candidate_plan_id(candidate),
            ),
            default=None,
        )
        shortlist_replay_regret_ratio = None
        if oracle is not None and shortlist_selected is not None:
            shortlist_replay_regret_ratio = _safe_ratio(
                float(shortlist_selected["measured_step_us"])
                - float(oracle["measured_step_us"]),
                float(oracle["measured_step_us"]),
            )
        exhaustive_attempts = exhaustive.get("candidate_attempts") or ()
        exhaustive_attempt_ids = {
            _candidate_plan_id(candidate)
            for candidate in (
                exhaustive_attempts
                if exhaustive_attempts
                else exhaustive_candidates
            )
        }
        ranked_membership_match = (
            replay_ids == actual_guided_ids
            if guided_selection_policy in {"ranked", "simulation_only_ranked"}
            else None
        )
        rows.append(
            {
                "task_name": key[0],
                "variant_name": key[1],
                "microbatch_size": key[2],
                "budget_bytes": key[3],
                "matrix_pass_index": key[4],
                "full_measurement_candidate_count": full_count,
                "peakaware_measured_candidate_count": guided_count,
                "full_candidate_attempt_count": full_attempt_count,
                "peakaware_candidate_attempt_count": guided_attempt_count,
                "candidate_measurements_saved": full_count - guided_count,
                "candidate_measurement_reduction_rate": _safe_ratio(
                    float(full_count - guided_count), float(full_count)
                ),
                "full_measurement_gpu_hours": None if full_us is None else full_us / 3_600_000_000.0,
                "peakaware_measurement_gpu_hours": None if guided_us is None else guided_us / 3_600_000_000.0,
                "gpu_hours_saved": None
                if full_us is None or guided_us is None
                else (full_us - guided_us) / 3_600_000_000.0,
                "gpu_hours_reduction_rate": None
                if full_us is None or guided_us is None
                else _safe_ratio(full_us - guided_us, full_us),
                "candidate_validation_speedup": None
                if full_us is None or guided_us is None
                else _safe_ratio(full_us, guided_us),
                "full_total_search_gpu_hours": None
                if full_total_us is None
                else full_total_us / 3_600_000_000.0,
                "peakaware_total_search_gpu_hours": None
                if guided_total_us is None
                else guided_total_us / 3_600_000_000.0,
                "total_search_gpu_hours_saved": None
                if full_total_us is None or guided_total_us is None
                else (full_total_us - guided_total_us) / 3_600_000_000.0,
                "total_search_gpu_hours_reduction_rate": None
                if full_total_us is None or guided_total_us is None
                else _safe_ratio(full_total_us - guided_total_us, full_total_us),
                "total_search_speedup": None
                if full_total_us is None or guided_total_us is None
                else _safe_ratio(full_total_us, guided_total_us),
                "all_strategy_actual_run_speedup": None
                if full_total_us is None or guided_total_us is None
                else _safe_ratio(full_total_us, guided_total_us),
                "full_cost_source": full_source,
                "guided_cost_source": guided_source,
                "full_selected_plan_id": exhaustive.get("selected_plan_id"),
                "guided_selected_plan_id": guided_plan_id,
                "oracle_fastest_feasible_plan_id": None if oracle is None else _candidate_plan_id(oracle),
                "guided_selection_reference_available": guided_reference is not None,
                "guided_selection_reference_feasible": guided_reference_feasible,
                "guided_selected_is_oracle": (
                    None if oracle is None or guided_reference is None else guided_plan_id == _candidate_plan_id(oracle)
                ),
                "guided_selection_time_regret_us": guided_time_regret_us,
                "guided_selection_time_regret_ratio": guided_time_regret_ratio,
                "guided_selection_peak_regret_bytes": guided_peak_regret_bytes,
                "guided_selection_peak_regret_ratio": guided_peak_regret_ratio,
                "guided_selection_policy": guided_selection_policy,
                "ranked_topk_membership_match": ranked_membership_match,
                "topk_membership_match": ranked_membership_match,
                "guided_candidates_covered_by_exhaustive": actual_guided_ids.issubset(
                    exhaustive_attempt_ids
                ),
                "guided_shortlist_oracle_covered": (
                    None
                    if oracle is None
                    else _candidate_plan_id(oracle) in actual_guided_ids
                ),
                "guided_shortlist_replay_selected_plan_id": None
                if shortlist_selected is None
                else _candidate_plan_id(shortlist_selected),
                "guided_shortlist_replay_time_regret_ratio": shortlist_replay_regret_ratio,
                "guided_measured_peak_bytes": guided.get("measured_peak_bytes"),
                "guided_measured_step_us": guided.get("measured_step_us"),
            }
        )
    full_count = sum(int(row["full_measurement_candidate_count"]) for row in rows)
    guided_count = sum(int(row["peakaware_measured_candidate_count"]) for row in rows)
    cost_rows = [
        row
        for row in rows
        if row["full_measurement_gpu_hours"] is not None
        and row["peakaware_measurement_gpu_hours"] is not None
    ]
    full_hours = sum(float(row["full_measurement_gpu_hours"]) for row in cost_rows)
    guided_hours = sum(float(row["peakaware_measurement_gpu_hours"]) for row in cost_rows)
    total_cost_rows = [
        row
        for row in rows
        if row["full_total_search_gpu_hours"] is not None
        and row["peakaware_total_search_gpu_hours"] is not None
    ]
    full_total_hours = sum(float(row["full_total_search_gpu_hours"]) for row in total_cost_rows)
    guided_total_hours = sum(float(row["peakaware_total_search_gpu_hours"]) for row in total_cost_rows)
    guided_time_regrets = [
        float(row["guided_selection_time_regret_ratio"])
        for row in rows
        if row["guided_selection_time_regret_ratio"] is not None
    ]
    guided_peak_regrets = [
        float(row["guided_selection_peak_regret_ratio"])
        for row in rows
        if row["guided_selection_peak_regret_ratio"] is not None
    ]
    exhaustive_keys = set(exhaustive_by_key)
    guided_keys = set(guided_by_key)
    guided_oracle_flags = [
        bool(row["guided_selected_is_oracle"])
        for row in rows
        if row["guided_selected_is_oracle"] is not None
    ]
    paired_validation_speedups = [
        float(row["candidate_validation_speedup"])
        for row in rows
        if row["candidate_validation_speedup"] is not None
    ]
    paired_total_speedups = [
        float(row["total_search_speedup"])
        for row in rows
        if row["total_search_speedup"] is not None
    ]
    shortlist_replay_regrets = [
        float(row["guided_shortlist_replay_time_regret_ratio"])
        for row in rows
        if row["guided_shortlist_replay_time_regret_ratio"] is not None
    ]
    by_task_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_rows[str(row["task_name"])].append(row)

    def summarize_actual_rows(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        validation_rows = [
            row
            for row in task_rows
            if row["full_measurement_gpu_hours"] is not None
            and row["peakaware_measurement_gpu_hours"] is not None
        ]
        total_rows = [
            row
            for row in task_rows
            if row["full_total_search_gpu_hours"] is not None
            and row["peakaware_total_search_gpu_hours"] is not None
        ]
        task_full_validation_hours = sum(
            float(row["full_measurement_gpu_hours"])
            for row in validation_rows
        )
        task_guided_validation_hours = sum(
            float(row["peakaware_measurement_gpu_hours"])
            for row in validation_rows
        )
        task_full_total_hours = sum(
            float(row["full_total_search_gpu_hours"])
            for row in total_rows
        )
        task_guided_total_hours = sum(
            float(row["peakaware_total_search_gpu_hours"])
            for row in total_rows
        )
        task_total_speedups = [
            float(row["total_search_speedup"])
            for row in task_rows
            if row["total_search_speedup"] is not None
        ]
        task_regrets = [
            float(row["guided_selection_time_regret_ratio"])
            for row in task_rows
            if row["guided_selection_time_regret_ratio"] is not None
        ]
        return {
            "paired_record_count": len(task_rows),
            "full_candidate_attempt_count": sum(
                int(row["full_candidate_attempt_count"]) for row in task_rows
            ),
            "peakaware_candidate_attempt_count": sum(
                int(row["peakaware_candidate_attempt_count"]) for row in task_rows
            ),
            "full_measurement_candidate_count": sum(
                int(row["full_measurement_candidate_count"]) for row in task_rows
            ),
            "peakaware_measured_candidate_count": sum(
                int(row["peakaware_measured_candidate_count"]) for row in task_rows
            ),
            "full_measurement_gpu_hours": task_full_validation_hours,
            "peakaware_measurement_gpu_hours": task_guided_validation_hours,
            "candidate_validation_speedup": _safe_ratio(
                task_full_validation_hours,
                task_guided_validation_hours,
            ),
            "full_total_search_gpu_hours": task_full_total_hours,
            "peakaware_total_search_gpu_hours": task_guided_total_hours,
            "total_search_speedup": _safe_ratio(
                task_full_total_hours,
                task_guided_total_hours,
            ),
            "all_strategy_actual_run_speedup": _safe_ratio(
                task_full_total_hours,
                task_guided_total_hours,
            ),
            "paired_total_search_speedup_p50": _percentile(task_total_speedups, 0.50),
            "paired_total_search_speedup_min": (
                None if not task_total_speedups else min(task_total_speedups)
            ),
            "paired_total_search_speedup_max": (
                None if not task_total_speedups else max(task_total_speedups)
            ),
            "guided_selection_mean_time_regret_ratio": _mean(task_regrets),
            "guided_selected_oracle_rate": _mean(
                [
                    1.0 if row["guided_selected_is_oracle"] else 0.0
                    for row in task_rows
                    if row["guided_selected_is_oracle"] is not None
                ]
            ),
        }
    return {
        "evidence_mode": "paired_actual_exhaustive_and_guided_runs",
        "guided_selection_policy": guided_selection_policy,
        "exhaustive_input_record_count": len(exhaustive_records),
        "guided_input_record_count": len(guided_records),
        "exhaustive_failed_record_count": sum(1 for record in exhaustive_records if record.get("status") != "ok"),
        "guided_failed_record_count": sum(1 for record in guided_records if record.get("status") != "ok"),
        "exhaustive_unmatched_ok_record_count": len(exhaustive_keys - guided_keys),
        "guided_unmatched_ok_record_count": len(guided_keys - exhaustive_keys),
        "paired_record_count": len(rows),
        "cost_observation_count": len(cost_rows),
        "full_measurement_candidate_count": full_count,
        "peakaware_measured_candidate_count": guided_count,
        "full_candidate_attempt_count": sum(
            int(row["full_candidate_attempt_count"]) for row in rows
        ),
        "peakaware_candidate_attempt_count": sum(
            int(row["peakaware_candidate_attempt_count"]) for row in rows
        ),
        "candidate_measurements_saved": full_count - guided_count,
        "candidate_measurement_reduction_rate": _safe_ratio(float(full_count - guided_count), float(full_count)),
        "full_measurement_gpu_hours": full_hours,
        "peakaware_measurement_gpu_hours": guided_hours,
        "gpu_hours_saved": full_hours - guided_hours,
        "gpu_hours_reduction_rate": _safe_ratio(full_hours - guided_hours, full_hours),
        "candidate_validation_speedup": _safe_ratio(full_hours, guided_hours),
        "total_search_cost_observation_count": len(total_cost_rows),
        "full_total_search_gpu_hours": full_total_hours,
        "peakaware_total_search_gpu_hours": guided_total_hours,
        "total_search_gpu_hours_saved": full_total_hours - guided_total_hours,
        "total_search_gpu_hours_reduction_rate": _safe_ratio(
            full_total_hours - guided_total_hours,
            full_total_hours,
        ),
        "total_search_speedup": _safe_ratio(full_total_hours, guided_total_hours),
        "all_strategy_actual_run_speedup": _safe_ratio(
            full_total_hours,
            guided_total_hours,
        ),
        "paired_candidate_validation_speedup_p50": _percentile(
            paired_validation_speedups,
            0.50,
        ),
        "paired_total_search_speedup_p10": _percentile(paired_total_speedups, 0.10),
        "paired_total_search_speedup_p50": _percentile(paired_total_speedups, 0.50),
        "paired_total_search_speedup_p90": _percentile(paired_total_speedups, 0.90),
        "paired_total_search_speedup_min": (
            None if not paired_total_speedups else min(paired_total_speedups)
        ),
        "paired_total_search_speedup_max": (
            None if not paired_total_speedups else max(paired_total_speedups)
        ),
        "guided_selection_reference_count": len(guided_time_regrets),
        "guided_selected_oracle_rate": _mean(
            [1.0 if value else 0.0 for value in guided_oracle_flags]
        ),
        "guided_selection_mean_time_regret_ratio": _mean(guided_time_regrets),
        "guided_selection_p50_time_regret_ratio": _percentile(guided_time_regrets, 0.50),
        "guided_selection_p90_time_regret_ratio": _percentile(guided_time_regrets, 0.90),
        "guided_selection_max_time_regret_ratio": None
        if not guided_time_regrets
        else max(guided_time_regrets),
        "guided_selection_mean_peak_regret_ratio": _mean(guided_peak_regrets),
        "guided_selection_p90_peak_regret_ratio": _percentile(guided_peak_regrets, 0.90),
        "guided_selection_max_peak_regret_ratio": None
        if not guided_peak_regrets
        else max(guided_peak_regrets),
        "ranked_topk_membership_match_rate": _mean(
            [
                1.0 if row["ranked_topk_membership_match"] else 0.0
                for row in rows
                if row["ranked_topk_membership_match"] is not None
            ]
        ),
        "topk_membership_match_rate": _mean(
            [
                1.0 if row["ranked_topk_membership_match"] else 0.0
                for row in rows
                if row["ranked_topk_membership_match"] is not None
            ]
        ),
        "guided_candidate_coverage_rate": _mean(
            [
                1.0 if row["guided_candidates_covered_by_exhaustive"] else 0.0
                for row in rows
            ]
        ),
        "guided_shortlist_oracle_coverage_rate": _mean(
            [
                1.0 if row["guided_shortlist_oracle_covered"] else 0.0
                for row in rows
                if row["guided_shortlist_oracle_covered"] is not None
            ]
        ),
        "guided_shortlist_replay_mean_time_regret_ratio": _mean(
            shortlist_replay_regrets
        ),
        "guided_shortlist_replay_p90_time_regret_ratio": _percentile(
            shortlist_replay_regrets,
            0.90,
        ),
        "guided_shortlist_replay_max_time_regret_ratio": (
            None if not shortlist_replay_regrets else max(shortlist_replay_regrets)
        ),
        "by_task": {
            task_name: summarize_actual_rows(task_rows)
            for task_name, task_rows in sorted(by_task_rows.items())
        },
        "grouped_final_selection": _grouped_final_selection(
            exhaustive_records,
            guided_records,
        ),
        "grouped_shortlist_replay": _grouped_shortlist_replay(
            exhaustive_records,
            guided_records,
        ),
        "rows": rows,
    }
