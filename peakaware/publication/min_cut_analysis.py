from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OFFICIAL_MIN_CUT_API = "torch._functorch.partitioners.min_cut_rematerialization_partition"
OFFICIAL_MIN_CUT_METHOD = "pytorch_min_cut"
OFFICIAL_MIN_CUT_METHOD_ID = "pytorch_aot_min_cut"
OFFICIAL_MIN_CUT_SOLVER = "PyTorch min-cut rematerialization partitioner"


def load_jsonl_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_path in paths:
        path = Path(input_path)
        if path.is_dir():
            path = path / "records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                payload["_source_path"] = str(path)
                payload["_source_line"] = line_number
                records.append(payload)
    return records


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    )


def _activation_ratio(record: Mapping[str, Any]) -> float | None:
    method_config = record.get("method_config")
    if not isinstance(method_config, Mapping):
        return None
    value = method_config.get("activation_memory_budget")
    if not _finite_number(value):
        return None
    ratio = float(value)
    return ratio if 0.0 <= ratio <= 1.0 else None


def _identity_audit(record: Mapping[str, Any], ratio: float | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    identity = record.get("runtime_identity")
    if record.get("method") != OFFICIAL_MIN_CUT_METHOD:
        reasons.append("method_is_not_pytorch_min_cut")
    if not isinstance(identity, Mapping):
        return False, [*reasons, "runtime_identity_missing"]
    if identity.get("method_id") != OFFICIAL_MIN_CUT_METHOD_ID:
        reasons.append("runtime_method_id_mismatch")
    if identity.get("api") != OFFICIAL_MIN_CUT_API:
        reasons.append("runtime_api_mismatch")
    if identity.get("is_real") is not True:
        reasons.append("runtime_not_marked_real")
    provenance = identity.get("provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("runtime_provenance_missing")
    else:
        if provenance.get("solver") != OFFICIAL_MIN_CUT_SOLVER:
            reasons.append("runtime_solver_mismatch")
        if provenance.get("partitioner_cost_model") != "inductor":
            reasons.append("partitioner_cost_model_is_not_inductor")
        provenance_ratio = provenance.get("memory_budget_ratio")
        if ratio is None or not _finite_number(provenance_ratio) or float(provenance_ratio) != ratio:
            reasons.append("runtime_ratio_mismatch")
    return not reasons, reasons


def _qualified(record: Mapping[str, Any], identity_verified: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not identity_verified:
        reasons.append("official_runtime_identity_not_verified")
    if record.get("status") != "ok":
        reasons.append(f"status={record.get('status')}")
    correctness = record.get("correctness_report")
    if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
        reasons.append("correctness_not_passed")
    if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
        reasons.append("seed_missing")
    if (
        not isinstance(record.get("microbatch_size"), int)
        or isinstance(record.get("microbatch_size"), bool)
        or int(record["microbatch_size"]) <= 0
    ):
        reasons.append("microbatch_size_invalid")
    environment = record.get("environment")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("gpu_uuid"), str):
        reasons.append("gpu_uuid_missing")
    if not isinstance(record.get("execution_fingerprint"), str):
        reasons.append("execution_fingerprint_missing")
    identity = record.get("runtime_identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("model_sha256"), str):
        reasons.append("model_sha256_missing")
    measurement = record.get("measurement_aggregate")
    if not isinstance(measurement, Mapping):
        reasons.append("measurement_missing")
    else:
        if measurement.get("publication_qualified") is not True:
            reasons.append("publication_measurement_not_qualified")
        if not _finite_number(measurement.get("overall_peak_bytes")) or float(
            measurement.get("overall_peak_bytes", 0)
        ) <= 0:
            reasons.append("overall_peak_missing")
        if not _finite_number(measurement.get("overall_event_us")) or float(
            measurement.get("overall_event_us", 0)
        ) <= 0:
            reasons.append("overall_event_time_missing")
        if not _finite_number(measurement.get("overall_wall_us")) or float(
            measurement.get("overall_wall_us", 0)
        ) <= 0:
            reasons.append("overall_wall_time_missing")
        repeats = measurement.get("measurement_repeats")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
            reasons.append("measurement_repeats_invalid")
        warmup_steps = measurement.get("measurement_warmup_steps")
        if (
            not isinstance(warmup_steps, int)
            or isinstance(warmup_steps, bool)
            or warmup_steps < 0
        ):
            reasons.append("measurement_warmup_steps_invalid")
    return not reasons, reasons


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def _point_is_dominated(point: Mapping[str, Any], points: Sequence[Mapping[str, Any]]) -> bool:
    peak = float(point["overall_peak_bytes"])
    step = float(point["overall_event_us"])
    for other in points:
        if other is point:
            continue
        other_peak = float(other["overall_peak_bytes"])
        other_step = float(other["overall_event_us"])
        if other_peak <= peak and other_step <= step and (other_peak < peak or other_step < step):
            return True
    return False


def analyze_official_min_cut(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_ratio: float = 1.0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("no records were provided")
    if not 0.0 <= baseline_ratio <= 1.0:
        raise ValueError("baseline_ratio must be in [0, 1]")

    audited: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        ratio = _activation_ratio(record)
        identity_verified, identity_reasons = _identity_audit(record, ratio)
        qualified, qualification_reasons = _qualified(record, identity_verified)
        task_name = record.get("task_name")
        row = {
            "task_name": task_name,
            "display_name": record.get("display_name") or task_name,
            "activation_memory_budget": ratio,
            "identity_verified": identity_verified,
            "identity_failure_reasons": identity_reasons,
            "qualified": qualified,
            "qualification_failure_reasons": qualification_reasons,
            "status": record.get("status"),
            "source_path": record.get("_source_path"),
            "source_line": record.get("_source_line"),
            "workload_fingerprint": record.get("workload_fingerprint"),
            "backend": record.get("backend"),
            "microbatch_size": record.get("microbatch_size"),
            "gpu_name": (record.get("environment") or {}).get("gpu_name")
            if isinstance(record.get("environment"), Mapping)
            else None,
            "gpu_uuid": (record.get("environment") or {}).get("gpu_uuid")
            if isinstance(record.get("environment"), Mapping)
            else None,
            "torch_version": (record.get("environment") or {}).get("torch_version")
            if isinstance(record.get("environment"), Mapping)
            else None,
            "seed": record.get("seed"),
            "execution_fingerprint": record.get("execution_fingerprint"),
            "model_sha256": (
                (record.get("runtime_identity") or {}).get("model_sha256")
                if isinstance(record.get("runtime_identity"), Mapping)
                else None
            ),
        }
        audited.append(row)
        if qualified and isinstance(task_name, str) and ratio is not None:
            grouped[(task_name, ratio)].append(dict(record))

    rows: list[dict[str, Any]] = []
    for (task_name, ratio), group in sorted(grouped.items()):
        measurements = [record["measurement_aggregate"] for record in group]
        identities = [record["runtime_identity"] for record in group]
        row = {
            "task_name": task_name,
            "display_name": group[0].get("display_name") or task_name,
            "activation_memory_budget": ratio,
            "replicate_count": len(group),
            "overall_peak_bytes": _median(float(item["overall_peak_bytes"]) for item in measurements),
            "overall_event_us": _median(float(item["overall_event_us"]) for item in measurements),
            "overall_wall_us": _median(float(item["overall_wall_us"]) for item in measurements),
            "fw_residual_count": _median(len(identity.get("fw_residual_names") or ()) for identity in identities),
            "measurement_repeats": sorted({int(item["measurement_repeats"]) for item in measurements}),
            "measurement_warmup_steps": sorted(
                {int(item["measurement_warmup_steps"]) for item in measurements}
            ),
            "workload_fingerprints": sorted({str(record.get("workload_fingerprint")) for record in group}),
            "backends": sorted({str(record.get("backend")) for record in group}),
            "microbatch_sizes": sorted({int(record["microbatch_size"]) for record in group}),
            "seeds": sorted({int(record["seed"]) for record in group}),
            "gpu_uuids": sorted({str(record["environment"]["gpu_uuid"]) for record in group}),
            "execution_fingerprints": sorted(
                {str(record["execution_fingerprint"]) for record in group}
            ),
            "model_sha256s": sorted(
                {str(record["runtime_identity"]["model_sha256"]) for record in group}
            ),
            "replicate_bindings": sorted(
                (
                    int(record["seed"]),
                    str(record["environment"]["gpu_uuid"]),
                    str(record["execution_fingerprint"]),
                    str(record["runtime_identity"]["model_sha256"]),
                )
                for record in group
            ),
        }
        rows.append(row)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)

    task_summaries: list[dict[str, Any]] = []
    monotonicity_checks: list[dict[str, Any]] = []
    all_ratios = sorted(
        {
            float(row["activation_memory_budget"])
            for row in audited
            if row["activation_memory_budget"] is not None
        }
    )
    attempted_task_names = sorted(
        {
            str(row["task_name"])
            for row in audited
            if isinstance(row["task_name"], str) and row["activation_memory_budget"] is not None
        }
    )
    for task_name, task_rows in sorted(by_task.items()):
        task_rows.sort(key=lambda item: float(item["activation_memory_budget"]))
        baseline = next(
            (row for row in task_rows if float(row["activation_memory_budget"]) == baseline_ratio),
            None,
        )
        for row in task_rows:
            if baseline is None:
                row["peak_reduction_vs_baseline"] = None
                row["time_overhead_vs_baseline"] = None
            else:
                row["peak_reduction_vs_baseline"] = (
                    float(baseline["overall_peak_bytes"]) - float(row["overall_peak_bytes"])
                ) / float(baseline["overall_peak_bytes"])
                row["time_overhead_vs_baseline"] = (
                    float(row["overall_event_us"]) / float(baseline["overall_event_us"]) - 1.0
                )
        for row in task_rows:
            row["pareto_dominated"] = _point_is_dominated(row, task_rows)
        for previous, current in zip(task_rows, task_rows[1:]):
            check = {
                "task_name": task_name,
                "from_ratio": previous["activation_memory_budget"],
                "to_ratio": current["activation_memory_budget"],
                "residual_count_nondecreasing": (
                    float(current["fw_residual_count"]) >= float(previous["fw_residual_count"])
                ),
                "physical_peak_nondecreasing": (
                    float(current["overall_peak_bytes"]) >= float(previous["overall_peak_bytes"])
                ),
                "step_time_nonincreasing": (
                    float(current["overall_event_us"]) <= float(previous["overall_event_us"])
                ),
            }
            monotonicity_checks.append(check)
        workload_fingerprints = sorted(
            {fingerprint for row in task_rows for fingerprint in row["workload_fingerprints"]}
        )
        seeds = sorted({seed for row in task_rows for seed in row["seeds"]})
        gpu_uuids = sorted({uuid for row in task_rows for uuid in row["gpu_uuids"]})
        execution_fingerprints = sorted(
            {fingerprint for row in task_rows for fingerprint in row["execution_fingerprints"]}
        )
        model_sha256s = sorted({digest for row in task_rows for digest in row["model_sha256s"]})
        seed_sets_match = len({tuple(row["seeds"]) for row in task_rows}) == 1
        gpu_uuid_sets_match = len({tuple(row["gpu_uuids"]) for row in task_rows}) == 1
        execution_fingerprint_sets_match = (
            len({tuple(row["execution_fingerprints"]) for row in task_rows}) == 1
        )
        model_sha256_sets_match = len({tuple(row["model_sha256s"]) for row in task_rows}) == 1
        replicate_binding_sets_match = (
            len(
                {
                    tuple(tuple(binding) for binding in row["replicate_bindings"])
                    for row in task_rows
                }
            )
            == 1
        )
        task_summaries.append(
            {
                "task_name": task_name,
                "display_name": task_rows[0]["display_name"],
                "ratios": [row["activation_memory_budget"] for row in task_rows],
                "complete_ratio_coverage": [row["activation_memory_budget"] for row in task_rows]
                == all_ratios,
                "workload_consistent": len(workload_fingerprints) == 1,
                "workload_fingerprints": workload_fingerprints,
                "seed_consistent": seed_sets_match,
                "seeds": seeds,
                "gpu_uuid_consistent": gpu_uuid_sets_match,
                "gpu_uuids": gpu_uuids,
                "execution_protocol_consistent": execution_fingerprint_sets_match,
                "execution_fingerprints": execution_fingerprints,
                "model_initialization_consistent": model_sha256_sets_match,
                "model_sha256s": model_sha256s,
                "replicate_bindings_consistent": replicate_binding_sets_match,
                "pareto_ratios": [
                    row["activation_memory_budget"] for row in task_rows if not row["pareto_dominated"]
                ],
            }
        )

    qualified_count = sum(1 for row in audited if row["qualified"])
    identity_verified_count = sum(1 for row in audited if row["identity_verified"])
    paired_protocol_task_count = sum(
        1
        for item in task_summaries
        if item["complete_ratio_coverage"]
        and item["workload_consistent"]
        and item["replicate_bindings_consistent"]
    )
    minimum_replicates = min((row["replicate_count"] for row in rows), default=0)
    paired_protocol_ready = (
        paired_protocol_task_count == len(attempted_task_names) and bool(attempted_task_names)
    )
    comparison_blockers = []
    if not paired_protocol_ready:
        comparison_blockers.append(
            "ratio points are not fully paired on seed, GPU UUID, execution protocol, and model initialization"
        )
    if minimum_replicates < 5:
        comparison_blockers.append(
            "fewer than five independent replicates per task-ratio point; repeat-level samples are not independent replicates"
        )
    comparison_blockers.extend(
        [
            "no PeakAware record uses the same AOT-eager workload and publication measurement protocol",
            "activation_memory_budget controls saved activations, not the complete physical peak budget",
        ]
    )
    summary = {
        "attempted_record_count": len(audited),
        "identity_verified_count": identity_verified_count,
        "qualified_record_count": qualified_count,
        "identity_verified_rate": identity_verified_count / len(audited),
        "qualification_rate": qualified_count / len(audited),
        "task_count": len(attempted_task_names),
        "qualified_task_count": len(task_summaries),
        "ratios": all_ratios,
        "complete_task_count": sum(1 for item in task_summaries if item["complete_ratio_coverage"]),
        "workload_consistent_task_count": sum(1 for item in task_summaries if item["workload_consistent"]),
        "seed_consistent_task_count": sum(1 for item in task_summaries if item["seed_consistent"]),
        "gpu_uuid_consistent_task_count": sum(
            1 for item in task_summaries if item["gpu_uuid_consistent"]
        ),
        "execution_protocol_consistent_task_count": sum(
            1 for item in task_summaries if item["execution_protocol_consistent"]
        ),
        "model_initialization_consistent_task_count": sum(
            1 for item in task_summaries if item["model_initialization_consistent"]
        ),
        "replicate_binding_consistent_task_count": sum(
            1 for item in task_summaries if item["replicate_bindings_consistent"]
        ),
        "paired_protocol_task_count": paired_protocol_task_count,
        "paired_protocol_ready": paired_protocol_ready,
        "residual_monotonic_pass_count": sum(
            1 for item in monotonicity_checks if item["residual_count_nondecreasing"]
        ),
        "physical_peak_monotonic_pass_count": sum(
            1 for item in monotonicity_checks if item["physical_peak_nondecreasing"]
        ),
        "step_time_monotonic_pass_count": sum(
            1 for item in monotonicity_checks if item["step_time_nonincreasing"]
        ),
        "adjacent_transition_count": len(monotonicity_checks),
        "minimum_replicates_per_point": minimum_replicates,
        "comparison_readiness": (
            "official_baseline_paired_canary"
            if paired_protocol_ready and minimum_replicates < 5
            else "official_baseline_replicated"
            if paired_protocol_ready
            else "official_baseline_unpaired_canary"
        ),
        "comparison_blockers": comparison_blockers,
    }
    return {
        "schema_version": "official_pytorch_min_cut_analysis_v1",
        "baseline_ratio": baseline_ratio,
        "runtime_identity": {
            "method": OFFICIAL_MIN_CUT_METHOD,
            "method_id": OFFICIAL_MIN_CUT_METHOD_ID,
            "api": OFFICIAL_MIN_CUT_API,
            "solver": OFFICIAL_MIN_CUT_SOLVER,
            "partitioner_cost_model": "inductor",
            "is_real": True,
        },
        "summary": summary,
        "rows": rows,
        "task_summaries": task_summaries,
        "monotonicity_checks": monotonicity_checks,
        "record_audit": audited,
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _gib(value: float) -> str:
    return f"{value / (1024 ** 3):.3f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    baseline_ratio = float(payload["baseline_ratio"])
    minimum_replicates = int(summary["minimum_replicates_per_point"])
    paired_status = "通过" if summary["paired_protocol_ready"] else "未通过"
    lines = [
        "# 官方 PyTorch min-cut / Memory Budget 基线分析",
        "",
        "## 结论",
        "",
        (
            f"三档预算共 {summary['attempted_record_count']} 条记录，其中 "
            f"{summary['identity_verified_count']}/{summary['attempted_record_count']} 通过官方运行时身份审计，"
            f"{summary['qualified_record_count']}/{summary['attempted_record_count']} 通过正确性与 publication 测量资格。"
        ),
        "",
        (
            f"配对协议审计{paired_status}：{summary['paired_protocol_task_count']}/{summary['task_count']} 个模型"
            "在三档间保持同 seed、同 GPU UUID、同 execution fingerprint 和同模型初始化摘要。"
        ),
        "",
        (
            "这些记录真实调用 `torch._functorch.partitioners.min_cut_rematerialization_partition`，"
            "并使用 `activation_memory_budget` 生成不同 AOT 前后向分区；它们不是 PeakAware 搜索空间中的 "
            "`torch_min_cut` proxy。"
        ),
        "",
        f"当前结果只能作为正式外部 planner 的可运行性与机制 canary。每个模型—ratio 点最少只有 {minimum_replicates} 次独立进程运行，"
        "且尚无同协议 PeakAware 曲线，因此不能据此宣称 PeakAware 整体优于官方 min-cut。",
        "",
        "## 实测 Pareto 数据",
        "",
        "主时间指标为独立 full-step 窗口的 CUDA event 中位数；峰值为 20 次 full-step 窗口的最大 allocated peak。",
        "",
        f"| 模型 | activation ratio | 完整峰值 (GiB) | Full-step (ms) | FW residual 数 | 峰值降低 vs ratio={baseline_ratio:g} | 时间开销 vs ratio={baseline_ratio:g} | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['display_name']} | {row['activation_memory_budget']:g} | "
            f"{_gib(row['overall_peak_bytes'])} | {row['overall_event_us'] / 1000.0:.3f} | "
            f"{row['fw_residual_count']:.0f} | {_percent(row['peak_reduction_vs_baseline'])} | "
            f"{_percent(row['time_overhead_vs_baseline'])} | "
            f"{'否（被支配）' if row['pareto_dominated'] else '是'} |"
        )
    transitions = summary["adjacent_transition_count"]
    lines.extend(
        [
            "",
            "## 单调性审计",
            "",
            (
                f"保存 residual 数随 ratio 增加的单调性为 {summary['residual_monotonic_pass_count']}/{transitions}；"
                f"完整物理峰值为 {summary['physical_peak_monotonic_pass_count']}/{transitions}；"
                f"执行时间为 {summary['step_time_monotonic_pass_count']}/{transitions}。"
            ),
            "",
            "| 模型 | ratio 转换 | residual 非减 | 完整峰值非减 | 时间非增 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for check in payload["monotonicity_checks"]:
        lines.append(
            f"| {check['task_name']} | {check['from_ratio']:g}→{check['to_ratio']:g} | "
            f"{'✓' if check['residual_count_nondecreasing'] else '✗'} | "
            f"{'✓' if check['physical_peak_nondecreasing'] else '✗'} | "
            f"{'✓' if check['step_time_nonincreasing'] else '✗'} |"
        )
    lines.extend(
        [
            "",
            "`activation_memory_budget` 对保存激活数量的控制是有效的，但它不是完整峰值预算。当前观察到的"
            "完整峰值/时间非单调点可能来自重计算临时量、allocator、workspace 或独立运行波动；在完成 5 个"
            "独立 replicate 前不能只归因于其中一种机制。这也是 PeakAware 需要完整生命周期仿真和重复验证的理由。",
            "",
            "## 对 PeakAware 寻优设计的启示",
            "",
            "1. 保留图切分思想：把保存/重算决策表述为带计算代价的 cut，而不是只枚举固定模板。",
            "2. 替换约束：cut 的容量不能只表示保存激活字节，应接入 PeakAware 的存储对象生命周期、融合组 workspace、梯度和优化器状态峰值。",
            "3. 双层求解：min-cut 或拉格朗日扫描负责快速产生候选，完整轨迹仿真器负责预算可行性判定与时间排序。",
            "4. 最终验证：搜索阶段不实跑候选，只实跑最终策略；exhaustive 全策略执行仅作为离线 oracle 上界。",
            "",
            "## 公平比较还缺什么",
            "",
            "- 对 ratio=0/0.5/1 至少运行 5 个独立 replicate，并随机化 ratio 执行顺序；20 个 repeat 只能降低单次运行测量噪声，不能替代独立重复。",
            "- 在完全相同的模型、microbatch、AOT-eager backend、FP32、优化器、warmup 和 20-repeat 协议下运行 PeakAware 最终计划。",
            "- 对两种 planner 都报告完整峰值—时间 Pareto、预算命中率和相对 exhaustive oracle regret。",
            "- 正文中将官方 min-cut、`torch_min_cut` proxy、Checkmate-style estimator 三者分开命名。",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "official_min_cut_analysis.json"
    csv_path = output_dir / "official_min_cut_pareto.csv"
    report_path = output_dir / "OFFICIAL_MIN_CUT_BASELINE_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "task_name",
        "display_name",
        "activation_memory_budget",
        "replicate_count",
        "overall_peak_bytes",
        "overall_event_us",
        "overall_wall_us",
        "fw_residual_count",
        "peak_reduction_vs_baseline",
        "time_overhead_vs_baseline",
        "pareto_dominated",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "report": report_path}
