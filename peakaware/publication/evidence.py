from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from peakaware.experiments import ExperimentRecord, experiment_records_from_dicts
from peakaware.publication.artifact import verify_publication_artifact


EVIDENCE_GATE_SCHEMA_VERSION = "0.1"
REQUIRED_BUDGET_RATIOS = (0.5, 0.65, 0.8, 0.95, 1.0)
REQUIRED_WORKLOAD_COUNT = 4
REQUIRED_REPEAT_COUNT = 5


@dataclass(frozen=True)
class EvidenceGateResult:
    gate_id: str
    name: str
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def evaluate_publication_evidence_gates(
    artifact_root: str | Path,
    *,
    records_json: str | Path | None = None,
    summary_json: str | Path | None = None,
    workload_manifest: str | Path | None = None,
    budget_manifest: str | Path | None = None,
    qualification_summary: str | Path | None = None,
    require_frozen: bool = False,
    required_budget_ratios: Sequence[float] = REQUIRED_BUDGET_RATIOS,
    required_repeat_count: int = REQUIRED_REPEAT_COUNT,
    required_workload_count: int = REQUIRED_WORKLOAD_COUNT,
) -> dict[str, Any]:
    """Evaluate thesis evidence gates for a publication artifact.

    The result is an audit, not a scientific claim.  A draft artifact may be
    structurally valid while still failing the frozen thesis gates.
    """

    root = Path(artifact_root)
    records_path = Path(records_json) if records_json is not None else _first_existing(
        root / "records.json",
        root / "raw" / "records.json",
    )
    summary_path = Path(summary_json) if summary_json is not None else _first_existing(
        root / "summary.json",
        root / "derived" / "summary.json",
    )
    workload_path = Path(workload_manifest) if workload_manifest is not None else _first_existing(
        root / "configs" / "workloads.json",
        root / "workloads.json",
    )
    budget_path = Path(budget_manifest) if budget_manifest is not None else _first_existing(
        root / "configs" / "budgets.json",
        root / "budgets.json",
    )
    qualification_path = Path(qualification_summary) if qualification_summary is not None else _first_existing(
        root / "qualification.records.jsonl.summary.json",
        root / "qualification_summary.json",
        root / "derived" / "qualification_summary.json",
    )

    records, record_errors = _load_records(records_path)
    qualification_payload, qualification_errors = _load_qualification_summary(qualification_path)
    artifact_payload = verify_publication_artifact(
        root,
        records_json=records_path,
        summary_json=summary_path,
        require_frozen=require_frozen,
    )
    gates = [
        _gate_g1_baseline_identity(
            records,
            qualification_payload,
            required_workload_count=required_workload_count,
        ),
        _gate_g2_workloads(records, workload_path, required_workload_count=required_workload_count),
        _gate_g3_runtime_identity(records),
        _gate_g4_inductor(records, required_workload_count=required_workload_count),
        _gate_g5_budget_coverage(
            records,
            budget_path,
            required_budget_ratios=tuple(required_budget_ratios),
            required_workload_count=required_workload_count,
            required_repeat_count=required_repeat_count,
        ),
        _gate_g6_measurement(records, required_repeat_count=required_repeat_count),
        _gate_g7_evidence_freeze(root, require_frozen=require_frozen),
        _gate_g8_figures_tables(artifact_payload, require_frozen=require_frozen),
    ]
    if qualification_errors:
        gates.append(
            EvidenceGateResult(
                "G-qualification",
                "qualification summary load",
                False,
                tuple(qualification_errors),
                metadata={"qualification_summary": str(qualification_path) if qualification_path else None},
            )
        )
    if record_errors:
        gates.insert(
            0,
            EvidenceGateResult(
                "G-records",
                "records load",
                False,
                tuple(record_errors),
                metadata={"records_json": str(records_path) if records_path else None},
            ),
        )
    error_count = sum(len(gate.errors) for gate in gates)
    warning_count = sum(len(gate.warnings) for gate in gates)
    passed_count = sum(1 for gate in gates if gate.passed)
    return {
        "schema_version": EVIDENCE_GATE_SCHEMA_VERSION,
        "artifact_root": str(root),
        "require_frozen": require_frozen,
        "ok": error_count == 0 and all(gate.passed for gate in gates),
        "record_count": len(records),
        "passed_gate_count": passed_count,
        "total_gate_count": len(gates),
        "error_count": error_count,
        "warning_count": warning_count,
        "gates": [_gate_payload(gate) for gate in gates],
        "artifact_check": {
            "ok": bool(artifact_payload["ok"]),
            "error_count": artifact_payload["error_count"],
            "warning_count": artifact_payload["warning_count"],
            "checks": [check["name"] for check in artifact_payload["checks"]],
        },
    }


def _gate_g1_baseline_identity(
    records: Sequence[ExperimentRecord],
    qualification_summary: dict[str, Any] | None,
    *,
    required_workload_count: int,
) -> EvidenceGateResult:
    plan_rows = [
        plan
        for record in records
        for plan in record.measured_plan_results
        if plan.get("plan_id")
    ]
    plan_ids = {str(plan.get("plan_id")) for plan in plan_rows}
    identity_counts = _baseline_identity_counts(plan_rows)
    qualification_scope = _qualification_scope(qualification_summary)
    qualification_methods = _qualified_methods(qualification_summary)
    gate_qualification_methods = (
        qualification_methods if qualification_scope["task_count"] >= required_workload_count else set()
    )
    has_min_cut = _has_real_plan(records, "min_cut")
    has_block_ac = _has_real_plan(records, "block")
    has_sac = _has_real_plan(records, "sac") or _has_real_plan(records, "selective")
    if "pytorch_min_cut" in gate_qualification_methods:
        has_min_cut = True
    if "block_ac" in gate_qualification_methods:
        has_block_ac = True
    if "sac" in gate_qualification_methods:
        has_sac = True
    errors = []
    if "all_save" not in plan_ids:
        errors.append("missing measured all_save baseline")
    if not has_min_cut:
        errors.append("missing real PyTorch AOT min-cut baseline provenance/runtime marker")
    if not has_block_ac:
        errors.append("missing real block activation-checkpoint baseline provenance/runtime marker")
    if not has_sac:
        errors.append("missing real selective activation-checkpoint baseline provenance/runtime marker")
    return EvidenceGateResult(
        "G-1",
        "baseline identity",
        not errors,
        tuple(errors),
        metadata={
            "measured_plan_ids": sorted(plan_ids),
            "identity_counts": identity_counts,
            "qualified_methods": sorted(qualification_methods),
            "gate_qualified_methods": sorted(gate_qualification_methods),
            "qualification_passed": None
            if qualification_summary is None
            else qualification_summary.get("qualification_passed"),
            "qualification_scope": qualification_scope,
            "required_workload_count": required_workload_count,
        },
    )


def _gate_g2_workloads(
    records: Sequence[ExperimentRecord],
    workload_path: Path | None,
    *,
    required_workload_count: int,
) -> EvidenceGateResult:
    tasks = sorted({record.task_name for record in records})
    errors = []
    warnings = []
    manifest_workloads = []
    if len(tasks) < required_workload_count:
        errors.append(f"expected at least {required_workload_count} workloads, found {len(tasks)}")
    if workload_path is None or not workload_path.is_file():
        errors.append("missing workload manifest")
    else:
        try:
            payload = json.loads(workload_path.read_text(encoding="utf-8"))
            manifest_workloads = payload.get("workloads", [])
            if not isinstance(manifest_workloads, list):
                raise ValueError("workloads must be a list")
        except Exception as exc:
            errors.append(f"invalid workload manifest: {exc}")
        else:
            display_names = [
                str(item.get("workload", {}).get("display_name", ""))
                for item in manifest_workloads
                if isinstance(item, dict)
            ]
            if len(display_names) < required_workload_count:
                errors.append("workload manifest does not cover all required workloads")
            if any(name in {"BERT-Base", "GPT-2"} for name in display_names):
                warnings.append("standard transformer names require matching standard specs")
    return EvidenceGateResult(
        "G-2",
        "workload manifest",
        not errors,
        tuple(errors),
        tuple(warnings),
        metadata={
            "tasks": tasks,
            "workload_manifest": str(workload_path) if workload_path else None,
            "manifest_workload_count": len(manifest_workloads),
        },
    )


def _gate_g3_runtime_identity(records: Sequence[ExperimentRecord]) -> EvidenceGateResult:
    ok_records = [record for record in records if record.status == "ok"]
    missing = [
        _record_key(record)
        for record in ok_records
        if record.selected_aot_partition_runtime is not True
    ]
    errors = []
    if not ok_records:
        errors.append("no ok rows with runtime identity")
    if missing:
        errors.append(f"{len(missing)} ok rows lack selected lowered-AOT runtime identity")
    return EvidenceGateResult(
        "G-3",
        "runtime identity",
        not errors,
        tuple(errors),
        metadata={"ok_records": len(ok_records), "missing_examples": missing[:5]},
    )


def _gate_g4_inductor(
    records: Sequence[ExperimentRecord],
    *,
    required_workload_count: int,
) -> EvidenceGateResult:
    inductor_tasks = sorted(
        {
            record.task_name
            for record in records
            if record.status == "ok"
            and str(record.config_fingerprint.get("compile_backend")) == "inductor"
        }
    )
    errors = []
    if len(inductor_tasks) < required_workload_count:
        errors.append(
            f"expected Inductor ok rows for {required_workload_count} workloads, found {len(inductor_tasks)}"
        )
    return EvidenceGateResult(
        "G-4",
        "Inductor matrix",
        not errors,
        tuple(errors),
        metadata={"inductor_tasks": inductor_tasks},
    )


def _gate_g5_budget_coverage(
    records: Sequence[ExperimentRecord],
    budget_path: Path | None,
    *,
    required_budget_ratios: tuple[float, ...],
    required_workload_count: int,
    required_repeat_count: int,
) -> EvidenceGateResult:
    errors = []
    cells = []
    if budget_path is None or not budget_path.is_file():
        errors.append("missing budget manifest with all-save-relative ratios")
    else:
        try:
            payload = json.loads(budget_path.read_text(encoding="utf-8"))
            cells = payload.get("cells", [])
            if not isinstance(cells, list):
                raise ValueError("cells must be a list")
        except Exception as exc:
            errors.append(f"invalid budget manifest: {exc}")
        else:
            if len(cells) < required_workload_count:
                errors.append(f"expected budget cells for {required_workload_count} workloads")
            for cell in cells:
                ratios = tuple(float(value) for value in cell.get("ratios", ()))
                missing = [ratio for ratio in required_budget_ratios if not _contains_ratio(ratios, ratio)]
                if missing:
                    errors.append(f"{cell.get('task_name', '<unknown>')} missing budget ratios {missing}")
                if cell.get("reference_count", 0) < required_repeat_count:
                    errors.append(
                        f"{cell.get('task_name', '<unknown>')} all-save reference count is below frozen repeat target"
                    )
    observed_budgets = sorted({record.budget_bytes for record in records})
    return EvidenceGateResult(
        "G-5",
        "budget coverage",
        not errors,
        tuple(errors),
        metadata={
            "budget_manifest": str(budget_path) if budget_path else None,
            "cell_count": len(cells),
            "observed_budget_count": len(observed_budgets),
            "required_ratios": list(required_budget_ratios),
        },
    )


def _gate_g6_measurement(
    records: Sequence[ExperimentRecord],
    *,
    required_repeat_count: int,
) -> EvidenceGateResult:
    ok_records = [record for record in records if record.status == "ok"]
    low_repeat = [
        _record_key(record)
        for record in ok_records
        if (record.measurement_repeats or 0) < required_repeat_count
    ]
    pass_counts: dict[str, set[int]] = {}
    for record in records:
        pass_counts.setdefault(record.task_name, set()).add(record.matrix_pass_index)
    errors = []
    if low_repeat:
        errors.append(f"{len(low_repeat)} ok rows have fewer than {required_repeat_count} measurement repeats")
    if any(len(indices) < required_repeat_count for indices in pass_counts.values()):
        errors.append("fewer than 5 independent process attempts per workload")
    return EvidenceGateResult(
        "G-6",
        "measurement protocol",
        not errors,
        tuple(errors),
        metadata={
            "ok_records": len(ok_records),
            "low_repeat_examples": low_repeat[:5],
            "attempts_per_task": {task: len(indices) for task, indices in sorted(pass_counts.items())},
        },
    )


def _gate_g7_evidence_freeze(root: Path, *, require_frozen: bool) -> EvidenceGateResult:
    manifest_path = root / "manifest.json"
    errors = []
    metadata: dict[str, Any] = {"manifest": str(manifest_path), "require_frozen": require_frozen}
    if not manifest_path.is_file():
        errors.append("missing artifact manifest")
    else:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid manifest: {exc}")
        else:
            metadata["evidence_status"] = payload.get("evidence_status")
            metadata["scope"] = payload.get("scope")
            metadata["git_dirty"] = (payload.get("git") or {}).get("dirty")
            if payload.get("evidence_status") != "frozen":
                errors.append("manifest evidence_status is not frozen")
            if (payload.get("git") or {}).get("dirty") is not False:
                errors.append("frozen evidence requires a clean git state")
    return EvidenceGateResult("G-7", "evidence freeze", not errors, tuple(errors), metadata=metadata)


def _gate_g8_figures_tables(
    artifact_payload: dict[str, Any],
    *,
    require_frozen: bool,
) -> EvidenceGateResult:
    errors = []
    if not artifact_payload.get("ok"):
        errors.append("publication artifact structural validation failed")
    checks = {
        str(check.get("name")): check
        for check in artifact_payload.get("checks", ())
        if isinstance(check, dict)
    }
    check_names = set(checks)
    for required in ("figures", "tables"):
        if required not in check_names:
            errors.append(f"missing {required} validation check")
        elif not checks[required].get("ok"):
            errors.append(f"{required} validation failed")
    if require_frozen:
        for required in ("figures", "tables"):
            check = checks.get(required)
            if check and check.get("ok") and check.get("metadata", {}).get(f"{required[:-1]}_warning_count"):
                errors.append(f"{required} frozen validation produced warnings")
    return EvidenceGateResult(
        "G-8",
        "figure and table generation",
        not errors,
        tuple(errors),
        metadata={"artifact_ok": artifact_payload.get("ok"), "checks": sorted(check_names)},
    )


def _has_real_plan(records: Sequence[ExperimentRecord], token: str) -> bool:
    for record in records:
        for plan in record.measured_plan_results:
            plan_id = str(plan.get("plan_id", "")).lower()
            if token not in plan_id:
                continue
            if _plan_identity_kind(plan) == "real":
                return True
    return False


def _baseline_identity_counts(plans: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"real": 0, "proxy": 0, "unknown": 0}
    for plan in plans:
        kind = _plan_identity_kind(plan)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _plan_identity_kind(plan: dict[str, Any]) -> str:
    plan_id = str(plan.get("plan_id", "")).lower()
    provenance = plan.get("strategy_provenance")
    if not isinstance(provenance, dict):
        provenance = plan.get("provenance", plan.get("baseline_provenance", {}))
    provenance_text = json.dumps(provenance, sort_keys=True, default=str).lower()
    source = ""
    if isinstance(provenance, dict):
        source = str(provenance.get("source", "")).lower()
    runtime = plan.get("runtime_identity") or plan.get("runtime_marker")
    explicit_real = (
        plan.get("is_real") is True
        or plan.get("baseline_is_real") is True
        or (isinstance(runtime, dict) and runtime.get("is_real") is True)
        or "real" in provenance_text
    )
    legacy_proxy_id = plan_id in {"torch_min_cut", "block_checkpoint"}
    explicit_proxy = (
        plan.get("is_real") is False
        or plan.get("baseline_is_real") is False
        or "proxy" in plan_id
        or "proxy" in source
        or "proxy" in provenance_text
        or (legacy_proxy_id and not explicit_real)
    )
    if explicit_real and not explicit_proxy:
        return "real"
    if explicit_proxy:
        return "proxy"
    return "unknown"


def _load_records(records_path: Path | None) -> tuple[tuple[ExperimentRecord, ...], list[str]]:
    if records_path is None or not records_path.is_file():
        return (), ["missing records.json"]
    try:
        raw_payload = json.loads(records_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, list):
            raise ValueError("records JSON must contain a list")
        return experiment_records_from_dicts(raw_payload), []
    except Exception as exc:
        return (), [f"invalid records.json: {exc}"]


def _load_qualification_summary(summary_path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if summary_path is None:
        return None, []
    if not summary_path.is_file():
        return None, [f"missing qualification summary: {summary_path}"]
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("qualification summary must contain an object")
        return payload, []
    except Exception as exc:
        return None, [f"invalid qualification summary: {exc}"]


def _qualified_methods(summary: dict[str, Any] | None) -> set[str]:
    if not summary:
        return set()
    method_qualification = summary.get("method_qualification")
    if not isinstance(method_qualification, dict):
        return set()
    qualified: set[str] = set()
    for methods in method_qualification.values():
        if not isinstance(methods, dict):
            continue
        for method, counts in methods.items():
            if not isinstance(counts, dict):
                continue
            if int(counts.get("qualified", 0)) > 0:
                qualified.add(str(method))
    return qualified


def _qualification_scope(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"task_count": 0, "tasks": []}
    tasks: list[str] = []
    matrix = summary.get("matrix")
    if isinstance(matrix, dict) and isinstance(matrix.get("tasks"), list):
        tasks = [str(task) for task in matrix["tasks"]]
    return {"task_count": len(set(tasks)), "tasks": sorted(set(tasks))}


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _contains_ratio(ratios: Sequence[float], expected: float) -> bool:
    return any(abs(ratio - expected) < 1e-6 for ratio in ratios)


def _record_key(record: ExperimentRecord) -> str:
    return (
        f"{record.task_name}/{record.variant_name}/budget={record.budget_bytes}/"
        f"pass={record.matrix_pass_index}"
    )


def _gate_payload(gate: EvidenceGateResult) -> dict[str, Any]:
    return {
        "gate_id": gate.gate_id,
        "name": gate.name,
        "passed": gate.passed,
        "errors": list(gate.errors),
        "warnings": list(gate.warnings),
        "metadata": gate.metadata or {},
    }
