from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from peakaware.experiments import ExperimentRecord
from peakaware.workload_manifest import canonical_json


CALIBRATION_SCHEMA_VERSION = "1.0"
BUDGET_PLAN_SCHEMA_VERSION = "0.1"
Backend = Literal["aot_eager", "inductor"]
Cohort = Literal["tuning", "evaluation"]
MeasurementStatus = Literal["ok", "budget_violation", "oom"]
SelectionStatus = Literal["selected", "infeasible"]
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


class LegacyBudgetConversionError(ValueError):
    """Raised when an activation ratio is inferred from a physical budget."""


def _nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _fingerprint(value: str, name: str) -> None:
    if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character hex fingerprint")


def _backend(value: str) -> None:
    if value not in ("aot_eager", "inductor"):
        raise ValueError("backend must be 'aot_eager' or 'inductor'")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a non-negative integer")


def _ratio(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ratio must be a number in [0, 1]")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("ratio must be in [0, 1]")


def _normalized_ratio(value: float) -> float:
    _ratio(value)
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def _identity(source_key: str, backend: str, environment: str) -> tuple[str, str, str]:
    _fingerprint(source_key, "source_key")
    _backend(backend)
    _fingerprint(environment, "environment")
    return source_key, backend, environment


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AllSaveReference:
    """Physical all-save calibration under one source/backend/environment.

    Each replicate is an independent process with a unique seed and process ID.
    The replicate statistic is the maximum overall peak across its repeats, and
    P_ref is the median of at least five replicate maxima.
    """

    source_key: str
    backend: Backend
    environment: str
    replicate_seeds: tuple[int, ...]
    replicate_process_ids: tuple[int, ...]
    per_replicate_peak_bytes: tuple[tuple[int, ...], ...]
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "replicate_seeds", tuple(self.replicate_seeds))
        object.__setattr__(self, "replicate_process_ids", tuple(self.replicate_process_ids))
        object.__setattr__(
            self,
            "per_replicate_peak_bytes",
            tuple(tuple(peaks) for peaks in self.per_replicate_peak_bytes),
        )
        _identity(self.source_key, self.backend, self.environment)
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema: {self.schema_version}")
        if len(self.replicate_seeds) < 5:
            raise ValueError("all-save reference requires at least 5 independent replicates")
        if len(set(self.replicate_seeds)) != len(self.replicate_seeds):
            raise ValueError("all-save replicate seeds must be unique")
        if len(set(self.replicate_process_ids)) != len(self.replicate_process_ids):
            raise ValueError("all-save replicate process IDs must be unique")
        lengths = {
            len(self.replicate_seeds),
            len(self.replicate_process_ids),
            len(self.per_replicate_peak_bytes),
        }
        if len(lengths) != 1:
            raise ValueError("all-save seeds, process IDs, and peak groups must have equal length")
        for seed in self.replicate_seeds:
            _seed(seed)
        for process_id in self.replicate_process_ids:
            _positive_int(process_id, "process_id")
        for peaks in self.per_replicate_peak_bytes:
            if not peaks:
                raise ValueError(
                    "each all-save replicate requires at least one overall peak repeat"
                )
            for peak in peaks:
                _positive_int(peak, "all-save peak")

    @property
    def replicate_max_peak_bytes(self) -> tuple[int, ...]:
        return tuple(max(peaks) for peaks in self.per_replicate_peak_bytes)

    @property
    def p_ref_bytes(self) -> int | float:
        return statistics.median(self.replicate_max_peak_bytes)

    @property
    def P_ref(self) -> int | float:
        return self.p_ref_bytes

    def physical_budget(self, ratio: float) -> int:
        _ratio(ratio)
        return math.floor(float(ratio) * self.p_ref_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_key": self.source_key,
            "backend": self.backend,
            "environment": self.environment,
            "replicate_seeds": list(self.replicate_seeds),
            "replicate_process_ids": list(self.replicate_process_ids),
            "per_replicate_peak_bytes": [list(peaks) for peaks in self.per_replicate_peak_bytes],
            "replicate_max_peak_bytes": list(self.replicate_max_peak_bytes),
            "p_ref_bytes": self.p_ref_bytes,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def policy_id_for(
    *,
    ratio: float,
    fw_graph_identity: str,
    bw_graph_identity: str,
    residual_identity: str,
    source_key: str,
    backend: Backend,
    environment: str,
) -> str:
    """Derive the stable policy ID from every canonical policy identity field."""

    _ratio(ratio)
    _nonempty_string(fw_graph_identity, "fw_graph_identity")
    _nonempty_string(bw_graph_identity, "bw_graph_identity")
    _nonempty_string(residual_identity, "residual_identity")
    _identity(source_key, backend, environment)
    return _sha256(
        {
            "ratio": _normalized_ratio(ratio),
            "fw_graph_identity": fw_graph_identity,
            "bw_graph_identity": bw_graph_identity,
            "residual_identity": residual_identity,
            "source_key": source_key,
            "backend": backend,
            "environment": environment,
        }
    )


@dataclass(frozen=True)
class MinCutPolicy:
    """A preregistered ratio with complete forward/backward compile provenance.

    ``policy_id`` must be the canonical hash of this record. Compilation aliases
    are permitted only when forward graph, backward graph, residual, source,
    backend, and environment identities are all equal; ratio is deliberately not
    part of ``compile_identity`` so explicit ratios may alias one executable.
    """

    policy_id: str
    ratio: float
    fw_graph_identity: str
    bw_graph_identity: str
    residual_identity: str
    source_key: str
    backend: Backend
    environment: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio", _normalized_ratio(self.ratio))
        expected = policy_id_for(
            ratio=self.ratio,
            fw_graph_identity=self.fw_graph_identity,
            bw_graph_identity=self.bw_graph_identity,
            residual_identity=self.residual_identity,
            source_key=self.source_key,
            backend=self.backend,
            environment=self.environment,
        )
        if self.policy_id != expected:
            raise ValueError("policy_id does not match the canonical policy content hash")

    @classmethod
    def create(
        cls,
        *,
        ratio: float,
        fw_graph_identity: str,
        bw_graph_identity: str,
        residual_identity: str,
        source_key: str,
        backend: Backend,
        environment: str,
    ) -> MinCutPolicy:
        return cls(
            policy_id=policy_id_for(
                ratio=ratio,
                fw_graph_identity=fw_graph_identity,
                bw_graph_identity=bw_graph_identity,
                residual_identity=residual_identity,
                source_key=source_key,
                backend=backend,
                environment=environment,
            ),
            ratio=ratio,
            fw_graph_identity=fw_graph_identity,
            bw_graph_identity=bw_graph_identity,
            residual_identity=residual_identity,
            source_key=source_key,
            backend=backend,
            environment=environment,
        )

    @property
    def compile_identity(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.source_key,
            self.backend,
            self.environment,
            self.fw_graph_identity,
            self.bw_graph_identity,
            self.residual_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "ratio": self.ratio,
            "fw_graph_identity": self.fw_graph_identity,
            "bw_graph_identity": self.bw_graph_identity,
            "residual_identity": self.residual_identity,
            "source_key": self.source_key,
            "backend": self.backend,
            "environment": self.environment,
        }


@dataclass(frozen=True)
class PolicyMeasurement:
    """One independent-process policy measurement.

    Tuning rows are policy-sweep observations and therefore have no budget slot:
    both budget fields must be ``None`` and status is ``ok`` or ``oom``.
    Evaluation rows belong to one frozen physical-budget slot and require both
    budget fields. Their ``ok``/``budget_violation`` status is recomputed from
    the maximum per-run peak. OOM rows never carry a synthetic peak or time.
    """

    policy_id: str
    source_key: str
    backend: Backend
    environment: str
    cohort: Cohort
    seed: int
    process_id: int
    budget_ratio: float | None
    budget_bytes: int | None
    per_run_peak_bytes: tuple[int, ...]
    median_event_step_seconds: float | None
    status: MeasurementStatus = "ok"

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_run_peak_bytes", tuple(self.per_run_peak_bytes))
        _fingerprint(self.policy_id, "policy_id")
        _identity(self.source_key, self.backend, self.environment)
        if self.cohort not in ("tuning", "evaluation"):
            raise ValueError("cohort must be 'tuning' or 'evaluation'")
        _seed(self.seed)
        _positive_int(self.process_id, "process_id")
        if self.status not in ("ok", "budget_violation", "oom"):
            raise ValueError("invalid measurement status")
        if self.cohort == "tuning":
            if self.budget_ratio is not None or self.budget_bytes is not None:
                raise ValueError("tuning policy-sweep measurements cannot name a budget slot")
            if self.status == "budget_violation":
                raise ValueError("tuning policy-sweep measurements have no budget to violate")
        else:
            if self.budget_ratio is None or self.budget_bytes is None:
                raise ValueError("evaluation measurements require a complete budget slot")
            object.__setattr__(self, "budget_ratio", _normalized_ratio(self.budget_ratio))
            if (
                isinstance(self.budget_bytes, bool)
                or not isinstance(self.budget_bytes, int)
                or self.budget_bytes < 0
            ):
                raise ValueError("budget_bytes must be a non-negative integer")
        if self.status == "oom":
            if self.per_run_peak_bytes or self.median_event_step_seconds is not None:
                raise ValueError("OOM measurements cannot contain a peak or event-step time")
            return
        if not self.per_run_peak_bytes:
            raise ValueError("non-OOM measurements require per-run peaks")
        for peak in self.per_run_peak_bytes:
            _positive_int(peak, "per-run peak")
        value = self.median_event_step_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non-OOM measurements require a median event-step time")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("median event-step time must be finite and positive")
        object.__setattr__(self, "median_event_step_seconds", float(value))
        if self.cohort == "evaluation":
            if self.status != self.recomputed_status:
                raise ValueError("evaluation status does not match its measured peak and budget")

    @property
    def replicate_max_peak_bytes(self) -> int | None:
        return max(self.per_run_peak_bytes) if self.per_run_peak_bytes else None

    @property
    def budget_slot(self) -> tuple[float, int] | None:
        if self.budget_ratio is None or self.budget_bytes is None:
            return None
        return self.budget_ratio, self.budget_bytes

    @property
    def recomputed_status(self) -> MeasurementStatus:
        """Recompute the auditable outcome from OOM state or the physical budget."""

        if self.status == "oom":
            return "oom"
        if self.cohort == "tuning":
            return "ok"
        if max(self.per_run_peak_bytes) <= self.budget_bytes:
            return "ok"
        return "budget_violation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "source_key": self.source_key,
            "backend": self.backend,
            "environment": self.environment,
            "cohort": self.cohort,
            "seed": self.seed,
            "process_id": self.process_id,
            "budget_ratio": self.budget_ratio,
            "budget_bytes": self.budget_bytes,
            "status": self.status,
            "per_run_peak_bytes": list(self.per_run_peak_bytes),
            "replicate_max_peak_bytes": self.replicate_max_peak_bytes,
            "median_event_step_seconds": self.median_event_step_seconds,
        }


def _measurement_evidence_sha256(measurements: Sequence[PolicyMeasurement]) -> str:
    rows = sorted((item.to_dict() for item in measurements), key=canonical_json)
    return _sha256(rows)


@dataclass(frozen=True)
class BudgetSelection:
    """Frozen tuning decision for one physical budget, including infeasibility.

    The evidence hash binds the decision to every preregistered tuning replicate.
    ``infeasible`` retains that evidence but has no selected-policy statistics.
    """

    ratio: float
    budget_bytes: int
    status: SelectionStatus
    policy_id: str | None
    tuning_max_peak_bytes: int | None
    tuning_median_event_step_seconds: float | None
    tuning_seeds: tuple[int, ...]
    tuning_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio", _normalized_ratio(self.ratio))
        object.__setattr__(self, "tuning_seeds", tuple(self.tuning_seeds))
        if isinstance(self.budget_bytes, bool) or not isinstance(self.budget_bytes, int):
            raise ValueError("budget_bytes must be a non-negative integer")
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes must be a non-negative integer")
        _fingerprint(self.tuning_evidence_sha256, "tuning_evidence_sha256")
        if self.status == "infeasible":
            if (
                self.policy_id is not None
                or self.tuning_max_peak_bytes is not None
                or self.tuning_median_event_step_seconds is not None
                or self.tuning_seeds
            ):
                raise ValueError("infeasible selections cannot contain selected-policy statistics")
            return
        if self.status != "selected":
            raise ValueError("selection status must be 'selected' or 'infeasible'")
        if self.policy_id is None:
            raise ValueError("selected budget requires a policy_id")
        _fingerprint(self.policy_id, "policy_id")
        if self.tuning_max_peak_bytes is None:
            raise ValueError("selected budget requires tuning peak evidence")
        _positive_int(self.tuning_max_peak_bytes, "tuning_max_peak_bytes")
        if self.tuning_max_peak_bytes > self.budget_bytes:
            raise ValueError("selected policy exceeds its physical budget")
        if not self.tuning_seeds or len(set(self.tuning_seeds)) != len(self.tuning_seeds):
            raise ValueError("selected budget requires unique tuning seeds")
        for seed in self.tuning_seeds:
            _seed(seed)
        time = self.tuning_median_event_step_seconds
        if isinstance(time, bool) or not isinstance(time, (int, float)):
            raise ValueError("selected budget requires a finite positive tuning time")
        if not math.isfinite(float(time)) or time <= 0:
            raise ValueError("selected budget requires a finite positive tuning time")
        object.__setattr__(self, "tuning_median_event_step_seconds", float(time))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "budget_bytes": self.budget_bytes,
            "status": self.status,
            "policy_id": self.policy_id,
            "tuning_max_peak_bytes": self.tuning_max_peak_bytes,
            "tuning_median_event_step_seconds": self.tuning_median_event_step_seconds,
            "tuning_seeds": list(self.tuning_seeds),
            "tuning_evidence_sha256": self.tuning_evidence_sha256,
        }


def derive_physical_budgets(
    reference: AllSaveReference, ratios: Sequence[float]
) -> tuple[int, ...]:
    """Return ``floor(ratio * P_ref)`` without inferring any policy ratio."""

    return tuple(reference.physical_budget(ratio) for ratio in ratios)


def build_budget_plan_from_records(
    records: Sequence[ExperimentRecord],
    *,
    ratios: Sequence[float],
    evidence_status: str = "provisional",
    min_reference_count: int = 5,
) -> dict[str, Any]:
    if evidence_status not in {"draft", "provisional", "frozen"}:
        raise ValueError("evidence_status must be draft, provisional, or frozen")
    if (
        isinstance(min_reference_count, bool)
        or not isinstance(min_reference_count, int)
        or min_reference_count <= 0
    ):
        raise ValueError("min_reference_count must be positive")
    normalized_ratios = tuple(_normalized_ratio(ratio) for ratio in ratios)
    if not normalized_ratios:
        raise ValueError("ratios must not be empty")
    if len(set(normalized_ratios)) != len(normalized_ratios):
        raise ValueError("ratios must be unique")

    grouped: dict[tuple[str, int, str, str, str], list[ExperimentRecord]] = {}
    for record in records:
        if record.status != "ok" or record.all_save_measured_peak_bytes is None:
            continue
        config = record.config_fingerprint or {}
        grouped.setdefault(
            (
                record.task_name,
                record.microbatch_size,
                str(config.get("capture_backend", "unknown")),
                str(config.get("compile_backend", "unknown")),
                str(config.get("device", "unknown")),
            ),
            [],
        ).append(record)
    cells: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key in sorted(grouped):
        task_name, microbatch_size, capture_backend, compile_backend, device = key
        samples = grouped[key]
        peaks = [int(record.all_save_measured_peak_bytes) for record in samples]
        p_ref = statistics.median(peaks)
        reference_count = len(peaks)
        complete = reference_count >= min_reference_count
        if not complete:
            warnings.append(
                f"{task_name}/{capture_backend}/{compile_backend}/mb{microbatch_size} has "
                f"{reference_count} all-save references; expected at least {min_reference_count}"
            )
        cells.append(
            {
                "cell_id": _sha256(
                    {
                        "task_name": task_name,
                        "microbatch_size": microbatch_size,
                        "capture_backend": capture_backend,
                        "compile_backend": compile_backend,
                        "device": device,
                    }
                ),
                "task_name": task_name,
                "microbatch_size": microbatch_size,
                "capture_backend": capture_backend,
                "compile_backend": compile_backend,
                "device": device,
                "reference_count": reference_count,
                "reference_complete": complete,
                "all_save_reference_peak_bytes": peaks,
                "p_ref_bytes": p_ref,
                "ratios": list(normalized_ratios),
                "physical_budgets_bytes": [math.floor(ratio * p_ref) for ratio in normalized_ratios],
                "source_records": [
                    {
                        "task_name": record.task_name,
                        "variant_name": record.variant_name,
                        "budget_bytes": record.budget_bytes,
                        "matrix_pass_index": record.matrix_pass_index,
                        "all_save_measured_peak_bytes": record.all_save_measured_peak_bytes,
                    }
                    for record in sorted(
                        samples,
                        key=lambda item: (
                            item.variant_name,
                            item.matrix_pass_index,
                            item.budget_bytes,
                            item.all_save_measured_peak_bytes or 0,
                        ),
                    )
                ],
            }
        )
    if not cells:
        raise ValueError("records contain no ok all-save reference measurements")
    complete = all(cell["reference_complete"] for cell in cells)
    if evidence_status == "frozen" and not complete:
        raise ValueError("frozen budget plan requires complete all-save references")
    return {
        "schema_version": BUDGET_PLAN_SCHEMA_VERSION,
        "evidence_status": evidence_status,
        "min_reference_count": min_reference_count,
        "ratios": list(normalized_ratios),
        "cell_count": len(cells),
        "complete": complete,
        "warning_count": len(warnings),
        "warnings": warnings,
        "cells": cells,
    }


def validate_budget_plan_manifest(
    budget_manifest: str | Path | dict[str, Any],
    *,
    require_frozen: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(budget_manifest, (str, Path)):
        path = Path(budget_manifest)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return _budget_plan_validation_payload(
                str(path),
                False,
                0,
                (f"invalid budget manifest: {exc}",),
                (),
                require_frozen,
            )
        source = str(path)
    else:
        payload = budget_manifest
        source = "<mapping>"
    if not isinstance(payload, dict):
        errors.append("budget manifest must contain a JSON object")
        return _budget_plan_validation_payload(source, False, 0, tuple(errors), (), require_frozen)
    if payload.get("schema_version") != BUDGET_PLAN_SCHEMA_VERSION:
        errors.append("unsupported budget plan schema_version")
    evidence_status = payload.get("evidence_status")
    if require_frozen and evidence_status != "frozen":
        errors.append("expected frozen budget manifest")
    min_reference_count = payload.get("min_reference_count")
    if isinstance(min_reference_count, bool) or not isinstance(min_reference_count, int) or min_reference_count <= 0:
        errors.append("min_reference_count must be a positive integer")
        min_reference_count = 1
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("budget manifest must list cells")
        cells = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cells[{index}] must be an object")
            continue
        reference_count = cell.get("reference_count")
        reference_complete = cell.get("reference_complete")
        peaks = cell.get("all_save_reference_peak_bytes")
        ratios = cell.get("ratios")
        budgets = cell.get("physical_budgets_bytes")
        if isinstance(reference_count, bool) or not isinstance(reference_count, int) or reference_count <= 0:
            errors.append(f"cells[{index}] reference_count must be positive")
        if not isinstance(peaks, list) or not all(isinstance(value, int) and value > 0 for value in peaks):
            errors.append(f"cells[{index}] has invalid all-save reference peaks")
        elif isinstance(reference_count, int) and len(peaks) != reference_count:
            errors.append(f"cells[{index}] reference_count does not match peak list")
        if isinstance(reference_count, int) and reference_count < int(min_reference_count):
            if reference_complete is not False:
                errors.append(f"cells[{index}] incomplete reference cell is not marked incomplete")
            warnings.append(f"cells[{index}] has fewer than {min_reference_count} all-save references")
        if not isinstance(ratios, list) or not isinstance(budgets, list) or len(ratios) != len(budgets):
            errors.append(f"cells[{index}] ratios and physical_budgets_bytes must have equal length")
        elif isinstance(peaks, list) and peaks:
            p_ref = statistics.median(peaks)
            expected = [math.floor(float(ratio) * p_ref) for ratio in ratios]
            if budgets != expected:
                errors.append(f"cells[{index}] physical budgets do not match floor(ratio * P_ref)")
    complete = bool(payload.get("complete"))
    if require_frozen and not complete:
        errors.append("frozen budget manifest must be complete")
    return _budget_plan_validation_payload(
        source,
        not errors,
        len(cells),
        tuple(errors),
        tuple(warnings),
        require_frozen,
    )


def _budget_plan_validation_payload(
    source: str,
    ok: bool,
    cell_count: int,
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    require_frozen: bool,
) -> dict[str, Any]:
    return {
        "schema_version": BUDGET_PLAN_SCHEMA_VERSION,
        "budget_manifest": source,
        "ok": ok,
        "require_frozen": require_frozen,
        "cell_count": cell_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": list(errors),
        "warnings": list(warnings),
    }


def compile_representatives(policies: Sequence[MinCutPolicy]) -> tuple[MinCutPolicy, ...]:
    """Compile once only for exact fw/bw/residual/source/backend/environment aliases."""

    representatives: dict[tuple[str, str, str, str, str, str], MinCutPolicy] = {}
    for policy in sorted(policies, key=lambda item: item.policy_id):
        representatives.setdefault(policy.compile_identity, policy)
    return tuple(representatives[key] for key in sorted(representatives))


def _validate_policy_measurements(
    policies: Sequence[MinCutPolicy], measurements: Sequence[PolicyMeasurement]
) -> dict[str, list[PolicyMeasurement]]:
    if not policies:
        raise ValueError("at least one preregistered policy is required")
    by_id: dict[str, MinCutPolicy] = {}
    for policy in policies:
        if policy.policy_id in by_id:
            raise ValueError(f"duplicate policy_id: {policy.policy_id}")
        by_id[policy.policy_id] = policy
    grouped = {policy_id: [] for policy_id in by_id}
    seen: set[tuple[str, str, int, float | None, int | None]] = set()
    seen_processes: set[tuple[str, int]] = set()
    for measurement in measurements:
        policy = by_id.get(measurement.policy_id)
        if policy is None:
            raise ValueError(f"measurement references unregistered policy: {measurement.policy_id}")
        if (measurement.source_key, measurement.backend, measurement.environment) != (
            policy.source_key,
            policy.backend,
            policy.environment,
        ):
            raise ValueError(f"measurement identity mismatch for policy {policy.policy_id}")
        slot = (measurement.budget_ratio, measurement.budget_bytes)
        seed_key = (measurement.policy_id, measurement.cohort, measurement.seed, *slot)
        process_key = (measurement.cohort, measurement.process_id)
        if seed_key in seen:
            raise ValueError("duplicate seed within a policy/cohort/budget slot")
        if process_key in seen_processes:
            raise ValueError("duplicate process identity within a cohort")
        seen.add(seed_key)
        seen_processes.add(process_key)
        grouped[measurement.policy_id].append(measurement)
    return grouped


def select_policy_for_budget(
    *,
    ratio: float,
    budget_bytes: int,
    policies: Sequence[MinCutPolicy],
    measurements: Sequence[PolicyMeasurement],
) -> BudgetSelection:
    """Freeze a discrete tuning choice without interpolation or monotonic assumptions.

    Every preregistered policy must have one to three independent tuning-process
    replicates. A policy is feasible only when every replicate completed and every
    replicate maximum is within the physical budget. Candidates minimize median
    replicate time, then maximum peak, then canonical policy ID. If none is
    feasible, an evidence-bound ``infeasible`` selection is returned.
    """

    _ratio(ratio)
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("budget_bytes must be a non-negative integer")
    if any(measurement.cohort != "tuning" for measurement in measurements):
        raise ValueError("policy selection accepts tuning policy-sweep measurements only")
    grouped = _validate_policy_measurements(policies, measurements)
    candidates: list[tuple[float, int, str, tuple[int, ...]]] = []
    for policy in policies:
        replicates = grouped[policy.policy_id]
        if not 1 <= len(replicates) <= 3:
            raise ValueError("each policy requires 1..3 independent tuning processes")
        if any(measurement.status != "ok" for measurement in replicates):
            continue
        max_peak = max(max(measurement.per_run_peak_bytes) for measurement in replicates)
        if max_peak > budget_bytes:
            continue
        median_time = statistics.median(
            float(measurement.median_event_step_seconds) for measurement in replicates
        )
        seeds = tuple(sorted(measurement.seed for measurement in replicates))
        candidates.append((median_time, max_peak, policy.policy_id, seeds))
    evidence = _measurement_evidence_sha256(measurements)
    if not candidates:
        return BudgetSelection(
            ratio=ratio,
            budget_bytes=budget_bytes,
            status="infeasible",
            policy_id=None,
            tuning_max_peak_bytes=None,
            tuning_median_event_step_seconds=None,
            tuning_seeds=(),
            tuning_evidence_sha256=evidence,
        )
    median_time, max_peak, policy_id, seeds = min(candidates)
    return BudgetSelection(
        ratio=ratio,
        budget_bytes=budget_bytes,
        status="selected",
        policy_id=policy_id,
        tuning_max_peak_bytes=max_peak,
        tuning_median_event_step_seconds=median_time,
        tuning_seeds=seeds,
        tuning_evidence_sha256=evidence,
    )


@dataclass(frozen=True)
class CalibrationManifest:
    """Complete calibration protocol with frozen tuning and held-out evaluation.

    Reference, tuning, and evaluation seed sets are pairwise disjoint. Tuning may
    reuse a seed across policies for paired independent-process sweeps, but each
    policy has one to three unique seed/process replicates. Every selected budget
    slot has at least five held-out unique seed/process replicates for exactly its
    frozen policy. Infeasible slots require no evaluation rows. Evaluation data
    cannot create or alter selections.
    """

    reference: AllSaveReference
    ratios: tuple[float, ...]
    policies: tuple[MinCutPolicy, ...]
    tuning_measurements: tuple[PolicyMeasurement, ...]
    evaluation_measurements: tuple[PolicyMeasurement, ...]
    selections: tuple[BudgetSelection, ...]
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratios", tuple(self.ratios))
        for ratio in self.ratios:
            _ratio(ratio)
        object.__setattr__(self, "ratios", tuple(_normalized_ratio(ratio) for ratio in self.ratios))
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "tuning_measurements", tuple(self.tuning_measurements))
        object.__setattr__(self, "evaluation_measurements", tuple(self.evaluation_measurements))
        object.__setattr__(self, "selections", tuple(self.selections))
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema: {self.schema_version}")
        if not self.ratios or len(set(self.ratios)) != len(self.ratios):
            raise ValueError("manifest ratios must be non-empty and unique")
        expected_identity = _identity(
            self.reference.source_key, self.reference.backend, self.reference.environment
        )
        for policy in self.policies:
            if (policy.source_key, policy.backend, policy.environment) != expected_identity:
                raise ValueError(f"policy identity mismatch for {policy.policy_id}")
        _validate_policy_measurements(
            self.policies, (*self.tuning_measurements, *self.evaluation_measurements)
        )
        if any(item.cohort != "tuning" for item in self.tuning_measurements):
            raise ValueError("tuning_measurements must contain only tuning cohort data")
        if any(item.cohort != "evaluation" for item in self.evaluation_measurements):
            raise ValueError("evaluation_measurements must contain only evaluation cohort data")

        reference_seeds = set(self.reference.replicate_seeds)
        tuning_seeds = {item.seed for item in self.tuning_measurements}
        evaluation_seeds = {item.seed for item in self.evaluation_measurements}
        if reference_seeds & tuning_seeds or reference_seeds & evaluation_seeds:
            raise ValueError("all-save, tuning, and evaluation seed sets must be pairwise disjoint")
        if tuning_seeds & evaluation_seeds:
            raise ValueError("all-save, tuning, and evaluation seed sets must be pairwise disjoint")

        budgets = derive_physical_budgets(self.reference, self.ratios)
        expected_selections = tuple(
            select_policy_for_budget(
                ratio=ratio,
                budget_bytes=budget,
                policies=self.policies,
                measurements=self.tuning_measurements,
            )
            for ratio, budget in zip(self.ratios, budgets)
        )
        if self.selections != expected_selections:
            raise ValueError("selections must be frozen exactly from preregistered tuning data")

        required_slots = {
            (selection.ratio, selection.budget_bytes, selection.policy_id)
            for selection in self.selections
            if selection.status == "selected"
        }
        evaluation_groups: dict[tuple[float, int, str], list[PolicyMeasurement]] = {}
        for measurement in self.evaluation_measurements:
            key = (measurement.budget_ratio, measurement.budget_bytes, measurement.policy_id)
            if key not in required_slots:
                raise ValueError("evaluation row does not match a frozen feasible budget selection")
            evaluation_groups.setdefault(key, []).append(measurement)
        for slot in required_slots:
            rows = evaluation_groups.get(slot, [])
            if len(rows) < 5:
                raise ValueError(
                    "each feasible frozen budget requires at least 5 evaluation processes"
                )
            if len({row.seed for row in rows}) != len(rows):
                raise ValueError("evaluation budget processes must have unique seeds")
            if len({row.process_id for row in rows}) != len(rows):
                raise ValueError("evaluation budget processes must have unique process IDs")

    @property
    def compile_policies(self) -> tuple[MinCutPolicy, ...]:
        return compile_representatives(self.policies)

    @property
    def policy_aliases(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        aliases: dict[tuple[str, str, str, str, str, str], list[str]] = {}
        for policy in self.policies:
            aliases.setdefault(policy.compile_identity, []).append(policy.policy_id)
        return tuple((min(ids), tuple(sorted(ids))) for _, ids in sorted(aliases.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference": self.reference.to_dict(),
            "ratios": list(self.ratios),
            "physical_budgets_bytes": list(derive_physical_budgets(self.reference, self.ratios)),
            "policies": [policy.to_dict() for policy in self.policies],
            "compile_policy_ids": [policy.policy_id for policy in self.compile_policies],
            "policy_aliases": [
                {"compile_policy_id": representative, "policy_ids": list(ids)}
                for representative, ids in self.policy_aliases
            ],
            "tuning_measurements": [item.to_dict() for item in self.tuning_measurements],
            "evaluation_measurements": [item.to_dict() for item in self.evaluation_measurements],
            "selections": [selection.to_dict() for selection in self.selections],
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def physical_budget_to_ratio(*_args: Any, **_kwargs: Any) -> float:
    """Reject the legacy ``(B - F) / Smax`` conversion by construction."""

    raise LegacyBudgetConversionError(
        "physical budgets are not converted to activation ratios; use preregistered explicit ratios"
    )


convert_physical_budget_to_ratio = physical_budget_to_ratio
convert_physical_budget_to_activation_memory_budget = physical_budget_to_ratio
