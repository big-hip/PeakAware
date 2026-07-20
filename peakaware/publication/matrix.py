from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from peakaware.config import PeakAwareConfig
from peakaware.experiments import (
    ExperimentRecord,
    experiment_records_to_dicts,
    run_experiment_matrix,
    summarize_experiment_records,
    summarize_experiment_records_by_variant,
    write_experiment_baseline_comparison_json,
    write_experiment_cache_reuse_json,
    write_experiment_hint_ablation_json,
    write_experiment_json,
    write_experiment_layered_accuracy_json,
    write_experiment_simulation_error_json,
    write_experiment_steady_state_json,
    write_experiment_summary_json,
    write_experiment_variant_summary_json,
)
from peakaware.publication.budget_calibration import validate_budget_plan_manifest
from peakaware.publication.figures import build_publication_figures


PUBLICATION_MATRIX_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class PublicationMatrixCell:
    task_name: str
    microbatch_size: int
    capture_backend: str
    compile_backend: str
    device: str
    budget_bytes: tuple[int, ...]


@dataclass(frozen=True)
class PublicationMatrixRun:
    records: tuple[ExperimentRecord, ...]
    cells: tuple[PublicationMatrixCell, ...]
    output_root: Path


ExperimentRunner = Callable[..., tuple[ExperimentRecord, ...]]


def load_publication_matrix_cells(
    budget_manifest: str | Path | dict[str, Any],
    *,
    require_frozen_budget: bool = False,
    limit_cells: int | None = None,
    limit_budgets: int | None = None,
) -> tuple[PublicationMatrixCell, ...]:
    if isinstance(budget_manifest, (str, Path)):
        payload = json.loads(Path(budget_manifest).read_text(encoding="utf-8"))
    else:
        payload = budget_manifest
    validation = validate_budget_plan_manifest(payload, require_frozen=require_frozen_budget)
    if not validation["ok"]:
        raise ValueError(f"invalid budget manifest: {validation['errors']}")
    raw_cells = payload["cells"]
    if limit_cells is not None:
        if limit_cells <= 0:
            raise ValueError("limit_cells must be positive")
        raw_cells = raw_cells[:limit_cells]
    cells: list[PublicationMatrixCell] = []
    for raw in raw_cells:
        budgets = tuple(int(value) for value in raw["physical_budgets_bytes"])
        if limit_budgets is not None:
            if limit_budgets <= 0:
                raise ValueError("limit_budgets must be positive")
            budgets = budgets[:limit_budgets]
        cells.append(
            PublicationMatrixCell(
                task_name=str(raw["task_name"]),
                microbatch_size=int(raw["microbatch_size"]),
                capture_backend=str(raw["capture_backend"]),
                compile_backend=str(raw["compile_backend"]),
                device=str(raw["device"]),
                budget_bytes=budgets,
            )
        )
    return tuple(cells)


def run_publication_matrix_from_budget_plan(
    budget_manifest: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    diagnostic_hints: str = "both",
    matrix_passes: int = 1,
    top_k: int = 3,
    measurement_warmup_steps: int = 1,
    measurement_repeats: int = 3,
    selection_objective: str = "min_peak_then_time",
    require_frozen_budget: bool = False,
    limit_cells: int | None = None,
    limit_budgets: int | None = None,
    device_override: str | None = None,
    plan_artifact_dir: str | Path | None = None,
    figure_status: str = "draft",
    figure_ev_ids: Sequence[str] = (),
    runner: ExperimentRunner = run_experiment_matrix,
) -> PublicationMatrixRun:
    if diagnostic_hints == "both":
        variants = (("diagnostic_hints_on", True), ("diagnostic_hints_off", False))
    elif diagnostic_hints in {"on", "off"}:
        variants = ((f"diagnostic_hints_{diagnostic_hints}", diagnostic_hints == "on"),)
    else:
        raise ValueError("diagnostic_hints must be on, off, or both")
    if matrix_passes <= 0:
        raise ValueError("matrix_passes must be positive")
    root = Path(output_root)
    raw_root = root / "raw"
    derived_root = root / "derived"
    figure_root = root / "figures"
    raw_root.mkdir(parents=True, exist_ok=True)
    derived_root.mkdir(parents=True, exist_ok=True)
    cells = load_publication_matrix_cells(
        budget_manifest,
        require_frozen_budget=require_frozen_budget,
        limit_cells=limit_cells,
        limit_budgets=limit_budgets,
    )
    records: list[ExperimentRecord] = []
    for pass_index in range(matrix_passes):
        for variant_name, hints_enabled in variants:
            for cell in cells:
                config = _config_for_matrix_cell(
                    cell,
                    enable_diagnostic_hints=hints_enabled,
                    top_k=top_k,
                    measurement_warmup_steps=measurement_warmup_steps,
                    measurement_repeats=measurement_repeats,
                    selection_objective=selection_objective,
                )
                records.extend(
                    runner(
                        task_names=(cell.task_name,),
                        microbatch_sizes=(cell.microbatch_size,),
                        budget_bytes=cell.budget_bytes,
                        config=config,
                        variant_name=variant_name,
                        device=device_override or cell.device,
                        plan_artifact_dir=plan_artifact_dir,
                        matrix_pass_index=pass_index,
                        matrix_pass_count=matrix_passes,
                    )
                )
    record_tuple = tuple(records)
    _write_publication_matrix_outputs(
        record_tuple,
        root,
        raw_root,
        derived_root,
        figure_root,
        figure_status,
        tuple(figure_ev_ids),
    )
    return PublicationMatrixRun(record_tuple, cells, root)


def _config_for_matrix_cell(
    cell: PublicationMatrixCell,
    *,
    enable_diagnostic_hints: bool,
    top_k: int,
    measurement_warmup_steps: int,
    measurement_repeats: int,
    selection_objective: str,
) -> PeakAwareConfig:
    enable_inductor = cell.compile_backend == "inductor"
    enable_compile = enable_inductor or cell.compile_backend == "aot_eager"
    return PeakAwareConfig(
        top_k=top_k,
        enable_compile=enable_compile,
        enable_inductor=enable_inductor,
        capture_backend=cell.capture_backend if cell.capture_backend in {"auto", "aot", "fx"} else "auto",
        enable_diagnostic_hints=enable_diagnostic_hints,
        measurement_warmup_steps=measurement_warmup_steps,
        measurement_repeats=measurement_repeats,
        selection_objective=selection_objective,
    )


def _write_publication_matrix_outputs(
    records: tuple[ExperimentRecord, ...],
    root: Path,
    raw_root: Path,
    derived_root: Path,
    figure_root: Path,
    figure_status: str,
    figure_ev_ids: tuple[str, ...],
) -> None:
    records_json = root / "records.json"
    write_experiment_json(records, records_json)
    (raw_root / "records.json").write_text(
        json.dumps(experiment_records_to_dicts(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_experiment_summary_json(summarize_experiment_records(records), derived_root / "summary.json")
    write_experiment_variant_summary_json(
        summarize_experiment_records_by_variant(records),
        derived_root / "variant_summary.json",
    )
    write_experiment_baseline_comparison_json(records, derived_root / "baseline_comparison.json")
    write_experiment_hint_ablation_json(records, derived_root / "hint_ablation.json")
    write_experiment_cache_reuse_json(records, derived_root / "cache_reuse.json")
    write_experiment_layered_accuracy_json(records, derived_root / "layered_accuracy.json")
    write_experiment_simulation_error_json(records, derived_root / "simulation_error.json")
    write_experiment_steady_state_json(records, derived_root / "steady_state.json")
    build_publication_figures(
        records,
        figure_root,
        status=figure_status,
        ev_ids=figure_ev_ids,
        source_paths=(records_json,),
    )
