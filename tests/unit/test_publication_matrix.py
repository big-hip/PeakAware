import json

from peakaware.publication.matrix import (
    load_publication_matrix_cells,
    run_publication_matrix_from_budget_plan,
)
from tests.unit.test_publication_figures import _record


def _budget_manifest():
    return {
        "schema_version": "0.1",
        "evidence_status": "draft",
        "min_reference_count": 1,
        "ratios": [0.5, 1.0],
        "cell_count": 1,
        "complete": True,
        "warning_count": 0,
        "warnings": [],
        "cells": [
            {
                "cell_id": "cell",
                "task_name": "synthetic",
                "microbatch_size": 1,
                "capture_backend": "fx",
                "compile_backend": "eager",
                "device": "cpu",
                "reference_count": 1,
                "reference_complete": True,
                "all_save_reference_peak_bytes": [100],
                "p_ref_bytes": 100,
                "ratios": [0.5, 1.0],
                "physical_budgets_bytes": [50, 100],
                "source_records": [],
            }
        ],
    }


def test_load_publication_matrix_cells_respects_limits():
    cells = load_publication_matrix_cells(_budget_manifest(), limit_cells=1, limit_budgets=1)

    assert len(cells) == 1
    assert cells[0].task_name == "synthetic"
    assert cells[0].budget_bytes == (50,)


def test_run_publication_matrix_from_budget_plan_writes_outputs(tmp_path):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return (_record(),)

    result = run_publication_matrix_from_budget_plan(
        _budget_manifest(),
        tmp_path,
        diagnostic_hints="both",
        limit_budgets=1,
        runner=fake_runner,
    )

    assert len(result.records) == 2
    assert [call["variant_name"] for call in calls] == ["diagnostic_hints_on", "diagnostic_hints_off"]
    assert all(call["budget_bytes"] == (50,) for call in calls)
    assert (tmp_path / "records.json").is_file()
    assert (tmp_path / "raw" / "records.json").is_file()
    assert (tmp_path / "derived" / "summary.json").is_file()
    assert (tmp_path / "figures" / "F4_baseline_comparison" / "figure.svg").is_file()
    summary = json.loads((tmp_path / "derived" / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_records"] == 2


def test_run_publication_matrix_repeats_independent_matrix_passes(tmp_path):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return (_record(),)

    result = run_publication_matrix_from_budget_plan(
        _budget_manifest(),
        tmp_path,
        diagnostic_hints="both",
        matrix_passes=3,
        limit_budgets=1,
        runner=fake_runner,
    )

    assert len(result.records) == 6
    assert [call["matrix_pass_index"] for call in calls] == [0, 0, 1, 1, 2, 2]
    assert all(call["matrix_pass_count"] == 3 for call in calls)
    assert [call["variant_name"] for call in calls[:2]] == [
        "diagnostic_hints_on",
        "diagnostic_hints_off",
    ]


def test_run_publication_matrix_rejects_invalid_matrix_passes(tmp_path):
    try:
        run_publication_matrix_from_budget_plan(
            _budget_manifest(),
            tmp_path,
            matrix_passes=0,
            runner=lambda **_: (),
        )
    except ValueError as exc:
        assert "matrix_passes must be positive" in str(exc)
    else:
        raise AssertionError("non-positive matrix_passes was accepted")


def test_run_publication_matrix_requires_frozen_budget_when_requested(tmp_path):
    try:
        run_publication_matrix_from_budget_plan(
            _budget_manifest(),
            tmp_path,
            require_frozen_budget=True,
            runner=lambda **_: (),
        )
    except ValueError as exc:
        assert "expected frozen budget manifest" in str(exc)
    else:
        raise AssertionError("draft budget manifest was accepted as frozen input")


def test_run_publication_matrix_can_emit_frozen_figures_with_ev_id(tmp_path):
    result = run_publication_matrix_from_budget_plan(
        _budget_manifest(),
        tmp_path,
        diagnostic_hints="on",
        limit_budgets=1,
        figure_status="frozen",
        figure_ev_ids=("EV-TEST",),
        runner=lambda **_: (_record(),),
    )

    assert len(result.records) == 1
    provenance = json.loads(
        (tmp_path / "figures" / "F2_pareto" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "frozen"
    assert provenance["ev_ids"] == ["EV-TEST"]
