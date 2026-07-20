import json

from peakaware.experiments import summarize_experiment_records, write_experiment_json, write_experiment_summary_json
from peakaware.publication.artifact import build_publication_artifact_manifest
from peakaware.publication.evidence import evaluate_publication_evidence_gates
from peakaware.publication.figures import build_publication_figures
from peakaware.publication.tables import build_publication_tables
from tests.unit.test_publication_figures import _record


def test_evaluate_publication_evidence_gates_reports_draft_gaps(tmp_path):
    records = (_record(),)
    write_experiment_json(records, tmp_path / "records.json")
    write_experiment_summary_json(summarize_experiment_records(records), tmp_path / "summary.json")
    build_publication_figures(records, tmp_path / "figures", source_paths=(tmp_path / "records.json",))
    build_publication_tables(records, tmp_path / "tables", source_paths=(tmp_path / "records.json",))
    build_publication_artifact_manifest(
        tmp_path,
        run_id="draft-test",
        scope="draft",
        evidence_status="draft",
    )

    payload = evaluate_publication_evidence_gates(tmp_path)

    assert payload["ok"] is False
    assert payload["record_count"] == 1
    assert any(gate["gate_id"] == "G-8" and gate["passed"] for gate in payload["gates"])
    assert any(
        "manifest evidence_status is not frozen" in error
        for gate in payload["gates"]
        for error in gate["errors"]
    )


def test_evaluate_publication_evidence_gates_accepts_explicit_manifest_paths(tmp_path):
    records = (_record(),)
    write_experiment_json(records, tmp_path / "records.json")
    (tmp_path / "workloads.json").write_text(
        json.dumps({"workloads": [{"workload": {"display_name": "Tiny"}}] * 4}),
        encoding="utf-8",
    )
    (tmp_path / "budgets.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "task_name": f"task-{index}",
                        "ratios": [0.5, 0.65, 0.8, 0.95, 1.0],
                        "reference_count": 5,
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    build_publication_figures(records, tmp_path / "figures", source_paths=(tmp_path / "records.json",))
    build_publication_tables(records, tmp_path / "tables", source_paths=(tmp_path / "records.json",))

    payload = evaluate_publication_evidence_gates(
        tmp_path,
        workload_manifest=tmp_path / "workloads.json",
        budget_manifest=tmp_path / "budgets.json",
    )

    gates = {gate["gate_id"]: gate for gate in payload["gates"]}
    assert gates["G-2"]["metadata"]["manifest_workload_count"] == 4
    assert gates["G-5"]["metadata"]["cell_count"] == 4


def test_evidence_gate_distinguishes_proxy_and_real_baseline_rows(tmp_path):
    record = _record()
    rows = []
    for row in record.measured_plan_results:
        next_row = dict(row)
        if next_row["plan_id"] == "torch_min_cut":
            next_row["strategy_provenance"] = {"source": "pytorch_min_cut_proxy"}
        elif next_row["plan_id"] == "peakaware":
            next_row["plan_id"] = "sac"
            next_row["strategy_provenance"] = {"source": "real_torch_sac", "api": "torch.utils.checkpoint"}
            next_row["runtime_marker"] = {"is_real": True}
        rows.append(next_row)
    mutated = record.__class__(**{**record.__dict__, "measured_plan_results": tuple(rows)})
    write_experiment_json((mutated,), tmp_path / "records.json")

    payload = evaluate_publication_evidence_gates(tmp_path, records_json=tmp_path / "records.json")

    g1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "G-1")
    assert g1["metadata"]["identity_counts"]["proxy"] == 1
    assert g1["metadata"]["identity_counts"]["real"] == 1


def test_evidence_gate_uses_qualification_summary_for_real_baseline_coverage(tmp_path):
    record = _record()
    write_experiment_json((record,), tmp_path / "records.json")
    summary = {
        "qualification_passed": False,
        "matrix": {"tasks": ["a", "b", "c", "d"]},
        "method_qualification": {
            "aot_eager": {
                "pytorch_min_cut": {"qualified": 2, "unsupported": 0, "failure": 0},
                "block_ac": {"qualified": 2, "unsupported": 0, "failure": 0},
                "sac": {"qualified": 2, "unsupported": 0, "failure": 0},
            }
        },
    }
    summary_path = tmp_path / "qualification.records.jsonl.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    payload = evaluate_publication_evidence_gates(
        tmp_path,
        records_json=tmp_path / "records.json",
        qualification_summary=summary_path,
    )

    g1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "G-1")
    assert g1["passed"] is True
    assert g1["metadata"]["qualified_methods"] == ["block_ac", "pytorch_min_cut", "sac"]
    assert g1["metadata"]["gate_qualified_methods"] == ["block_ac", "pytorch_min_cut", "sac"]


def test_evidence_gate_does_not_count_narrow_qualification_summary_for_formal_g1(tmp_path):
    write_experiment_json((_record(),), tmp_path / "records.json")
    summary_path = tmp_path / "qualification.records.jsonl.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "qualification_passed": True,
                "matrix": {"tasks": ["gpt2"]},
                "method_qualification": {
                    "aot_eager": {
                        "pytorch_min_cut": {"qualified": 1, "unsupported": 0, "failure": 0},
                        "block_ac": {"qualified": 1, "unsupported": 0, "failure": 0},
                        "sac": {"qualified": 1, "unsupported": 0, "failure": 0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = evaluate_publication_evidence_gates(
        tmp_path,
        records_json=tmp_path / "records.json",
        qualification_summary=summary_path,
    )

    g1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "G-1")
    assert g1["passed"] is False
    assert g1["metadata"]["qualified_methods"] == ["block_ac", "pytorch_min_cut", "sac"]
    assert g1["metadata"]["gate_qualified_methods"] == []
    assert g1["metadata"]["qualification_scope"]["task_count"] == 1


def test_evidence_gate_reports_missing_explicit_qualification_summary(tmp_path):
    write_experiment_json((_record(),), tmp_path / "records.json")

    payload = evaluate_publication_evidence_gates(
        tmp_path,
        records_json=tmp_path / "records.json",
        qualification_summary=tmp_path / "missing.summary.json",
    )

    assert any(gate["gate_id"] == "G-qualification" for gate in payload["gates"])
    assert payload["ok"] is False
