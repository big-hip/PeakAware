import json

from peakaware.experiments import (
    experiment_records_from_dicts,
    summarize_experiment_records,
    write_experiment_json,
    write_experiment_summary_json,
)
from peakaware.publication.artifact import (
    build_publication_artifact_manifest,
    validate_publication_artifact_manifest,
    verify_publication_artifact,
)
from peakaware.publication.figures import build_publication_figures
from peakaware.publication.tables import build_publication_tables
from tests.unit.test_publication_figures import _record


def _build_artifact(root):
    records = (_record(),)
    write_experiment_json(records, root / "records.json")
    write_experiment_summary_json(summarize_experiment_records(records), root / "summary.json")
    build_publication_figures(
        records,
        root / "figures",
        status="draft",
        source_paths=(root / "records.json",),
    )


def _build_artifact_with_tables(root):
    _build_artifact(root)
    records = experiment_records_from_dicts(json.loads((root / "records.json").read_text(encoding="utf-8")))
    build_publication_tables(records, root / "tables", source_paths=(root / "records.json",))


def test_verify_publication_artifact_accepts_smoke_root(tmp_path):
    _build_artifact(tmp_path)

    payload = verify_publication_artifact(tmp_path)

    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert {check["name"] for check in payload["checks"]} == {
        "records",
        "summary",
        "figures",
        "records_to_figures",
    }


def test_verify_publication_artifact_rejects_summary_count_mismatch(tmp_path):
    _build_artifact(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["total_records"] = 999
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    payload = verify_publication_artifact(tmp_path)

    assert payload["ok"] is False
    assert any(
        "summary record count" in error
        for check in payload["checks"]
        for error in check["errors"]
    )


def test_verify_publication_artifact_require_frozen_delegates_to_figures(tmp_path):
    _build_artifact(tmp_path)

    payload = verify_publication_artifact(tmp_path, require_frozen=True)

    assert payload["ok"] is False
    assert any(
        "expected frozen figure status" in error
        for check in payload["checks"]
        for error in check["errors"]
    )


def test_verify_publication_artifact_validates_tables_when_present(tmp_path):
    _build_artifact_with_tables(tmp_path)

    payload = verify_publication_artifact(tmp_path)

    assert payload["ok"] is True
    assert "tables" in {check["name"] for check in payload["checks"]}


def test_verify_publication_artifact_require_frozen_delegates_to_tables(tmp_path):
    _build_artifact_with_tables(tmp_path)

    payload = verify_publication_artifact(tmp_path, require_frozen=True)

    assert payload["ok"] is False
    assert any(
        "expected frozen table status" in error
        for check in payload["checks"]
        for error in check["errors"]
    )


def test_build_and_validate_publication_artifact_manifest(tmp_path):
    _build_artifact(tmp_path)

    manifest = build_publication_artifact_manifest(
        tmp_path,
        run_id="draft-test",
        scope="draft",
        evidence_status="draft",
        known_limitations=("unit-test artifact",),
    )
    payload = validate_publication_artifact_manifest(tmp_path)
    full = verify_publication_artifact(tmp_path)

    assert manifest["run_id"] == "draft-test"
    assert manifest["files"]
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "checksums.sha256").is_file()
    assert payload["ok"] is True
    assert {check["name"] for check in full["checks"]} == {
        "records",
        "summary",
        "figures",
        "records_to_figures",
        "manifest",
        "checksums",
        "manifest_checksums",
    }


def test_publication_artifact_manifest_detects_tampering(tmp_path):
    _build_artifact(tmp_path)
    build_publication_artifact_manifest(
        tmp_path,
        run_id="draft-test",
        scope="draft",
        evidence_status="draft",
    )
    (tmp_path / "summary.json").write_text('{"total_records": 999}\n', encoding="utf-8")

    payload = validate_publication_artifact_manifest(tmp_path)

    assert payload["ok"] is False
    assert any(
        "mismatch" in error
        for check in payload["checks"]
        for error in check["errors"]
    )


def test_publication_artifact_manifest_require_frozen(tmp_path):
    _build_artifact(tmp_path)
    build_publication_artifact_manifest(
        tmp_path,
        run_id="draft-test",
        scope="draft",
        evidence_status="draft",
    )

    payload = validate_publication_artifact_manifest(tmp_path, require_frozen=True)

    assert payload["ok"] is False
    assert any(
        "expected frozen manifest" in error
        for check in payload["checks"]
        for error in check["errors"]
    )


def test_publication_artifact_manifest_requires_checksum_alignment(tmp_path):
    _build_artifact(tmp_path)
    build_publication_artifact_manifest(
        tmp_path,
        run_id="draft-test",
        scope="draft",
        evidence_status="draft",
    )
    checksum_path = tmp_path / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    payload = validate_publication_artifact_manifest(tmp_path)

    assert payload["ok"] is False
    assert any(
        "checksums missing manifest files" in error
        for check in payload["checks"]
        for error in check["errors"]
    )
