import csv
import json

from peakaware.publication.tables import (
    PUBLICATION_TABLE_IDS,
    build_publication_tables,
    validate_publication_table_artifacts,
)
from tests.unit.test_publication_figures import _record


def test_build_publication_tables_writes_markdown_csv_latex_and_provenance(tmp_path):
    artifacts = build_publication_tables((_record(),), tmp_path / "tables", source_paths=(tmp_path / "records.json",))

    assert {artifact.table_id for artifact in artifacts} == set(PUBLICATION_TABLE_IDS)
    t3 = tmp_path / "tables" / "T3_budget_results"
    assert (t3 / "table.md").is_file()
    assert (t3 / "table.tex").is_file()
    assert (t3 / "source.csv").is_file()
    assert (t3 / "source.schema.json").is_file()
    assert (t3 / "caption.md").is_file()
    provenance = json.loads((t3 / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["table_id"] == "T3"
    assert provenance["status"] == "draft"
    with (t3 / "source.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task_name"] == "synthetic"


def test_validate_publication_table_artifacts_reports_ok_and_tampering(tmp_path):
    build_publication_tables((_record(),), tmp_path / "tables")

    payload = validate_publication_table_artifacts(tmp_path / "tables")
    assert payload["ok"] is True

    (tmp_path / "tables" / "T3_budget_results" / "source.csv").write_text("broken\n", encoding="utf-8")
    failed = validate_publication_table_artifacts(tmp_path / "tables")
    assert failed["ok"] is False
    assert any(
        "row_count mismatch" in error or "source_csv_sha256 mismatch" in error
        for table in failed["tables"]
        for error in table["errors"]
    )
