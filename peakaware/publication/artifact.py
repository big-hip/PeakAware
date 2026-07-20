from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from peakaware.experiments import experiment_records_from_dicts
from peakaware.publication.figures import validate_publication_figure_artifacts
from peakaware.publication.tables import validate_publication_table_artifacts


PUBLICATION_ARTIFACT_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class PublicationArtifactCheck:
    name: str
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def verify_publication_artifact(
    artifact_root: str | Path,
    *,
    records_json: str | Path | None = None,
    summary_json: str | Path | None = None,
    figures_root: str | Path | None = None,
    tables_root: str | Path | None = None,
    manifest_json: str | Path | None = None,
    checksums_path: str | Path | None = None,
    require_frozen: bool = False,
) -> dict[str, Any]:
    """Validate a PeakAware publication artifact root.

    This verifier is intentionally structural: it proves the artifact is
    internally traceable enough for review, but it does not upgrade draft data to
    frozen evidence.  Frozen status still requires the evidence ledger gates.
    """

    root = Path(artifact_root)
    checks: list[PublicationArtifactCheck] = []
    if not root.is_dir():
        checks.append(PublicationArtifactCheck("artifact_root", False, (f"missing artifact root: {root}",)))
        return _summary(root, checks, require_frozen)

    records_path = Path(records_json) if records_json is not None else _first_existing(
        root / "records.json",
        root / "raw" / "records.json",
    )
    summary_path = Path(summary_json) if summary_json is not None else _first_existing(
        root / "summary.json",
        root / "derived" / "summary.json",
    )
    figures_path = Path(figures_root) if figures_root is not None else root / "figures"
    tables_path = Path(tables_root) if tables_root is not None else root / "tables"
    manifest_path = Path(manifest_json) if manifest_json is not None else root / "manifest.json"
    checksum_path = Path(checksums_path) if checksums_path is not None else root / "checksums.sha256"

    records_payload: list[dict[str, Any]] | None = None
    record_count: int | None = None
    if records_path is None or not records_path.is_file():
        checks.append(PublicationArtifactCheck("records", False, ("missing records.json",)))
    else:
        try:
            raw_payload = json.loads(records_path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, list):
                raise ValueError("records JSON must contain a list")
            records = experiment_records_from_dicts(raw_payload)
            records_payload = raw_payload
            record_count = len(records)
        except Exception as exc:
            checks.append(PublicationArtifactCheck("records", False, (f"invalid records.json: {exc}",)))
        else:
            checks.append(
                PublicationArtifactCheck(
                    "records",
                    True,
                    metadata={"path": str(records_path), "record_count": record_count},
                )
            )

    if summary_path is None or not summary_path.is_file():
        checks.append(PublicationArtifactCheck("summary", False, ("missing summary.json",)))
    else:
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary_payload, dict):
                raise ValueError("summary JSON must contain an object")
        except Exception as exc:
            checks.append(PublicationArtifactCheck("summary", False, (f"invalid summary.json: {exc}",)))
        else:
            errors: list[str] = []
            summary_count = summary_payload.get("total_records", summary_payload.get("record_count"))
            if record_count is not None and summary_count is not None and int(summary_count) != record_count:
                errors.append("summary record count does not match records.json")
            checks.append(
                PublicationArtifactCheck(
                    "summary",
                    not errors,
                    tuple(errors),
                    metadata={"path": str(summary_path), "record_count": summary_count},
                )
            )

    figures_payload = validate_publication_figure_artifacts(figures_path, require_frozen=require_frozen)
    checks.append(
        PublicationArtifactCheck(
            "figures",
            bool(figures_payload["ok"]),
            tuple(
                f"{figure['figure_dir']}: {error}"
                for figure in figures_payload["figures"]
                for error in figure["errors"]
            ),
            tuple(
                f"{figure['figure_dir']}: {warning}"
                for figure in figures_payload["figures"]
                for warning in figure["warnings"]
            ),
            metadata={
                "path": str(figures_path),
                "figure_count": len(figures_payload["figures"]),
                "figure_error_count": figures_payload["error_count"],
                "figure_warning_count": figures_payload["warning_count"],
            },
        )
    )

    if records_payload is not None:
        checks.append(
            PublicationArtifactCheck(
                "records_to_figures",
                True,
                metadata={
                    "records_path": str(records_path),
                    "figures_path": str(figures_path),
                    "record_count": len(records_payload),
                },
            )
        )
    if tables_root is not None or tables_path.is_dir():
        tables_payload = validate_publication_table_artifacts(tables_path, require_frozen=require_frozen)
        checks.append(
            PublicationArtifactCheck(
                "tables",
                bool(tables_payload["ok"]),
                tuple(
                    f"{table['table_dir']}: {error}"
                    for table in tables_payload["tables"]
                    for error in table["errors"]
                ),
                tuple(
                    f"{table['table_dir']}: {warning}"
                    for table in tables_payload["tables"]
                    for warning in table["warnings"]
                ),
                metadata={
                    "path": str(tables_path),
                    "table_count": len(tables_payload["tables"]),
                    "table_error_count": tables_payload["error_count"],
                    "table_warning_count": tables_payload["warning_count"],
                },
            )
        )
    if manifest_json is not None or manifest_path.is_file():
        checks.append(_check_manifest(root, manifest_path, require_frozen=require_frozen))
    if checksums_path is not None or checksum_path.is_file():
        checks.append(_check_checksums(root, checksum_path))
    if (manifest_json is not None or manifest_path.is_file()) and (
        checksums_path is not None or checksum_path.is_file()
    ):
        checks.append(_check_manifest_checksum_alignment(root, manifest_path, checksum_path))
    return _summary(root, checks, require_frozen)


def build_publication_artifact_manifest(
    artifact_root: str | Path,
    *,
    run_id: str,
    scope: str,
    evidence_status: str,
    output_manifest: str | Path | None = None,
    output_checksums: str | Path | None = None,
    known_limitations: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = Path(artifact_root)
    if not root.is_dir():
        raise FileNotFoundError(f"missing artifact root: {root}")
    if not run_id:
        raise ValueError("run_id must not be empty")
    if scope not in {"smoke", "full", "release", "draft"}:
        raise ValueError("scope must be smoke, full, release, or draft")
    if evidence_status not in {"smoke", "draft", "provisional", "frozen", "invalid"}:
        raise ValueError("unsupported evidence_status")

    manifest_path = Path(output_manifest) if output_manifest is not None else root / "manifest.json"
    checksums_path = Path(output_checksums) if output_checksums is not None else root / "checksums.sha256"
    excluded = {_relative_path(root, manifest_path), _relative_path(root, checksums_path)}
    files = _artifact_file_entries(root, excluded_paths=excluded)
    checksum_lines = [f"{entry['sha256']}  {entry['path']}" for entry in files]
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")

    manifest = {
        "artifact_schema_version": PUBLICATION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": scope,
        "evidence_status": evidence_status,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git": _git_state(Path(__file__).resolve().parents[2]),
        "files": files,
        "checksums_path": _relative_path(root, checksums_path),
        "known_limitations": list(known_limitations),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_publication_artifact_manifest(
    artifact_root: str | Path,
    *,
    manifest_json: str | Path | None = None,
    checksums_path: str | Path | None = None,
    require_frozen: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_root)
    manifest_path = Path(manifest_json) if manifest_json is not None else root / "manifest.json"
    checksum_path = Path(checksums_path) if checksums_path is not None else root / "checksums.sha256"
    checks = [
        _check_manifest(root, manifest_path, require_frozen=require_frozen),
        _check_checksums(root, checksum_path),
        _check_manifest_checksum_alignment(root, manifest_path, checksum_path),
    ]
    return _summary(root, checks, require_frozen)


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path {path} is outside artifact root {root}") from exc


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_file_entries(root: Path, *, excluded_paths: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = _relative_path(root, path)
        if relative in excluded_paths:
            continue
        stat = path.stat()
        entries.append(
            {
                "path": relative,
                "byte_size": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _run_git(args: tuple[str, ...], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_state(repo_root: Path) -> dict[str, Any]:
    status = _run_git(("status", "--porcelain"), repo_root)
    return {
        "commit": _run_git(("rev-parse", "HEAD"), repo_root),
        "short_commit": _run_git(("rev-parse", "--short=12", "HEAD"), repo_root),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _check_manifest(root: Path, manifest_path: Path, *, require_frozen: bool) -> PublicationArtifactCheck:
    if not manifest_path.is_file():
        return PublicationArtifactCheck("manifest", False, ("missing manifest.json",))
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest JSON must contain an object")
    except Exception as exc:
        return PublicationArtifactCheck("manifest", False, (f"invalid manifest.json: {exc}",))
    if payload.get("artifact_schema_version") != PUBLICATION_ARTIFACT_SCHEMA_VERSION:
        errors.append("unsupported artifact_schema_version")
    if not payload.get("run_id"):
        errors.append("manifest missing run_id")
    if payload.get("evidence_status") == "frozen":
        git = payload.get("git") or {}
        if git.get("dirty") is not False:
            errors.append("frozen manifest requires a clean git state")
    if require_frozen and payload.get("evidence_status") != "frozen":
        errors.append("expected frozen manifest evidence_status")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest must list artifact files")
    else:
        seen: set[str] = set()
        for index, entry in enumerate(files):
            prefix = f"files[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{prefix} must be an object")
                continue
            relative = entry.get("path")
            if not isinstance(relative, str) or not relative:
                errors.append(f"{prefix} missing path")
                continue
            if relative in seen:
                errors.append(f"duplicate manifest file path: {relative}")
            seen.add(relative)
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                errors.append(f"{prefix} path escapes artifact root")
                continue
            path = root / relative
            if not path.is_file():
                errors.append(f"{prefix} missing file: {relative}")
                continue
            if entry.get("byte_size") != path.stat().st_size:
                errors.append(f"{prefix} byte_size mismatch: {relative}")
            if entry.get("sha256") != _sha256_file(path):
                errors.append(f"{prefix} sha256 mismatch: {relative}")
    return PublicationArtifactCheck(
        "manifest",
        not errors,
        tuple(errors),
        tuple(warnings),
        metadata={"path": str(manifest_path), "file_count": len(files) if isinstance(files, list) else None},
    )


def _check_checksums(root: Path, checksums_path: Path) -> PublicationArtifactCheck:
    if not checksums_path.is_file():
        return PublicationArtifactCheck("checksums", False, ("missing checksums.sha256",))
    errors: list[str] = []
    entries = 0
    for line_number, line in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entries += 1
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: expected '<sha256>  <relative-path>'")
            continue
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            errors.append(f"line {line_number}: invalid sha256")
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"line {line_number}: path escapes artifact root")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"line {line_number}: missing file {relative}")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            errors.append(f"line {line_number}: checksum mismatch for {relative}")
    return PublicationArtifactCheck(
        "checksums",
        not errors,
        tuple(errors),
        metadata={"path": str(checksums_path), "entry_count": entries},
    )


def _check_manifest_checksum_alignment(
    root: Path,
    manifest_path: Path,
    checksums_path: Path,
) -> PublicationArtifactCheck:
    if not manifest_path.is_file() or not checksums_path.is_file():
        return PublicationArtifactCheck("manifest_checksums", False, ("missing manifest or checksums file",))
    errors: list[str] = []
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest_payload.get("files")
        if not isinstance(files, list):
            raise ValueError("manifest files must be a list")
        manifest_entries = {
            str(entry["path"]): str(entry["sha256"])
            for entry in files
            if isinstance(entry, dict) and "path" in entry and "sha256" in entry
        }
    except Exception as exc:
        return PublicationArtifactCheck(
            "manifest_checksums",
            False,
            (f"cannot compare manifest and checksums: {exc}",),
        )
    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed checksum entry")
            continue
        checksum_entries[relative] = checksum
    missing = sorted(set(manifest_entries) - set(checksum_entries))
    extra = sorted(set(checksum_entries) - set(manifest_entries))
    mismatched = sorted(
        path
        for path in set(manifest_entries) & set(checksum_entries)
        if manifest_entries[path] != checksum_entries[path]
    )
    if missing:
        errors.append(f"checksums missing manifest files: {missing}")
    if extra:
        errors.append(f"checksums list files absent from manifest: {extra}")
    if mismatched:
        errors.append(f"manifest/checksum hash mismatch: {mismatched}")
    return PublicationArtifactCheck(
        "manifest_checksums",
        not errors,
        tuple(errors),
        metadata={
            "artifact_root": str(root),
            "manifest_file_count": len(manifest_entries),
            "checksum_entry_count": len(checksum_entries),
        },
    )


def _summary(
    root: Path,
    checks: list[PublicationArtifactCheck],
    require_frozen: bool,
) -> dict[str, Any]:
    error_count = sum(len(check.errors) for check in checks)
    warning_count = sum(len(check.warnings) for check in checks)
    return {
        "artifact_root": str(root),
        "ok": error_count == 0 and all(check.ok for check in checks),
        "require_frozen": require_frozen,
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "errors": list(check.errors),
                "warnings": list(check.warnings),
                "metadata": check.metadata or {},
            }
            for check in checks
        ],
    }
