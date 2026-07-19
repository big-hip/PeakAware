from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from peakaware.cost.base import OpCost, OpSignature, current_hardware_version, current_software_version


@dataclass(frozen=True)
class ProfileRecord:
    signature_hash: str
    sample_count: int
    p50_us: float
    p90_us: float
    mean_us: float
    workspace_bytes: int
    source: str = "profile_db"


def _signature_hash(signature: OpSignature) -> str:
    import hashlib
    import json

    payload = {
        "op_name": signature.op_name,
        "target": signature.target,
        "input_bytes": signature.input_bytes,
        "output_bytes": signature.output_bytes,
        "dtype": signature.dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class ProfileDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.hardware_version = current_hardware_version()
        self.software_version = current_software_version()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS op_profile (
                    signature_hash TEXT PRIMARY KEY,
                    target TEXT NOT NULL DEFAULT '',
                    dtype TEXT NOT NULL DEFAULT '',
                    total_bytes INTEGER NOT NULL DEFAULT 0,
                    sample_count INTEGER NOT NULL,
                    p50_us REAL NOT NULL,
                    p90_us REAL NOT NULL,
                    mean_us REAL NOT NULL,
                    workspace_bytes INTEGER NOT NULL
                )
                """
            )

    def lookup_exact(self, signature: OpSignature) -> OpCost | None:
        signature_hash = _signature_hash(signature)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sample_count, p50_us, p90_us, mean_us, workspace_bytes FROM op_profile WHERE signature_hash = ?",
                (signature_hash,),
            ).fetchone()
        if row is None:
            return None
        sample_count, p50_us, p90_us, mean_us, workspace_bytes = row
        del p90_us, mean_us
        confidence = 1.0 if sample_count >= 10 else 0.8
        return OpCost(
            float(p50_us),
            int(workspace_bytes),
            "profile_db",
            confidence,
            hardware_version=self.hardware_version,
            software_version=self.software_version,
        )

    def lookup_nearest(self, signature: OpSignature) -> OpCost | None:
        target_bytes = max(signature.input_bytes + signature.output_bytes, 1)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT sample_count, p50_us, workspace_bytes, total_bytes
                FROM op_profile
                WHERE target = ? AND dtype = ? AND signature_hash != ?
                ORDER BY ABS(total_bytes - ?) ASC
                LIMIT 1
                """,
                (signature.target, signature.dtype, _signature_hash(signature), target_bytes),
            ).fetchone()
        if row is None:
            return None
        sample_count, p50_us, workspace_bytes, source_bytes = row
        scale = target_bytes / max(int(source_bytes), 1)
        confidence = 0.8 if sample_count >= 10 else 0.6
        return OpCost(
            float(p50_us) * scale,
            int(workspace_bytes * scale),
            "profile_db_interpolated",
            confidence,
            hardware_version=self.hardware_version,
            software_version=self.software_version,
        )

    def upsert_profile(self, signature: OpSignature, record: ProfileRecord) -> None:
        signature_hash = _signature_hash(signature)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO op_profile(signature_hash, target, dtype, total_bytes, sample_count, p50_us, p90_us, mean_us, workspace_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature_hash) DO UPDATE SET
                    target = excluded.target,
                    dtype = excluded.dtype,
                    total_bytes = excluded.total_bytes,
                    sample_count = excluded.sample_count,
                    p50_us = excluded.p50_us,
                    p90_us = excluded.p90_us,
                    mean_us = excluded.mean_us,
                    workspace_bytes = excluded.workspace_bytes
                """,
                (
                    signature_hash,
                    signature.target,
                    signature.dtype,
                    signature.input_bytes + signature.output_bytes,
                    record.sample_count,
                    record.p50_us,
                    record.p90_us,
                    record.mean_us,
                    record.workspace_bytes,
                ),
            )

    def invalidate_by_environment(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM op_profile")


class ExactProfileProvider:
    source = "profile_db_exact"

    def __init__(self, db: ProfileDB) -> None:
        self.db = db

    def supports(self, signature: OpSignature) -> bool:
        return self.db.lookup_exact(signature) is not None

    def estimate(self, signature: OpSignature) -> OpCost | None:
        cost = self.db.lookup_exact(signature)
        if cost is None:
            return None
        return OpCost(
            cost.estimated_us,
            cost.memory_bytes,
            self.source,
            cost.confidence,
            hardware_version=cost.hardware_version,
            software_version=cost.software_version,
        )


class InterpolatedProfileProvider:
    source = "profile_db_interpolated"

    def __init__(self, db: ProfileDB) -> None:
        self.db = db

    def supports(self, signature: OpSignature) -> bool:
        return self.db.lookup_nearest(signature) is not None

    def estimate(self, signature: OpSignature) -> OpCost | None:
        return self.db.lookup_nearest(signature)
