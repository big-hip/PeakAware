from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from peakaware.cost.base import OpCost, OpSignature


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
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS op_profile (
                    signature_hash TEXT PRIMARY KEY,
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
        return OpCost(float(p50_us), int(workspace_bytes), "profile_db", confidence)

    def lookup_nearest(self, signature: OpSignature) -> OpCost | None:
        del signature
        return None

    def upsert_profile(self, signature: OpSignature, record: ProfileRecord) -> None:
        signature_hash = _signature_hash(signature)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO op_profile(signature_hash, sample_count, p50_us, p90_us, mean_us, workspace_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature_hash) DO UPDATE SET
                    sample_count = excluded.sample_count,
                    p50_us = excluded.p50_us,
                    p90_us = excluded.p90_us,
                    mean_us = excluded.mean_us,
                    workspace_bytes = excluded.workspace_bytes
                """,
                (
                    signature_hash,
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
