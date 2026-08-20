from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from peakaware.cost.base import OpCost, OpSignature, current_hardware_version, current_software_version


_SCHEMA_VERSION = 2
_SCALABLE_ELEMENTWISE_TOKENS = (
    "add",
    "bitwise",
    "clone",
    "copy",
    "detach",
    "div",
    "dropout",
    "exp",
    "gelu",
    "log",
    "masked_fill",
    "mul",
    "neg",
    "pow",
    "relu",
    "sigmoid",
    "silu",
    "softmax",
    "sqrt",
    "sub",
    "tanh",
    "threshold",
    "where",
)
_MATMUL_TOKENS = ("addmm", "bmm", "linear", "matmul", "mm")


@dataclass(frozen=True)
class ProfileRecord:
    signature_hash: str
    sample_count: int
    p50_us: float
    p90_us: float
    mean_us: float
    workspace_bytes: int
    source: str = "profile_db"


def _shape_payload(shapes: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [[int(dimension) for dimension in shape] for shape in shapes]


def _signature_payload(signature: OpSignature) -> dict[str, object]:
    """Return the reusable kernel signature, excluding the unique FX node name."""

    return {
        "target": str(signature.target),
        "input_bytes": int(signature.input_bytes),
        "output_bytes": int(signature.output_bytes),
        "dtype": str(signature.dtype),
        "input_shapes": _shape_payload(signature.input_shapes),
        "output_shapes": _shape_payload(signature.output_shapes),
        "input_dtypes": [str(dtype) for dtype in signature.input_dtypes],
        "output_dtypes": [str(dtype) for dtype in signature.output_dtypes],
    }


def profile_signature_hash(signature: OpSignature) -> str:
    payload = _signature_payload(signature)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_shapes(payload: str) -> tuple[tuple[int, ...], ...]:
    try:
        values = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(
        tuple(int(dimension) for dimension in shape)
        for shape in values
        if isinstance(shape, list)
    )


def _decode_dtypes(payload: str) -> tuple[str, ...]:
    try:
        values = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= max(int(dimension), 0)
    return result


def _matmul_work(signature: OpSignature) -> int | None:
    shapes = [shape for shape in signature.input_shapes if len(shape) >= 2]
    if len(shapes) < 2:
        return None
    left, right = shapes[-2:]
    m, k = int(left[-2]), int(left[-1])
    right_k, n = int(right[-2]), int(right[-1])
    if k != right_k:
        if k == int(right[-1]):
            right_k, n = int(right[-1]), int(right[-2])
        elif int(left[-2]) == right_k:
            m, k = int(left[-1]), int(left[-2])
    if min(m, k, right_k, n) <= 0 or k != right_k:
        return None
    batch_shape = left[:-2] if len(left) >= len(right) else right[:-2]
    return 2 * max(_numel(batch_shape), 1) * m * n * k


def _attention_work(signature: OpSignature) -> int | None:
    shapes = [shape for shape in signature.input_shapes if len(shape) == 4]
    target = str(signature.target).lower()
    if "backward" in target:
        if len(shapes) < 4:
            return None
        query, key, value = shapes[1:4]
        backward_multiplier = 5
    else:
        if len(shapes) < 3:
            return None
        query, key, value = shapes[:3]
        backward_multiplier = 1
    batch, heads, query_length, head_dim = (int(value) for value in query)
    key_length = int(key[-2])
    value_dim = int(value[-1])
    if min(batch, heads, query_length, key_length, head_dim, value_dim) <= 0:
        return None
    return backward_multiplier * batch * heads * query_length * key_length * (head_dim + value_dim)


def _interpolation_scale(source: OpSignature, target: OpSignature) -> float | None:
    target_name = str(target.target).lower()
    source_name = str(source.target).lower()
    if source_name != target_name:
        return None
    if "scaled_dot_product" in target_name or "attention" in target_name:
        source_work = _attention_work(source)
        target_work = _attention_work(target)
    elif any(token in target_name for token in _MATMUL_TOKENS):
        source_work = _matmul_work(source)
        target_work = _matmul_work(target)
    elif any(token in target_name for token in _SCALABLE_ELEMENTWISE_TOKENS):
        source_work = max(int(source.input_bytes + source.output_bytes), 1)
        target_work = max(int(target.input_bytes + target.output_bytes), 1)
    else:
        # Convolution, normalization, reduction and fused kernels need either
        # an exact profile or an operator-specific model. Linear byte scaling
        # is not a defensible interpolation rule for them.
        return None
    if source_work is None or target_work is None or source_work <= 0:
        return None
    return float(target_work) / float(source_work)


class ProfileDB:
    def __init__(
        self,
        path: str | Path,
        *,
        hardware_version: str | None = None,
        software_version: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hardware_version = hardware_version or current_hardware_version()
        self.software_version = software_version or current_software_version()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection, table_name: str = "op_profile") -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                signature_hash TEXT NOT NULL,
                hardware_version TEXT NOT NULL,
                software_version TEXT NOT NULL,
                target TEXT NOT NULL,
                dtype TEXT NOT NULL,
                total_bytes INTEGER NOT NULL,
                input_shapes_json TEXT NOT NULL,
                output_shapes_json TEXT NOT NULL,
                input_dtypes_json TEXT NOT NULL,
                output_dtypes_json TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                p50_us REAL NOT NULL,
                p90_us REAL NOT NULL,
                mean_us REAL NOT NULL,
                workspace_bytes INTEGER NOT NULL,
                PRIMARY KEY(signature_hash, hardware_version, software_version)
            )
            """
        )

    def _init_schema(self) -> None:
        required_columns = {
            "signature_hash",
            "hardware_version",
            "software_version",
            "target",
            "dtype",
            "total_bytes",
            "input_shapes_json",
            "output_shapes_json",
            "input_dtypes_json",
            "output_dtypes_json",
            "sample_count",
            "p50_us",
            "p90_us",
            "mean_us",
            "workspace_bytes",
        }
        with self._connect() as conn:
            existing = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(op_profile)").fetchall()
            }
            if not existing:
                self._create_schema(conn)
            elif not required_columns.issubset(existing):
                # Preserve legacy records for auditability, but mark them as
                # environment-unscoped so they can never silently satisfy a
                # lookup on the current GPU/software stack.
                conn.execute("DROP TABLE IF EXISTS op_profile_v2_migration")
                self._create_schema(conn, "op_profile_v2_migration")
                legacy_columns = existing
                if {
                    "signature_hash",
                    "target",
                    "dtype",
                    "total_bytes",
                    "sample_count",
                    "p50_us",
                    "p90_us",
                    "mean_us",
                    "workspace_bytes",
                }.issubset(legacy_columns):
                    conn.execute(
                        """
                        INSERT INTO op_profile_v2_migration(
                            signature_hash, hardware_version, software_version,
                            target, dtype, total_bytes,
                            input_shapes_json, output_shapes_json,
                            input_dtypes_json, output_dtypes_json,
                            sample_count, p50_us, p90_us, mean_us, workspace_bytes
                        )
                        SELECT signature_hash, 'legacy:unscoped', 'legacy:unscoped',
                               target, dtype, total_bytes, '[]', '[]', '[]', '[]',
                               sample_count, p50_us, p90_us, mean_us, workspace_bytes
                        FROM op_profile
                        """
                    )
                conn.execute("DROP TABLE op_profile")
                conn.execute("ALTER TABLE op_profile_v2_migration RENAME TO op_profile")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS op_profile_environment_target "
                "ON op_profile(hardware_version, software_version, target, dtype)"
            )
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _row_signature(row: tuple[object, ...]) -> OpSignature:
        (
            target,
            dtype,
            total_bytes,
            input_shapes_json,
            output_shapes_json,
            input_dtypes_json,
            output_dtypes_json,
        ) = row
        input_shapes = _decode_shapes(str(input_shapes_json))
        output_shapes = _decode_shapes(str(output_shapes_json))
        input_dtypes = _decode_dtypes(str(input_dtypes_json))
        output_dtypes = _decode_dtypes(str(output_dtypes_json))
        # Byte totals are only a compatibility fallback for migrated or
        # externally generated rows. New rows retain exact input/output bytes
        # in their signature hash and shape payload.
        return OpSignature(
            op_name="profile_source",
            target=str(target),
            input_bytes=int(total_bytes),
            output_bytes=0,
            dtype=str(dtype),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
        )

    def lookup_exact(self, signature: OpSignature) -> OpCost | None:
        signature_hash = profile_signature_hash(signature)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT sample_count, p50_us, p90_us, mean_us, workspace_bytes
                FROM op_profile
                WHERE signature_hash = ? AND hardware_version = ? AND software_version = ?
                """,
                (signature_hash, self.hardware_version, self.software_version),
            ).fetchone()
        if row is None:
            return None
        sample_count, p50_us, p90_us, mean_us, workspace_bytes = row
        del p90_us, mean_us
        confidence = 1.0 if int(sample_count) >= 10 else 0.8
        return OpCost(
            float(p50_us),
            int(workspace_bytes),
            "profile_db",
            confidence,
            hardware_version=self.hardware_version,
            software_version=self.software_version,
        )

    def lookup_nearest(self, signature: OpSignature) -> OpCost | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sample_count, p50_us, workspace_bytes,
                       target, dtype, total_bytes,
                       input_shapes_json, output_shapes_json,
                       input_dtypes_json, output_dtypes_json
                FROM op_profile
                WHERE target = ? AND dtype = ?
                  AND hardware_version = ? AND software_version = ?
                  AND signature_hash != ?
                """,
                (
                    signature.target,
                    signature.dtype,
                    self.hardware_version,
                    self.software_version,
                    profile_signature_hash(signature),
                ),
            ).fetchall()
        best: tuple[float, tuple[object, ...], float] | None = None
        for row in rows:
            source = self._row_signature(tuple(row[3:]))
            scale = _interpolation_scale(source, signature)
            if scale is None or scale <= 0:
                continue
            distance = abs(math.log2(scale))
            if best is None or distance < best[0]:
                best = (distance, row, scale)
        if best is None:
            return None
        distance, row, scale = best
        sample_count, p50_us, workspace_bytes = row[:3]
        base_confidence = 0.8 if int(sample_count) >= 10 else 0.6
        confidence = max(0.3, base_confidence / (1.0 + distance))
        return OpCost(
            float(p50_us) * scale,
            max(0, int(round(int(workspace_bytes) * scale))),
            "profile_db_interpolated",
            confidence,
            hardware_version=self.hardware_version,
            software_version=self.software_version,
        )

    def upsert_profile(self, signature: OpSignature, record: ProfileRecord) -> None:
        signature_hash = profile_signature_hash(signature)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO op_profile(
                    signature_hash, hardware_version, software_version,
                    target, dtype, total_bytes,
                    input_shapes_json, output_shapes_json,
                    input_dtypes_json, output_dtypes_json,
                    sample_count, p50_us, p90_us, mean_us, workspace_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature_hash, hardware_version, software_version) DO UPDATE SET
                    target = excluded.target,
                    dtype = excluded.dtype,
                    total_bytes = excluded.total_bytes,
                    input_shapes_json = excluded.input_shapes_json,
                    output_shapes_json = excluded.output_shapes_json,
                    input_dtypes_json = excluded.input_dtypes_json,
                    output_dtypes_json = excluded.output_dtypes_json,
                    sample_count = excluded.sample_count,
                    p50_us = excluded.p50_us,
                    p90_us = excluded.p90_us,
                    mean_us = excluded.mean_us,
                    workspace_bytes = excluded.workspace_bytes
                """,
                (
                    signature_hash,
                    self.hardware_version,
                    self.software_version,
                    signature.target,
                    signature.dtype,
                    signature.input_bytes + signature.output_bytes,
                    json.dumps(_shape_payload(signature.input_shapes), separators=(",", ":")),
                    json.dumps(_shape_payload(signature.output_shapes), separators=(",", ":")),
                    json.dumps(list(signature.input_dtypes), separators=(",", ":")),
                    json.dumps(list(signature.output_dtypes), separators=(",", ":")),
                    record.sample_count,
                    record.p50_us,
                    record.p90_us,
                    record.mean_us,
                    record.workspace_bytes,
                ),
            )

    def invalidate_by_environment(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM op_profile WHERE hardware_version = ? AND software_version = ?",
                (self.hardware_version, self.software_version),
            )


class ExactProfileProvider:
    source = "profile_db_exact"
    cache_safe = True

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
    cache_safe = True

    def __init__(self, db: ProfileDB) -> None:
        self.db = db

    def supports(self, signature: OpSignature) -> bool:
        return self.db.lookup_nearest(signature) is not None

    def estimate(self, signature: OpSignature) -> OpCost | None:
        return self.db.lookup_nearest(signature)
