from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_costmodel_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_costmodel_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_family_separates_profile_analytical_and_fallback_routes():
    assert MODULE._source_family("profile_db_exact") == "profile_db_exact"
    assert MODULE._source_family("profile_db_interpolated") == "profile_db_interpolated"
    assert MODULE._source_family("structural_zero") == "structural_zero"
    assert MODULE._source_family("metadata_view_zero") == "metadata_zero"
    assert MODULE._source_family("analytical:sdpa_fused") == "analytical"
    assert (
        MODULE._source_family("legacy_adapter:zhanlu_analytical:Addmm")
        == "analytical"
    )
    assert MODULE._source_family("legacy_adapter:static_fallback") == "static_fallback"
    assert MODULE._source_family("roofline_fallback") == "roofline_fallback"


def test_aggregate_reports_fallback_by_count_and_estimated_time():
    aggregate = MODULE._aggregate(
        [
            {
                "ir_op_count": 10,
                "fallback_op_count": 2,
                "estimated_total_us": 100.0,
                "fallback_estimated_time_us": 30.0,
                "unresolved_op_count": 0,
            },
            {
                "ir_op_count": 5,
                "fallback_op_count": 1,
                "estimated_total_us": 50.0,
                "fallback_estimated_time_us": 5.0,
                "unresolved_op_count": 1,
            },
        ]
    )

    assert aggregate["task_count"] == 2
    assert aggregate["ir_op_count"] == 15
    assert aggregate["fallback_op_count"] == 3
    assert aggregate["fallback_op_rate"] == 0.2
    assert aggregate["fallback_estimated_time_rate"] == 35.0 / 150.0
    assert aggregate["unresolved_op_count"] == 1
