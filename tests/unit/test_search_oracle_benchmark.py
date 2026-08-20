from __future__ import annotations

from scripts.benchmark_search_oracle import benchmark_search_oracle


def test_search_oracle_benchmark_reports_exact_comparison_without_measurements() -> None:
    payload = benchmark_search_oracle(
        seeds=1,
        candidate_count=6,
        beam_widths=(2,),
        budget_quantiles=(0.5,),
    )

    assert payload["evidence_scope"] == "synthetic_small_graph_simulation_oracle"
    assert payload["candidate_measurements_used"] == 0
    assert len(payload["rows"]) == 7
    assert all(row["case_count"] == 1 for row in payload["rows"])
    assert all(row["mean_exact_plan_count"] == 64 for row in payload["rows"])
