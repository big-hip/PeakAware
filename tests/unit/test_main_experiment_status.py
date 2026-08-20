from scripts.evaluate_main_experiment_status import evaluate_main_experiment_status
from scripts.summarize_selected_regret import summarize_selected_regret


def test_main_experiment_status_reports_partial_evidence_blockers():
    records = [
        {
            "status": "ok",
            "variant_name": "default",
            "task_name": "tiny",
            "microbatch_size": 1,
            "budget_bytes": 100,
            "selected_plan_id": "peakaware",
            "measured_peak_bytes": 90,
            "measurement_repeats": 5,
            "measurement_warmup_steps": 0,
            "measured_plan_results": (
                {
                    "plan_id": "all_save",
                    "measured_peak_bytes": 120,
                    "measured_step_us": 10.0,
                    "strategy_provenance": {"source": "all_save_baseline"},
                },
                {
                    "plan_id": "torch_min_cut",
                    "measured_peak_bytes": 95,
                    "measured_step_us": 8.0,
                    "strategy_provenance": {"source": "pytorch_min_cut_proxy"},
                },
                {
                    "plan_id": "peakaware",
                    "measured_peak_bytes": 90,
                    "measured_step_us": 7.0,
                    "strategy_provenance": {"source": "greedy_bytes_per_cost"},
                },
            ),
        }
    ]

    payload = evaluate_main_experiment_status(
        records,
        selected_regret=summarize_selected_regret(records),
    )

    assert payload["main_question_status"] == "partial_evidence_needs_rerun"
    assert "baseline_provenance_proxy_only" in payload["blockers"]
    assert "missing_continuous_actual_vs_simulated_timeline" in payload["blockers"]
    assert "measurement_protocol_not_uniformly_steady_state" in payload["blockers"]
    assert "no_budget_violation_among_ok_records" in payload["strengths"]

    with_taxonomy = evaluate_main_experiment_status(
        records,
        selected_regret=summarize_selected_regret(records),
        failure_taxonomy={"failed_record_count": 0, "category_counts": {}, "error_type_counts": {}},
    )

    assert "failed_records_require_taxonomy" not in with_taxonomy["blockers"]
