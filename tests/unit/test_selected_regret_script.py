from scripts.summarize_selected_regret import summarize_selected_regret


def test_selected_regret_script_skips_time_regret_for_infeasible_selection():
    payload = summarize_selected_regret(
        [
            {
                "status": "ok",
                "variant_name": "default",
                "task_name": "tiny",
                "microbatch_size": 1,
                "budget_bytes": 100,
                "selected_plan_id": "fast_too_large",
                "measured_plan_results": (
                    {
                        "plan_id": "fast_too_large",
                        "measured_peak_bytes": 120,
                        "measured_step_us": 5.0,
                    },
                    {
                        "plan_id": "slow_feasible",
                        "measured_peak_bytes": 90,
                        "measured_step_us": 10.0,
                    },
                ),
            }
        ]
    )

    row = payload["rows"][0]

    assert row["selected_measured_feasible"] is False
    assert row["best_feasible_plan_id"] == "slow_feasible"
    assert row["selected_regret_us"] is None
    assert payload["selected_measured_feasible_rate"] == 0.0
