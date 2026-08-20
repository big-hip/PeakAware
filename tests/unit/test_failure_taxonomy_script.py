from scripts.summarize_failure_taxonomy import summarize_failure_taxonomy


def test_failure_taxonomy_classifies_budget_and_topk_failures():
    payload = summarize_failure_taxonomy(
        [
            {"status": "ok"},
            {
                "status": "failed",
                "task_name": "gpt2",
                "error_type": "InfeasibleBudgetError",
                "error_message": "fixed lower bound 100 bytes exceeds budget 50 bytes",
            },
            {
                "status": "failed",
                "task_name": "bert",
                "error_type": "InfeasibleBudgetError",
                "error_message": "no Top-K candidate passed dry-run and measurement:",
            },
        ]
    )

    assert payload["record_count"] == 3
    assert payload["failed_record_count"] == 2
    assert payload["category_counts"]["fixed_frontier_budget_infeasible"] == 1
    assert payload["category_counts"]["topk_no_candidate_passed_measurement"] == 1
