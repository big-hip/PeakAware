from scripts.audit_timeline_readiness import audit_records


def test_gpu_readiness_requires_real_samples_not_unavailable_summary():
    record = {
        "status": "ok",
        "task_name": "synthetic",
        "variant_name": "default",
        "selected_plan_id": "peakaware",
        "budget_bytes": 100,
        "selected_actual_overall_sampled_memory_trace": [{"time_us": 0.0, "allocated_bytes": 0}],
        "selected_lowered_fx_l2_simulated_memory_event_trace": [{"time_us": 0.0, "bytes": 0}],
        "selected_gpu_compute_summary": {
            "status": "unavailable",
            "sample_count": 0,
            "unavailable_reason": "nvidia_smi_unavailable",
        },
    }

    unavailable = audit_records([record], require_gpu_util=True)

    assert unavailable["timeline_ready"] is True
    assert unavailable["gpu_ready"] is False

    ready = audit_records(
        [
            {
                **record,
                "selected_gpu_util_trace": [{"time_us": 0.0, "gpu_util_percent": 80.0}],
                "selected_gpu_compute_summary": {"status": "ok", "sample_count": 1},
            }
        ],
        require_gpu_util=True,
    )

    assert ready["gpu_ready"] is True
    assert ready["gpu_ready_record_count"] == 1
