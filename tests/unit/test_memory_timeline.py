from peakaware.memory.timeline import (
    build_actual_memory_timeline,
    build_simulated_memory_timeline,
    fit_memory_timelines,
    merge_timeline_rows,
)
from scripts.generate_memory_timeline_fit import (
    _event_alignment_metadata,
    _gpu_util_rows,
    _render_event_gpu_svg,
    _simulated_event_rows,
)


def test_memory_timeline_fit_reports_zero_error_for_identical_curves():
    actual = build_actual_memory_timeline(
        {
            "fw_peak_bytes": 100,
            "after_fw_allocated_bytes": 60,
            "bw_peak_bytes": 140,
            "optimizer_peak_bytes": 130,
            "overall_peak_bytes": 140,
        }
    )
    simulated = build_simulated_memory_timeline(
        fw_peak_bytes=100,
        after_fw_retained_bytes=60,
        bw_peak_bytes=140,
        optimizer_peak_bytes=130,
        estimated_peak_bytes=140,
    )

    fit = fit_memory_timelines(actual, simulated)

    assert fit["point_count"] == 6
    assert fit["mae_bytes"] == 0
    assert fit["rmse_bytes"] == 0
    assert fit["max_abs_error_bytes"] == 0
    assert fit["peak_error_bytes"] == 0
    assert fit["normalized_rmse"] == 0
    assert fit["offset_fitted_simulated_bytes"] == 0
    assert fit["offset_fitted_rmse_bytes"] == 0
    assert fit["pearson_correlation"] == 1.0


def test_merge_timeline_rows_preserves_record_metadata_and_errors():
    actual = build_actual_memory_timeline(
        {
            "fw_peak_bytes": 120,
            "after_fw_allocated_bytes": 70,
            "bw_peak_bytes": 160,
            "optimizer_peak_bytes": 150,
            "overall_peak_bytes": 160,
        }
    )
    simulated = build_simulated_memory_timeline(
        fw_peak_bytes=100,
        after_fw_retained_bytes=60,
        bw_peak_bytes=140,
        optimizer_peak_bytes=130,
        estimated_peak_bytes=140,
    )

    rows = merge_timeline_rows(
        record={
            "task_name": "tiny",
            "variant_name": "default",
            "microbatch_size": 2,
            "budget_bytes": 1024,
            "selected_plan_id": "plan",
            "matrix_pass_index": 0,
        },
        actual=actual,
        simulated=simulated,
    )

    assert rows[1]["phase"] == "fw_peak"
    assert rows[1]["task_name"] == "tiny"
    assert rows[1]["actual_bytes"] == 120
    assert rows[1]["simulated_bytes"] == 100
    assert rows[1]["offset_fitted_simulated_bytes"] == 100
    assert rows[1]["error_bytes"] == 20


def test_memory_timeline_fit_reports_offset_corrected_error():
    actual = build_actual_memory_timeline(
        {
            "fw_peak_bytes": 110,
            "after_fw_allocated_bytes": 120,
            "bw_peak_bytes": 130,
            "optimizer_peak_bytes": 140,
            "overall_peak_bytes": 140,
        }
    )
    simulated = build_simulated_memory_timeline(
        fw_peak_bytes=10,
        after_fw_retained_bytes=20,
        bw_peak_bytes=30,
        optimizer_peak_bytes=40,
        estimated_peak_bytes=40,
    )

    fit = fit_memory_timelines(actual, simulated)
    rows = merge_timeline_rows(
        record={},
        actual=actual,
        simulated=simulated,
        offset_fitted_simulated_bytes=fit["offset_fitted_simulated_bytes"],
    )

    assert fit["offset_fitted_simulated_bytes"] > 0
    assert fit["offset_fitted_rmse_bytes"] < fit["rmse_bytes"]
    assert rows[1]["offset_fitted_simulated_bytes"] > rows[1]["simulated_bytes"]


def test_merge_timeline_rows_can_align_simulated_time_to_measured_total():
    actual = build_actual_memory_timeline(
        {
            "fw_peak_bytes": 100,
            "after_fw_allocated_bytes": 80,
            "bw_peak_bytes": 130,
            "optimizer_peak_bytes": 120,
            "overall_peak_bytes": 130,
        }
    )
    simulated = build_simulated_memory_timeline(
        fw_peak_bytes=90,
        after_fw_retained_bytes=70,
        bw_peak_bytes=110,
        optimizer_peak_bytes=100,
        estimated_peak_bytes=110,
    )

    rows = merge_timeline_rows(
        record={
            "measured_step_us": 5000.0,
            "measured_plan_results": ({"plan_id": "p", "estimated_step_us": 100.0},),
            "selected_plan_id": "p",
            "time_align": "matched-total",
        },
        actual=actual,
        simulated=simulated,
        x_axis="time",
    )

    assert rows[-1]["simulated_x"] == 5000.0
    assert rows[1]["simulated_x"] == 1000.0


def test_simulated_event_rows_align_each_phase_and_preserve_components():
    record = {
        "task_name": "tiny",
        "variant_name": "default",
        "microbatch_size": 1,
        "budget_bytes": 1024,
        "selected_plan_id": "p",
        "measured_fw_us": 100.0,
        "measured_bw_us": 300.0,
        "measured_optimizer_us": 600.0,
        "selected_simulated_memory_event_trace": (
            {"phase": "start", "event": "step_start", "time_us": 0.0, "bytes": 0},
            {
                "phase": "fw",
                "event": "op_end",
                "time_us": 10.0,
                "bytes": 10,
                "payload_bytes": 4,
                "workspace_bytes": 2,
                "live_storage_count": 1,
            },
            {"phase": "after_fw", "event": "phase_boundary", "time_us": 20.0, "bytes": 8},
            {"phase": "bw", "event": "op_end", "time_us": 25.0, "bytes": 11},
            {"phase": "bw", "event": "op_end", "time_us": 30.0, "bytes": 12},
            {"phase": "optimizer", "event": "phase_start", "time_us": 30.0, "bytes": 6},
            {"phase": "overall", "event": "step_end", "time_us": 50.0, "bytes": 6},
        ),
    }

    rows = _simulated_event_rows(record, sampled_rows=[])

    assert rows[1]["time_us"] == 50.0
    assert rows[2]["time_us"] == 100.0
    assert rows[3]["time_us"] == 250.0
    assert rows[4]["time_us"] == 400.0
    assert rows[5]["time_us"] == 400.0
    assert rows[6]["time_us"] == 1000.0
    assert rows[1]["payload_bytes"] == 4
    assert rows[1]["workspace_bytes"] == 2
    assert rows[1]["live_storage_count"] == 1

    metadata = _event_alignment_metadata(record, rows)
    assert metadata["time_alignment_kind"] == "phasewise_costmodel_to_measured"
    assert metadata["memory_model_kind"] == "aten_ir_liveness_l2_components"
    assert metadata["measured_phase_us"] == {"fw": 100.0, "bw": 300.0, "optimizer": 600.0}
    assert metadata["aligned_phase_span_us"]["fw"] == 100.0
    assert metadata["raw_costmodel_phase_span_us"]["optimizer"] == 20.0
    assert "payload_bytes" in metadata["component_fields"]


def test_gpu_util_rows_feed_combined_event_svg():
    record = {
        "task_name": "tiny",
        "variant_name": "default",
        "microbatch_size": 1,
        "budget_bytes": 1024,
        "selected_plan_id": "p",
        "selected_gpu_util_trace": (
            {
                "time_us": 0.0,
                "event": "sample",
                "device_index": 0,
                "gpu_util_percent": 10.0,
                "memory_util_percent": 20.0,
                "memory_used_mib": 100.0,
                "power_w": 80.0,
                "source": "nvidia-smi",
            },
            {
                "time_us": 50.0,
                "event": "sample",
                "device_index": 0,
                "gpu_util_percent": 90.0,
                "memory_util_percent": 70.0,
                "memory_used_mib": 200.0,
                "power_w": 180.0,
                "source": "nvidia-smi",
            },
        ),
    }
    sampled_rows = [
        {"time_us": 0.0, "allocated_bytes": 10, "allocator_peak_bytes": 10},
        {"time_us": 50.0, "allocated_bytes": 30, "allocator_peak_bytes": 40},
    ]
    simulated_rows = [
        {"time_us": 0.0, "bytes": 12, "event": "phase_start"},
        {"time_us": 50.0, "bytes": 35, "event": "op_end"},
    ]

    gpu_rows = _gpu_util_rows(record)
    svg = _render_event_gpu_svg(record, sampled_rows, simulated_rows, gpu_rows, {"rmse_bytes": 1})

    assert gpu_rows[1]["gpu_util_percent"] == 90.0
    assert "GPU util" in svg
    assert "Mem util" in svg
    assert "memory and GPU utilization" in svg
