from peakaware.memory.predictive_trajectory import (
    apply_reference_trajectory_prediction,
    evaluate_reference_trajectory_prediction,
    fit_reference_trajectory_calibration,
)


def _trace(scale: int, *, phase_offset: float = 0.0):
    rows = []
    for phase, start in (("FW", 0.0), ("BW", 10.0), ("OPT", 20.0)):
        for index, value in enumerate((scale, scale + 10, scale + 5)):
            rows.append(
                {
                    "event_id": f"{phase}-{index}",
                    "phase": phase,
                    "time_us": start + index + phase_offset,
                    "current_bytes": value,
                    "parameter_bytes": 2,
                    "optimizer_state_bytes": 3,
                    "saved_activation_bytes": value - 5,
                    "gradient_bytes": 0,
                    "temporary_bytes": 0,
                    "workspace_component_bytes": 0,
                    "allocator_residual_bytes": 0,
                }
            )
    return rows


def test_reference_prediction_preserves_timestamps_and_components():
    reference = _trace(100)
    target = _trace(200, phase_offset=5.0)
    calibration = fit_reference_trajectory_calibration(
        [
            {"batch": 2, "sequence": 512, "events": reference},
            {"batch": 4, "sequence": 512, "events": _trace(120)},
        ],
        target_shape={"batch": 8, "sequence": 512},
        grid_size=21,
    )
    predicted = apply_reference_trajectory_prediction(target, calibration)
    assert [row["time_us"] for row in predicted] == [row["time_us"] for row in target]
    assert [row["event_id"] for row in predicted] == [row["event_id"] for row in target]
    assert all(int(row["component_balance_bytes"]) == 0 for row in predicted)
    assert all(
        int(row["current_bytes"])
        == int(row["physical_component_sum_bytes"])
        + int(row["trajectory_model_residual_bytes"])
        for row in predicted
    )


def test_evaluation_reports_peak_and_time_errors():
    predicted = _trace(110)
    actual = _trace(100)
    metrics = evaluate_reference_trajectory_prediction(predicted, actual)
    assert metrics["peak_ape_pct"] > 0.0
    assert metrics["time_ape_pct"] == 0.0
    assert metrics["target_trajectory_consumed_for_prediction"] is False
