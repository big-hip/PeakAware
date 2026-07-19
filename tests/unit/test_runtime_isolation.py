import time

from peakaware.runtime.isolation import run_in_worker_process


def _double(payload):
    return payload["x"] * 2


def _raises(payload):
    del payload
    raise RuntimeError("boom")


def _sleeps(payload):
    time.sleep(payload["seconds"])
    return "done"


def _large_value(payload):
    return bytes(payload["size"])


def test_worker_process_returns_value():
    result = run_in_worker_process(_double, {"x": 21}, timeout_s=5)

    assert result.ok
    assert result.value == 42


def test_worker_process_returns_value_larger_than_queue_pipe_buffer():
    result = run_in_worker_process(_large_value, {"size": 8 << 20}, timeout_s=5)

    assert result.ok
    assert len(result.value) == 8 << 20


def test_worker_process_returns_exception_details():
    result = run_in_worker_process(_raises, {}, timeout_s=5)

    assert not result.ok
    assert result.error_type == "RuntimeError"
    assert result.message == "boom"
    assert "RuntimeError" in result.traceback


def test_worker_process_times_out():
    result = run_in_worker_process(_sleeps, {"seconds": 2}, timeout_s=0.1)

    assert not result.ok
    assert result.timed_out
    assert result.error_type == "TimeoutError"
