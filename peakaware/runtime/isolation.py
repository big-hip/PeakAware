from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    value: Any = None
    error_type: str | None = None
    message: str | None = None
    traceback: str | None = None
    timed_out: bool = False


def serialize_candidate_request(payload: Any) -> Any:
    return payload


def _worker_main(fn: Callable[[Any], Any], payload: Any, output: mp.Queue) -> None:
    try:
        serialized_value = pickle.dumps(fn(payload), protocol=pickle.HIGHEST_PROTOCOL)
        output.put(WorkerResult(ok=True, value=serialized_value))
    except Exception as exc:
        output.put(
            WorkerResult(
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        )


def terminate_on_timeout(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def collect_worker_result(output: mp.Queue, process: mp.Process, timeout_s: float) -> WorkerResult:
    try:
        result = output.get(timeout=timeout_s)
    except queue.Empty:
        timed_out = process.is_alive()
        if timed_out:
            terminate_on_timeout(process)
            return WorkerResult(ok=False, error_type="TimeoutError", message="worker timed out", timed_out=True)
        process.join(timeout=1)
        if process.exitcode == 0:
            return WorkerResult(ok=False, error_type="EmptyWorkerResult", message="worker exited without a result")
        return WorkerResult(ok=False, error_type="WorkerExit", message=f"worker exited with code {process.exitcode}")
    process.join(timeout=1)
    if process.is_alive():
        terminate_on_timeout(process)
    if not result.ok:
        return result
    try:
        return WorkerResult(ok=True, value=pickle.loads(result.value))
    except Exception as exc:
        return WorkerResult(
            ok=False,
            error_type=type(exc).__name__,
            message=f"failed to deserialize worker result: {exc}",
            traceback=traceback.format_exc(),
        )


def run_in_worker_process(fn: Callable[[Any], Any], payload: Any, *, timeout_s: float = 60.0) -> WorkerResult:
    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_worker_main, args=(fn, serialize_candidate_request(payload), output))
    process.start()
    try:
        return collect_worker_result(output, process, timeout_s)
    finally:
        output.close()
        output.join_thread()
