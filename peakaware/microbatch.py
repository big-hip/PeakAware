from __future__ import annotations

from peakaware.api import optimize_training
from peakaware.config import PeakAwareConfig
from peakaware.contracts import MicrobatchCandidateResult, MicrobatchSearchResult, TrainingTaskSpec


def optimize_microbatches(
    task: TrainingTaskSpec,
    microbatch_sizes: tuple[int, ...],
    *,
    memory_budget_bytes: int,
    config: PeakAwareConfig | None = None,
) -> MicrobatchSearchResult:
    candidates: list[MicrobatchCandidateResult] = []
    failures: list[str] = []
    for microbatch_size in microbatch_sizes:
        model = task.build_model()
        optimizer = task.build_optimizer(model)
        args, kwargs = task.build_batch(microbatch_size)
        try:
            result = optimize_training(
                model,
                args,
                example_kwargs=kwargs,
                loss_fn=task.loss_fn,
                optimizer=optimizer,
                memory_budget_bytes=memory_budget_bytes,
                config=config,
            )
        except Exception as exc:
            failures.append(f"{microbatch_size}: {type(exc).__name__}: {exc}")
            continue
        step_us = max(result.executable.measured_step_us, 1.0)
        candidates.append(
            MicrobatchCandidateResult(
                microbatch_size=microbatch_size,
                result=result,
                useful_samples_per_second=microbatch_size * 1_000_000.0 / step_us,
            )
        )
    if not candidates:
        raise RuntimeError(f"no feasible microbatch candidate: {failures}")
    candidates.sort(key=lambda candidate: (-candidate.useful_samples_per_second, candidate.microbatch_size))
    return MicrobatchSearchResult(candidates=tuple(candidates), selected=candidates[0])
