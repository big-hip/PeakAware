from __future__ import annotations

import torch
from torch import nn

from peakaware.contracts import TrainingTaskSpec


class TinyResidual(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.a = nn.Linear(width, width)
        self.b = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.a(x).relu()
        hidden = self.b(residual).relu()
        return self.out(hidden + residual)


def build_tiny_residual_task(width: int = 8) -> TrainingTaskSpec:
    def build_batch(batch_size: int) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.randn(batch_size, width),), {}

    return TrainingTaskSpec(
        name=f"tiny_residual_w{width}",
        build_model=lambda: TinyResidual(width),
        build_batch=build_batch,
        loss_fn=lambda out: out.pow(2).mean(),
        build_optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.01),
        dynamic_shapes=None,
    )


class TrainingTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TrainingTaskSpec] = {}

    def register(self, task: TrainingTaskSpec) -> None:
        if task.name in self._tasks:
            raise ValueError(f"duplicate training task: {task.name}")
        self._tasks[task.name] = task

    def get(self, name: str) -> TrainingTaskSpec:
        return self._tasks[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))

    @classmethod
    def with_defaults(cls) -> "TrainingTaskRegistry":
        registry = cls()
        registry.register(build_tiny_residual_task())
        return registry
