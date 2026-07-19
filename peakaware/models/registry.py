from __future__ import annotations

from dataclasses import dataclass
from functools import partial

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


class TinyMLP(nn.Module):
    def __init__(self, width: int = 8, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyAttentionBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = float(x.shape[-1]) ** -0.5
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        hidden = self.proj(attended + x).relu()
        return self.out(hidden).mean(dim=1)


@dataclass(frozen=True)
class DenseBatchBuilder:
    width: int

    def __call__(self, batch_size: int) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.randn(batch_size, self.width),), {}


@dataclass(frozen=True)
class SequenceBatchBuilder:
    sequence_length: int
    width: int

    def __call__(self, batch_size: int) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.randn(batch_size, self.sequence_length, self.width),), {}


def squared_mean_loss(out: torch.Tensor) -> torch.Tensor:
    return out.pow(2).mean()


def build_sgd_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01)


def build_tiny_residual_task(width: int = 8) -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name=f"tiny_residual_w{width}",
        build_model=partial(TinyResidual, width),
        build_batch=DenseBatchBuilder(width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
    )


def build_tiny_mlp_task(width: int = 8, depth: int = 3) -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name=f"tiny_mlp_w{width}_d{depth}",
        build_model=partial(TinyMLP, width, depth),
        build_batch=DenseBatchBuilder(width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
    )


def build_tiny_attention_task(width: int = 8, sequence_length: int = 4) -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name=f"tiny_attention_w{width}_s{sequence_length}",
        build_model=partial(TinyAttentionBlock, width),
        build_batch=SequenceBatchBuilder(sequence_length, width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
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
        registry.register(build_tiny_mlp_task())
        registry.register(build_tiny_attention_task())
        return registry
