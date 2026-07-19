from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from peakaware.contracts import TrainingTaskSpec, WorkloadSpec


WORKLOAD_SCHEMA_VERSION = "1.0"


def _workload_spec(
    *,
    registry_key: str,
    display_name: str,
    model_family: str,
    implementation: str,
    model_config: dict[str, object],
    input_config: dict[str, object],
    optimizer_config: dict[str, object],
    loss_config: dict[str, object],
    compute_dtype: str,
    parameter_dtype: str,
) -> WorkloadSpec:
    return WorkloadSpec(
        schema_version=WORKLOAD_SCHEMA_VERSION,
        registry_key=registry_key,
        display_name=display_name,
        model_family=model_family,
        implementation=implementation,
        model_config=model_config,
        input_config=input_config,
        optimizer_config=optimizer_config,
        loss_config=loss_config,
        compute_dtype=compute_dtype,
        parameter_dtype=parameter_dtype,
    )


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


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, sequence_length: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.ln_1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask.view(1, 1, sequence_length, sequence_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.ln_1(x)
        batch_size, sequence_length, width = normalized.shape
        qkv = self.qkv(normalized).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(self.causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, sequence_length, width)
        x = residual + self.proj(attended)
        return x + self.mlp(self.ln_2(x))


class GPT2Like(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        sequence_length: int = 32,
        width: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(sequence_length, width)
        self.blocks = nn.ModuleList(
            CausalSelfAttentionBlock(width, num_heads, sequence_length)
            for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.register_buffer("position_ids", torch.arange(sequence_length), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids) + self.position_embedding(self.position_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


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


@dataclass(frozen=True)
class ImageBatchBuilder:
    channels: int
    height: int
    width: int

    def __call__(self, batch_size: int) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.randn(batch_size, self.channels, self.height, self.width),), {}


@dataclass(frozen=True)
class TokenBatchBuilder:
    sequence_length: int
    vocab_size: int

    def __call__(self, batch_size: int) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.randint(0, self.vocab_size, (batch_size, self.sequence_length)),), {}


def squared_mean_loss(out: torch.Tensor) -> torch.Tensor:
    return out.pow(2).mean()


def logits_squared_mean_loss(out: object) -> torch.Tensor:
    logits = out if isinstance(out, torch.Tensor) else getattr(out, "logits")
    return logits.float().pow(2).mean()


def build_sgd_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01)


def build_adamw_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def build_resnet50_model(num_classes: int = 10) -> nn.Module:
    from torchvision.models import resnet50

    return resnet50(weights=None, num_classes=num_classes)


def build_vit_b16_model(num_classes: int = 10, image_size: int = 224) -> nn.Module:
    from torchvision.models import vit_b_16

    return vit_b_16(weights=None, num_classes=num_classes, image_size=image_size)


def build_bert_base_model(
    num_labels: int = 2,
    vocab_size: int = 30522,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 256,
) -> nn.Module:
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


def build_gpt2_model(
    vocab_size: int = 50257,
    n_positions: int = 32,
    n_embd: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
) -> nn.Module:
    return GPT2Like(
        vocab_size=vocab_size,
        sequence_length=n_positions,
        width=n_embd,
        num_layers=n_layer,
        num_heads=n_head,
    )


def build_tiny_residual_task(width: int = 8) -> TrainingTaskSpec:
    registry_key = f"tiny_residual_w{width}"
    return TrainingTaskSpec(
        name=registry_key,
        build_model=partial(TinyResidual, width),
        build_batch=DenseBatchBuilder(width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key=registry_key,
            display_name=f"TinyResidual-W{width}",
            model_family="synthetic_residual_mlp",
            implementation="peakaware.models.registry.TinyResidual",
            model_config={"width": width, "output_width": 1},
            input_config={
                "kind": "dense",
                "shape_without_batch": [width],
                "dtype": "torch.float32",
                "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
            },
            optimizer_config={
                "name": "SGD",
                "lr": 0.01,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": None,
                "differentiable": False,
                "fused": None,
            },
            loss_config={"name": "squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_tiny_mlp_task(width: int = 8, depth: int = 3) -> TrainingTaskSpec:
    registry_key = f"tiny_mlp_w{width}_d{depth}"
    return TrainingTaskSpec(
        name=registry_key,
        build_model=partial(TinyMLP, width, depth),
        build_batch=DenseBatchBuilder(width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key=registry_key,
            display_name=f"TinyMLP-W{width}-D{depth}",
            model_family="synthetic_mlp",
            implementation="peakaware.models.registry.TinyMLP",
            model_config={"width": width, "depth": depth, "output_width": 1},
            input_config={
                "kind": "dense",
                "shape_without_batch": [width],
                "dtype": "torch.float32",
                "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
            },
            optimizer_config={
                "name": "SGD",
                "lr": 0.01,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": None,
                "differentiable": False,
                "fused": None,
            },
            loss_config={"name": "squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_tiny_attention_task(width: int = 8, sequence_length: int = 4) -> TrainingTaskSpec:
    registry_key = f"tiny_attention_w{width}_s{sequence_length}"
    return TrainingTaskSpec(
        name=registry_key,
        build_model=partial(TinyAttentionBlock, width),
        build_batch=SequenceBatchBuilder(sequence_length, width),
        loss_fn=squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key=registry_key,
            display_name=f"TinyAttention-W{width}-S{sequence_length}",
            model_family="synthetic_attention",
            implementation="peakaware.models.registry.TinyAttentionBlock",
            model_config={"width": width, "output_width": 1},
            input_config={
                "kind": "dense_sequence",
                "shape_without_batch": [sequence_length, width],
                "dtype": "torch.float32",
                "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
            },
            optimizer_config={
                "name": "SGD",
                "lr": 0.01,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": None,
                "differentiable": False,
                "fused": None,
            },
            loss_config={"name": "squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_resnet50_task(image_size: int = 224, num_classes: int = 10) -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name="resnet50",
        build_model=partial(build_resnet50_model, num_classes=num_classes),
        build_batch=ImageBatchBuilder(3, image_size, image_size),
        loss_fn=logits_squared_mean_loss,
        build_optimizer=build_sgd_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key="resnet50",
            display_name="ResNet-50",
            model_family="convolutional_neural_network",
            implementation="torchvision.models.resnet50",
            model_config={
                "architecture": "ResNet",
                "block": "Bottleneck",
                "block_counts": [3, 4, 6, 3],
                "num_classes": num_classes,
                "weights": None,
                "zero_init_residual": False,
                "groups": 1,
                "width_per_group": 64,
                "replace_stride_with_dilation": [False, False, False],
                "norm_layer": "torch.nn.BatchNorm2d",
            },
            input_config={
                "kind": "image",
                "shape_without_batch": [3, image_size, image_size],
                "dtype": "torch.float32",
                "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
            },
            optimizer_config={
                "name": "SGD",
                "lr": 0.01,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": None,
                "differentiable": False,
                "fused": None,
            },
            loss_config={"name": "logits_squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_vit_b16_task(image_size: int = 224, num_classes: int = 10) -> TrainingTaskSpec:
    return TrainingTaskSpec(
        name="vit_b_16",
        build_model=partial(build_vit_b16_model, num_classes=num_classes, image_size=image_size),
        build_batch=ImageBatchBuilder(3, image_size, image_size),
        loss_fn=logits_squared_mean_loss,
        build_optimizer=build_adamw_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key="vit_b_16",
            display_name="ViT-B/16",
            model_family="vision_transformer",
            implementation="torchvision.models.vit_b_16",
            model_config={
                "architecture": "VisionTransformer",
                "patch_size": 16,
                "num_layers": 12,
                "num_heads": 12,
                "hidden_dim": 768,
                "mlp_dim": 3072,
                "image_size": image_size,
                "num_classes": num_classes,
                "weights": None,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "norm_layer": "partial(torch.nn.LayerNorm, eps=1e-06)",
                "representation_size": None,
            },
            input_config={
                "kind": "image",
                "shape_without_batch": [3, image_size, image_size],
                "dtype": "torch.float32",
                "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
            },
            optimizer_config={
                "name": "AdamW",
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            loss_config={"name": "logits_squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_bert_base_task(
    sequence_length: int = 32,
    vocab_size: int = 30522,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 256,
) -> TrainingTaskSpec:
    display_name = f"BERT-like-{num_hidden_layers}L-{hidden_size}H"
    return TrainingTaskSpec(
        name="bert_base",
        build_model=partial(
            build_bert_base_model,
            num_labels=2,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        ),
        build_batch=TokenBatchBuilder(sequence_length, vocab_size),
        loss_fn=logits_squared_mean_loss,
        build_optimizer=build_adamw_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key="bert_base",
            display_name=display_name,
            model_family="transformer_encoder",
            implementation="transformers.BertForSequenceClassification",
            model_config={
                "architecture": "BertForSequenceClassification",
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "intermediate_size": intermediate_size,
                "num_labels": 2,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
                "classifier_dropout": None,
                "pad_token_id": 0,
            },
            input_config={
                "kind": "token_ids",
                "shape_without_batch": [sequence_length],
                "dtype": "torch.int64",
                "vocab_size": vocab_size,
                "distribution": {"name": "randint", "low": 0, "high_exclusive": vocab_size},
            },
            optimizer_config={
                "name": "AdamW",
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            loss_config={"name": "logits_squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
    )


def build_gpt2_task(
    sequence_length: int = 32,
    vocab_size: int = 50257,
    n_embd: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
) -> TrainingTaskSpec:
    display_name = f"GPT2-like-{n_layer}L-{n_embd}H"
    return TrainingTaskSpec(
        name="gpt2",
        build_model=partial(
            build_gpt2_model,
            vocab_size=vocab_size,
            n_positions=sequence_length,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        ),
        build_batch=TokenBatchBuilder(sequence_length, vocab_size),
        loss_fn=logits_squared_mean_loss,
        build_optimizer=build_adamw_optimizer,
        dynamic_shapes=None,
        workload=_workload_spec(
            registry_key="gpt2",
            display_name=display_name,
            model_family="causal_transformer_decoder",
            implementation="peakaware.models.registry.GPT2Like",
            model_config={
                "architecture": "GPT2Like",
                "vocab_size": vocab_size,
                "sequence_length": sequence_length,
                "width": n_embd,
                "num_layers": n_layer,
                "num_heads": n_head,
                "mlp_ratio": 4,
                "lm_head_bias": False,
                "tie_word_embeddings": False,
            },
            input_config={
                "kind": "token_ids",
                "shape_without_batch": [sequence_length],
                "dtype": "torch.int64",
                "vocab_size": vocab_size,
                "distribution": {"name": "randint", "low": 0, "high_exclusive": vocab_size},
            },
            optimizer_config={
                "name": "AdamW",
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            loss_config={"name": "logits_squared_mean", "reduction": "mean"},
            compute_dtype="torch.float32",
            parameter_dtype="torch.float32",
        ),
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
        registry.register(build_resnet50_task())
        registry.register(build_vit_b16_task())
        registry.register(build_bert_base_task())
        registry.register(build_gpt2_task())
        return registry
