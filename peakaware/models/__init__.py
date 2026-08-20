from peakaware.contracts import WorkloadSpec

from .registry import (
    TinyAttentionBlock,
    GPT2Like,
    TinyMLP,
    TinyResidual,
    TrainingTaskRegistry,
    build_bert_base_task,
    build_bert_base_full_task,
    build_gpt2_task,
    build_gpt2_small_full_task,
    build_resnet50_task,
    build_tiny_attention_task,
    build_tiny_mlp_task,
    build_tiny_residual_task,
    build_vit_b16_task,
)

__all__ = [
    "TinyAttentionBlock",
    "GPT2Like",
    "TinyMLP",
    "TinyResidual",
    "TrainingTaskRegistry",
    "WorkloadSpec",
    "build_bert_base_task",
    "build_bert_base_full_task",
    "build_gpt2_task",
    "build_gpt2_small_full_task",
    "build_resnet50_task",
    "build_tiny_attention_task",
    "build_tiny_mlp_task",
    "build_tiny_residual_task",
    "build_vit_b16_task",
]
