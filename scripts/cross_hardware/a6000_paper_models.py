"""Measure eager full-step peak memory on Ascend 910B for the paper's actual 4 models.

Model definitions are copied verbatim from peakaware.models.registry:
- bert_base: transformers.BertForSequenceClassification (hidden=64, L=2, heads=4,
  intermediate=256, vocab=30522, num_labels=2, all dropouts 0.0), seq=32, batch=4
- gpt2: GPT2Like (vocab=50257, seq=32, width=64, L=2, heads=4), batch=4
- resnet50: torchvision.models.resnet50(num_classes=10), 224x224, batch=4
- vit_b_16: torchvision.models.vit_b_16(num_classes=10, image_size=224), batch=4

Optimizers match the registry: AdamW (bert/gpt2/vit), SGD (resnet50).
Loss: logits squared-mean (handles SequenceClassifierOutput.logits).
"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn


# ---------------- copied verbatim from peakaware.models.registry ----------------
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


def logits_squared_mean_loss(out: object) -> torch.Tensor:
    logits = out if isinstance(out, torch.Tensor) else getattr(out, "logits")
    return logits.float().pow(2).mean()


def build_bert():
    from transformers import BertConfig, BertForSequenceClassification
    config = BertConfig(
        vocab_size=30522,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
    )
    return BertForSequenceClassification(config)


def build_gpt2():
    return GPT2Like(vocab_size=50257, sequence_length=32, width=64, num_layers=2, num_heads=4)


def build_resnet50():
    from torchvision.models import resnet50
    return resnet50(weights=None, num_classes=10)


def build_vit():
    from torchvision.models import vit_b_16
    return vit_b_16(weights=None, num_classes=10, image_size=224)


def build_input(name: str, batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    if name in ("bert_base", "gpt2"):
        vocab = 30522 if name == "bert_base" else 50257
        return (torch.randint(0, vocab, (batch, 32), device=device),)
    return (torch.randn(batch, 3, 224, 224, device=device),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert_base", "gpt2", "resnet50", "vit_b_16"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(0)

    model = {
        "bert_base": build_bert,
        "gpt2": build_gpt2,
        "resnet50": build_resnet50,
        "vit_b_16": build_vit,
    }[args.model]()
    model = model.to("cuda")
    model.train()

    inputs = build_input(args.model, args.batch, torch.device("cuda"))
    if args.model == "resnet50":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(*inputs)
        logits_squared_mean_loss(out).backward()
        optimizer.step()

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    peaks = []
    for _ in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        one_step()
        torch.cuda.synchronize()
        peaks.append(int(torch.cuda.max_memory_allocated()))
    peak = max(peaks)

    torch.cuda.reset_peak_memory_stats()
    logits_squared_mean_loss(model(*inputs)).backward()
    torch.cuda.synchronize()
    fw_bw_peak = int(torch.cuda.max_memory_allocated())

    param = sum(p.numel() * p.element_size() for p in model.parameters())
    result = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "peak_allocated_bytes": peak,
        "peak_MB": round(peak / 1e6, 3),
        "per_step_peaks_MB": [round(p / 1e6, 3) for p in peaks],
        "fw_bw_peak_MB": round(fw_bw_peak / 1e6, 3),
        "param_bytes": int(param),
        "param_MB": round(param / 1e6, 3),
        "torch": torch.__version__,
        "torch_npu": "n/a",
    }
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
