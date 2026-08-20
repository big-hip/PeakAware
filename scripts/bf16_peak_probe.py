#!/usr/bin/env python3
"""BF16 sanity probe: does mixed/BF16 precision change allocator-visible peak?

Compares FP32 vs pure-BF16 vs FP32-master autocast on the two activation-heavy
models (ResNet-50, ViT-B/16) used by the ablation, at the same batch sizes.
Reports the allocator-visible peak of a full FW-BW-Adam step.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.models.registry import TrainingTaskRegistry

MIB = 1024 * 1024

TASKS = {"resnet50": 1, "vit_b_16": 4}


def measure_peak(model: torch.nn.Module, optimizer, task, batch: int, mode: str) -> tuple[float, int]:
    torch.manual_seed(7)
    random.seed(7)
    model.train()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    peaks = []
    for rep in range(4):
        args, kwargs = task.build_batch(batch)
        args = tuple(a.to("cuda").to(dtype) for a in args)
        kwargs = {k: v.to("cuda").to(dtype) for k, v in kwargs.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        for warm in range(2):
            optimizer.zero_grad(set_to_none=True)
            if mode == "autocast":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = task.loss_fn(model(*args, **kwargs))
            else:
                loss = task.loss_fn(model(*args, **kwargs))
            loss.backward()
            optimizer.step()
        # measured repetition
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        if mode == "autocast":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = task.loss_fn(model(*args, **kwargs))
        else:
            loss = task.loss_fn(model(*args, **kwargs))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        del loss
    peaks.sort()
    return peaks[len(peaks) // 2] / MIB, peaks[len(peaks) // 2]


def main() -> None:
    registry = TrainingTaskRegistry.with_defaults()
    for name, batch in TASKS.items():
        task = registry.get(name)
        # FP32 baseline
        model = task.build_model().to("cuda")
        model_bf16 = task.build_model().to("cuda").to(torch.bfloat16)
        opt_f32 = task.build_optimizer(model)
        opt_bf16 = task.build_optimizer(model_bf16)
        opt_autocast = task.build_optimizer(model)
        # pre-initialize optimizer states (one step)
        args, kwargs = task.build_batch(batch)
        args = tuple(a.to("cuda") for a in args)
        kwargs = {k: v.to("cuda") for k, v in kwargs.items()}
        for m, o, d in (
            (model, opt_f32, torch.float32),
            (model_bf16, opt_bf16, torch.bfloat16),
            (model, opt_autocast, torch.float32),
        ):
            o.zero_grad(set_to_none=True)
            loss = task.loss_fn(m(*tuple(a.to(d) for a in args), **{k: v.to(d) for k, v in kwargs.items()}))
            loss.backward()
            o.step()
            o.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        p32, b32 = measure_peak(model, opt_f32, task, batch, "fp32")
        pbf, bbf = measure_peak(model_bf16, opt_bf16, task, batch, "bf16")
        pac, bac = measure_peak(model, opt_autocast, task, batch, "autocast")
        print(f"{name}: FP32={p32:.1f}MiB  pureBF16={pbf:.1f}MiB ({(pbf/p32-1)*100:+.1f}%)  "
              f"autocast={pac:.1f}MiB ({(pac/p32-1)*100:+.1f}%)")
        del model, model_bf16, opt_f32, opt_bf16, opt_autocast
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
