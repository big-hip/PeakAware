"""Symmetric peak-memory measurement across cuda / npu with IDENTICAL code path.

Measures the allocator-visible peak of a complete FW-BW-OPT training step for
mlp_3, using torch.cuda or torch.npu depending on --device. The code path is
identical except for the device-backend API indirection.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "npu"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--steps-clean", action="store_true",
                        help="measure peak over a single clean step after thorough warmup")
    parser.add_argument("--fp32-precise", action="store_true",
                        help="disable tf32 / cudnn benchmark to isolate workspace effects")
    args = parser.parse_args()

    import torch

    dev_name = args.device
    if dev_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("cuda unavailable")
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        backend = torch.cuda
        dev = torch.device("cuda")
        if args.fp32_precise:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
    else:
        import torch_npu
        torch_npu.__version__ = "2.9.0"  # work around te_fusion parse bug
        torch.npu.set_device(0)
        torch.manual_seed(0)
        backend = torch.npu
        dev = torch.device("npu")

    hidden = args.hidden
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, 2 * hidden), torch.nn.GELU(),
        torch.nn.Linear(2 * hidden, 2 * hidden), torch.nn.GELU(),
        torch.nn.Linear(2 * hidden, hidden),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inp = torch.randn(args.batch, hidden, device=dev)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    def one_step():
        opt.zero_grad(set_to_none=True)
        model(inp).square().mean().backward()
        opt.step()

    # thorough warmup so all kernels / workspace are materialized
    for _ in range(args.warmup):
        one_step()
    backend.synchronize(dev if dev_name == "cuda" else None)

    backend.reset_peak_memory_stats()
    peaks = []
    for _ in range(args.steps):
        backend.reset_peak_memory_stats()
        one_step()
        backend.synchronize(dev if dev_name == "cuda" else None)
        peaks.append(int(backend.max_memory_allocated()))
    peak = max(peaks)
    reserved = int(backend.max_memory_reserved())

    # forward-only peak for diagnosis
    backend.reset_peak_memory_stats()
    model(inp)
    backend.synchronize(dev if dev_name == "cuda" else None)
    fw_peak = int(backend.max_memory_allocated())

    result = {
        "device": dev_name,
        "device_name": backend.get_device_name(0) if hasattr(backend, "get_device_name") else None,
        "batch": args.batch,
        "hidden": hidden,
        "param_bytes": int(param_bytes),
        "param_MB": round(param_bytes / 1e6, 3),
        "peak_allocated_bytes": peak,
        "peak_allocated_MB": round(peak / 1e6, 3),
        "peak_reserved_MB": round(reserved / 1e6, 3),
        "per_step_peaks_MB": [round(p / 1e6, 3) for p in peaks],
        "forward_only_peak_MB": round(fw_peak / 1e6, 3),
        "peak_over_param": round(peak / param_bytes, 2),
        "torch": torch.__version__,
        "fp32_precise": args.fp32_precise,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
