from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.utils.checkpoint as checkpoint


def _module_attrs(module_name: str, needles: tuple[str, ...]) -> list[str]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return []
    return sorted(name for name in dir(module) if any(needle in name.lower() for needle in needles))


def summarize_external_baseline_capabilities() -> dict[str, Any]:
    sac_available = hasattr(checkpoint, "create_selective_checkpoint_contexts")
    functorch_memory_attrs = _module_attrs("torch._functorch.config", ("memory", "budget"))
    inductor_memory_attrs = _module_attrs("torch._inductor.config", ("memory", "budget"))
    memory_budget_attrs = tuple(functorch_memory_attrs + inductor_memory_attrs)
    return {
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "baselines": {
            "selective_activation_checkpointing": {
                "status": "available" if sac_available else "unavailable",
                "api": "torch.utils.checkpoint.create_selective_checkpoint_contexts",
                "provenance": {
                    "module": "torch.utils.checkpoint",
                    "policy_api": "torch.utils.checkpoint.CheckpointPolicy"
                    if hasattr(checkpoint, "CheckpointPolicy")
                    else None,
                },
                "unavailable_reason": None if sac_available else "SAC context API is absent in this PyTorch build",
            },
            "aot_min_cut_proxy": {
                "status": "proxy",
                "api": "PeakAware torch_min_cut plan",
                "provenance": {
                    "plan_id": "torch_min_cut",
                    "strategy": "mandatory-save-only proxy for saved-activation min-cut",
                },
                "unavailable_reason": None,
            },
            "inductor_memory_budget": {
                "status": "available" if memory_budget_attrs else "unavailable",
                "api": None if not memory_budget_attrs else memory_budget_attrs,
                "provenance": {
                    "functorch_memory_attrs": functorch_memory_attrs,
                    "inductor_memory_attrs": inductor_memory_attrs,
                },
                "unavailable_reason": None
                if memory_budget_attrs
                else "No activation memory-budget config attribute was found in torch._functorch.config or torch._inductor.config",
            },
        },
    }
