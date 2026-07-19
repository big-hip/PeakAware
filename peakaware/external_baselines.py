from __future__ import annotations

from typing import Any

import torch
import torch.utils.checkpoint as checkpoint


def _safe_attr(module_name: str, attribute: str) -> tuple[bool, Any]:
    try:
        module = __import__(module_name, fromlist=(attribute,))
        return True, getattr(module, attribute)
    except (ImportError, AttributeError):
        return False, None


def summarize_external_baseline_capabilities() -> dict[str, Any]:
    sac_available = hasattr(checkpoint, "create_selective_checkpoint_contexts")
    budget_available, budget_value = _safe_attr("torch._functorch.config", "activation_memory_budget")
    min_cut_available, min_cut_api = _safe_attr(
        "torch._functorch.partitioners", "min_cut_rematerialization_partition"
    )
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
            "pytorch_aot_min_cut": {
                "status": "available" if min_cut_available and budget_available else "unavailable",
                "api": "torch._functorch.partitioners.min_cut_rematerialization_partition",
                "provenance": {
                    "activation_memory_budget": budget_value if budget_available else None,
                    "config_api": "torch._functorch.config.patch",
                    "partitioner_callable": getattr(min_cut_api, "__name__", None),
                },
                "unavailable_reason": None
                if min_cut_available and budget_available
                else "Native min-cut partitioner or activation_memory_budget config is unavailable",
            },
            "inductor_memory_budget": {
                "status": "available" if budget_available else "unavailable",
                "api": "torch._functorch.config.activation_memory_budget",
                "provenance": {
                    "activation_memory_budget": budget_value if budget_available else None,
                    "config_api": "torch._functorch.config.patch",
                },
                "unavailable_reason": None
                if budget_available
                else "torch._functorch.config.activation_memory_budget is unavailable",
            },
        },
    }
