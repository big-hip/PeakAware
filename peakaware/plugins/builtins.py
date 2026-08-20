from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from peakaware.capture import capture_joint_graph
from peakaware.cost.attention import ScaledDotProductAttentionCostProvider
from peakaware.cost.base import MetadataViewCostProvider, StructuralZeroCostProvider
from peakaware.cost.legacy_adapter import LegacyCostmodelAdapter
from peakaware.cost.profile_db import ExactProfileProvider, InterpolatedProfileProvider, ProfileDB
from peakaware.diagnostics import diagnose_plan, export_diagnostic_json, render_diagnostic_text

from .base import ServiceKind
from .registry import PluginRegistry, RegistrySnapshot


@dataclass(frozen=True)
class InductorCorrectionSummary:
    status: str
    confidence: float
    reason: str
    torch_version: str


class JointCapturePlugin:
    name = "joint_capture"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(ServiceKind.CAPTURE_BACKEND, "aot_or_fx", capture_joint_graph, priority=100)


class LegacyCostmodelPlugin:
    name = "legacy_costmodel"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "legacy_costmodel",
            LegacyCostmodelAdapter(),
            priority=30,
        )


class StructuralCostPlugin:
    name = "structural_cost"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "structural_zero",
            StructuralZeroCostProvider(),
            priority=120,
        )


class AttentionCostPlugin:
    name = "attention_cost"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "sdpa_fused_analytical",
            ScaledDotProductAttentionCostProvider(),
            priority=60,
        )


class MetadataViewCostPlugin:
    name = "metadata_view_cost"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "metadata_view_zero",
            MetadataViewCostProvider(),
            priority=110,
        )


class ProfileDBPlugin:
    name = "profile_db"
    version = "0.1"

    def __init__(self, path: str | Path) -> None:
        self.db = ProfileDB(path)

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service("profile_db", "default", self.db, priority=100)
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "profile_db_exact",
            ExactProfileProvider(self.db),
            priority=100,
        )
        registry.register_service(
            ServiceKind.COST_PROVIDER,
            "profile_db_interpolated",
            InterpolatedProfileProvider(self.db),
            priority=80,
        )


class PeakAnalysisPlugin:
    name = "peak_analysis"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_hook("after_ir_build", lambda ir, report: (ir, report), priority=0)


class PlanDiagnosticPlugin:
    name = "plan_diagnostic"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(ServiceKind.PLAN_DIAGNOSTIC, "diagnose_plan", diagnose_plan, priority=100)
        registry.register_service(ServiceKind.ROOT_CAUSE_RULE, "render_text", render_diagnostic_text, priority=10)
        registry.register_service(ServiceKind.ROOT_CAUSE_RULE, "export_json", export_diagnostic_json, priority=10)


class MinCutSeedPlugin:
    name = "min_cut_seed"
    version = "0.1"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(ServiceKind.CANDIDATE_POLICY, "min_cut_seed", "torch_default_partition_seed", priority=50)


class InductorCorrectionPlugin:
    name = "inductor_correction"
    version = "0.1"

    def correct(self, payload: Any) -> InductorCorrectionSummary:
        del payload
        return InductorCorrectionSummary(
            status="unavailable",
            confidence=0.0,
            reason="Inductor materialization summary is not available in this PyTorch build; use Top-K measurement",
            torch_version=torch.__version__,
        )

    def register(self, registry: PluginRegistry) -> None:
        registry.register_service(ServiceKind.RUNTIME_VALIDATOR, "inductor_correction", self.correct, priority=10)


def build_default_registry(*, profile_db_path: str | Path | None = None) -> RegistrySnapshot:
    registry = PluginRegistry()
    for plugin in (
        JointCapturePlugin(),
        StructuralCostPlugin(),
        MetadataViewCostPlugin(),
        AttentionCostPlugin(),
        LegacyCostmodelPlugin(),
        PeakAnalysisPlugin(),
        PlanDiagnosticPlugin(),
        MinCutSeedPlugin(),
        InductorCorrectionPlugin(),
    ):
        plugin.register(registry)
    if profile_db_path is not None:
        ProfileDBPlugin(profile_db_path).register(registry)
    return registry.freeze()
