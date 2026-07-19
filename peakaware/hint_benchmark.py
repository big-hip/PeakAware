from __future__ import annotations

from dataclasses import asdict
from typing import Any

from peakaware.contracts import FixedTimeline, JointTrainingIR, OpInfo, RegionInfo, StorageInfo, ValueInfo
from peakaware.cost.base import OpCost, OpSignature
from peakaware.search.engine import search_plans_with_diagnostics


class _SyntheticHintCostProvider:
    source = "synthetic_hint_cost"

    def supports(self, signature: OpSignature) -> bool:
        del signature
        return True

    def estimate(self, signature: OpSignature) -> OpCost:
        cost_by_target = {
            "target_slow_0": 3.0,
            "target_slow_1": 3.0,
            "target_fast_2": 2.0,
            "target_fast_3": 2.0,
        }
        return OpCost(
            estimated_us=cost_by_target.get(signature.target, 2.0),
            memory_bytes=signature.output_bytes,
            source=self.source,
            confidence=1.0,
        )


def _synthetic_hint_ir() -> tuple[JointTrainingIR, FixedTimeline]:
    ops = tuple(
        OpInfo(
            id=index,
            name=f"fw_{index}",
            target=f"target_slow_{index}" if index < 2 else f"target_fast_{index}",
            phase="fw",
            input_value_ids=(),
            output_value_ids=(index,),
            recomputable=True,
            mandatory_save_reason=None,
        )
        for index in range(4)
    )
    values = tuple(
        ValueInfo(
            id=index,
            producer_id=index,
            consumer_ids=(),
            storage_id=index,
            logical_nbytes=100,
            phase="fw",
            crosses_fw_bw=True,
            recomputable=True,
            mandatory_save_reason=None,
            name=f"v{index}",
        )
        for index in range(4)
    )
    storages = tuple(
        StorageInfo(id=index, value_ids=(index,), physical_nbytes=100, is_external=False)
        for index in range(4)
    )
    ir = JointTrainingIR(
        ops=ops,
        values=values,
        storages=storages,
        regions=(RegionInfo(id=0, name="synthetic_hint_region", op_ids=(0, 1, 2, 3)),),
        graph_key="synthetic_hint_graph",
    )
    fixed = FixedTimeline(
        parameter_bytes=100,
        buffer_bytes=0,
        gradient_bytes=0,
        optimizer_state_bytes=0,
        optimizer_temporary_bytes=500,
    )
    return ir, fixed


def run_synthetic_hint_effectiveness_benchmark() -> dict[str, Any]:
    ir, fixed = _synthetic_hint_ir()
    provider = _SyntheticHintCostProvider()
    _, enabled = search_plans_with_diagnostics(
        ir,
        fixed,
        budget_bytes=10_000,
        safety_margin_bytes=0,
        cost_provider=provider,
        enable_diagnostic_hints=True,
        top_k=2,
    )
    _, disabled = search_plans_with_diagnostics(
        ir,
        fixed,
        budget_bytes=10_000,
        safety_margin_bytes=0,
        cost_provider=provider,
        enable_diagnostic_hints=False,
        top_k=2,
    )
    row = {
        "case_name": "diagnostic_hint_reorders_greedy_candidates",
        "expected_hint_target_storage_ids": (0, 1),
        "enabled": asdict(enabled),
        "disabled": asdict(disabled),
        "candidate_match_delta": (
            enabled.diagnostic_hint_candidate_match_count
            - disabled.diagnostic_hint_candidate_match_count
        ),
        "order_delta_count_delta": (
            enabled.diagnostic_hint_order_delta_count
            - disabled.diagnostic_hint_order_delta_count
        ),
        "changed_search_order": enabled.diagnostic_hint_order_changed
        and not disabled.diagnostic_hint_order_changed,
    }
    return {
        "case_count": 1,
        "changed_search_order_case_count": int(row["changed_search_order"]),
        "verdict": "changed_search_order" if row["changed_search_order"] else "neutral",
        "rows": (row,),
    }
