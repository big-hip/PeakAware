from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def build_plan_evaluation_key(
    *,
    analysis_key: str,
    optimizer_mode: str,
    cost_database_version: str,
    search_policy_version: str,
    budget_bucket: int,
) -> str:
    return _stable_hash(
        {
            "kind": "plan_evaluation",
            "analysis_key": analysis_key,
            "optimizer_mode": optimizer_mode,
            "cost_database_version": cost_database_version,
            "search_policy_version": search_policy_version,
            "budget_bucket": budget_bucket,
        }
    )


def build_compiled_artifact_key(
    *,
    lowered_plan_fingerprint: str,
    state_signature: str,
    input_guards: tuple[tuple[str, str], ...],
    device_capability: str,
    torch_version: str,
    compiler_version: str,
    partition_plugin_version: str,
) -> str:
    return _stable_hash(
        {
            "kind": "compiled_artifact",
            "lowered_plan_fingerprint": lowered_plan_fingerprint,
            "state_signature": state_signature,
            "input_guards": input_guards,
            "device_capability": device_capability,
            "torch_version": torch_version,
            "compiler_version": compiler_version,
            "partition_plugin_version": partition_plugin_version,
        }
    )
