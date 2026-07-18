from __future__ import annotations

from enum import Enum
from typing import Protocol


class ServiceKind(str, Enum):
    CAPTURE_BACKEND = "capture_backend"
    IR_PASS = "ir_pass"
    FIXED_MEMORY_PROVIDER = "fixed_memory_provider"
    COST_PROVIDER = "cost_provider"
    CANDIDATE_POLICY = "candidate_policy"
    SEARCH_STRATEGY = "search_strategy"
    PARTITION_BACKEND = "partition_backend"
    RUNTIME_VALIDATOR = "runtime_validator"
    PLAN_DIAGNOSTIC = "plan_diagnostic"
    ROOT_CAUSE_RULE = "root_cause_rule"


class FailurePolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    DEGRADE = "degrade"
    OBSERVE_ONLY = "observe_only"


class PeakAwarePlugin(Protocol):
    name: str
    version: str

    def register(self, registry: object) -> None:
        ...
