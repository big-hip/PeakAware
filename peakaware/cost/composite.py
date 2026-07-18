from __future__ import annotations

from dataclasses import dataclass

from peakaware.cost.base import CostProvider, OpCost, OpSignature, RooflineFallbackProvider


@dataclass(frozen=True)
class CostQueryResult:
    cost: OpCost
    attempted_sources: tuple[str, ...]


class CompositeCostProvider:
    def __init__(self, providers: tuple[CostProvider, ...]) -> None:
        self.providers = providers or (RooflineFallbackProvider(),)

    def supports(self, signature: OpSignature) -> bool:
        return any(provider.supports(signature) for provider in self.providers)

    def estimate(self, signature: OpSignature) -> OpCost | None:
        result = self.estimate_with_provenance(signature)
        return None if result is None else result.cost

    def estimate_with_provenance(self, signature: OpSignature) -> CostQueryResult | None:
        attempted: list[str] = []
        for provider in self.providers:
            source = getattr(provider, "source", provider.__class__.__name__)
            attempted.append(str(source))
            if not provider.supports(signature):
                continue
            cost = provider.estimate(signature)
            if cost is not None:
                return CostQueryResult(cost=cost, attempted_sources=tuple(attempted))
        return None


def rank_providers(providers: tuple[CostProvider, ...]) -> tuple[CostProvider, ...]:
    priority = {
        "profile_db_exact": 100,
        "profile_db_interpolated": 80,
        "legacy_adapter:static_fallback": 50,
        "static_fallback": 30,
        "roofline_fallback": 10,
    }
    return tuple(sorted(providers, key=lambda provider: -priority.get(str(getattr(provider, "source", "")), 0)))


def build_composite_provider(providers: tuple[CostProvider, ...] = ()) -> CompositeCostProvider:
    return CompositeCostProvider(rank_providers(providers) + (RooflineFallbackProvider(),))
