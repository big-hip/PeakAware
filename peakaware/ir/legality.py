from __future__ import annotations

from torch import fx


_RNG_TOKENS = ("dropout", "rand", "randn", "bernoulli", "multinomial")
_SIDE_EFFECT_TOKENS = ("_", "copy", "resize", "setitem")


def requires_rng_preservation(node: fx.Node) -> bool:
    text = f"{node.name} {node.target}".lower()
    return any(token in text for token in _RNG_TOKENS)


def has_unmodeled_side_effect(node: fx.Node) -> bool:
    target = str(node.target).lower()
    if node.op in {"placeholder", "output"}:
        return False
    if target.endswith("_"):
        return True
    return any(token in target for token in _SIDE_EFFECT_TOKENS if token != "_")


def classify_recompute_legality(node: fx.Node) -> tuple[bool, str | None]:
    if node.op in {"placeholder", "get_attr", "output"}:
        return False, "external_or_graph_boundary"
    if requires_rng_preservation(node):
        return False, "requires_rng_preservation"
    if has_unmodeled_side_effect(node):
        return False, "unmodeled_side_effect"
    return True, None


def mark_mandatory_saves(node: fx.Node, crosses_fw_bw: bool) -> str | None:
    legal, reason = classify_recompute_legality(node)
    if not legal and crosses_fw_bw:
        return reason
    if node.op == "output":
        return "graph_output_pinning"
    return None
