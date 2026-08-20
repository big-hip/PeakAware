from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from peakaware.cost.base import OpSignature


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profile_sdpa_costs.py"
SPEC = importlib.util.spec_from_file_location("profile_sdpa_costs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qkv_shapes_extracts_forward_and_backward_operands():
    forward = OpSignature(
        "sdpa",
        "aten._scaled_dot_product_efficient_attention.default",
        0,
        0,
        input_shapes=((1, 12, 128, 64),) * 3,
    )
    backward = OpSignature(
        "sdpa_backward",
        "aten._scaled_dot_product_efficient_attention_backward.default",
        0,
        0,
        input_shapes=((1, 12, 128, 64),) * 5 + ((1, 12, 128), (), ()),
    )

    assert MODULE._qkv_shapes(forward) == (((1, 12, 128, 64),) * 3 + (False,))
    assert MODULE._qkv_shapes(backward) == (((1, 12, 128, 64),) * 3 + (True,))


def test_qkv_shapes_rejects_incomplete_signature():
    signature = OpSignature(
        "sdpa",
        "aten._scaled_dot_product_efficient_attention.default",
        0,
        0,
        input_shapes=((1, 12, 128),) * 3,
    )

    with pytest.raises(ValueError, match="unsupported SDPA signature"):
        MODULE._qkv_shapes(signature)


def test_output_bytes_sums_nested_tensor_outputs():
    value = (torch.ones(2, dtype=torch.float32), {"x": torch.ones(3, dtype=torch.float16)})

    assert MODULE._output_bytes(value) == 14
