from __future__ import annotations

from peakaware.contracts import JointTrainingIR, StorageInfo, ValueInfo
from peakaware.ir.alias import merge_alias_storage_groups
from peakaware.memory.simulator import _activation_storage_ids


def test_alias_merge_keeps_base_physical_size_for_expand_and_slice() -> None:
    merged = merge_alias_storage_groups(
        {
            0: ((0,), 128, True),
            1: ((1,), 1024, False),
            2: ((2,), 32, False),
        },
        ((1, 0), (2, 1)),
    )

    assert merged == {0: ((0, 1, 2), 128, True)}


def test_external_alias_storage_is_not_an_activation_candidate() -> None:
    ir = JointTrainingIR(
        ops=(),
        values=(
            ValueInfo(0, None, (), 0, 128, "input", False, False, None, "weight"),
            ValueInfo(1, 1, (), 0, 128, "fw", True, True, None, "weight_t"),
            ValueInfo(2, 2, (), 2, 256, "fw", True, True, None, "activation"),
        ),
        storages=(
            StorageInfo(0, (0, 1), 128, True),
            StorageInfo(2, (2,), 256, False),
        ),
        regions=(),
        graph_key="external-alias-filter",
    )

    assert _activation_storage_ids(ir) == frozenset({2})
