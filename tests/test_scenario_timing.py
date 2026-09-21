"""Timing semantics shared by scenario plans."""

from __future__ import annotations

import itertools
import random

import pytest

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.core.op import OpType
from ltm100.core.scenarios import Mixed, _seed_for


@pytest.mark.parametrize(
    ("search_weight", "expected_type"),
    [(0.0, OpType.ADD), (1.0, OpType.SEARCH)],
)
def test_mixed_applies_think_delay_to_add_and_search(
    monkeypatch, search_weight: float, expected_type: OpType
) -> None:
    monkeypatch.setattr(random.Random, "uniform", lambda self, low, high: high)
    scenario = Mixed(search_weight=search_weight, think=0.25)
    dataset = SyntheticAdapter(memories_per_user=4)

    op = next(scenario.plan("syn_user_00000", dataset, {"seed": 0}))

    assert op.type is expected_type
    assert op.delay == 0.25


def test_mixed_add_delay_preserves_the_seeded_operation_mix() -> None:
    user = "syn_user_00000"
    seed = 7
    search_weight = 0.6
    legacy_rng = random.Random(_seed_for(seed, user))
    expected: list[OpType] = []
    for _ in range(100):
        if legacy_rng.random() < search_weight:
            expected.append(OpType.SEARCH)
            legacy_rng.uniform(0.0, 0.25)
        else:
            expected.append(OpType.ADD)

    scenario = Mixed(search_weight=search_weight, think=0.25)
    dataset = SyntheticAdapter(memories_per_user=10)
    actual = [
        op.type
        for op in itertools.islice(
            scenario.plan(user, dataset, {"seed": seed}), len(expected)
        )
    ]

    assert actual == expected
