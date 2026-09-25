"""Tests for multi-process measurement-window coordination."""

from __future__ import annotations

import time

from ltm100.core.multiproc import run_shards
from ltm100.core.op import OpResult, OpType


def _synchronized_entry(args: dict, proc_index: int) -> list[OpResult]:
    sync = args["_measure_sync"]
    sync["queue"].put(("start", proc_index, ""))
    sync["start_release"].wait()
    started = time.time()
    time.sleep(0.01)
    ended = time.time()
    sync["queue"].put(("end", proc_index, ""))
    sync["end_release"].wait()
    return [
        OpResult(
            type=OpType.ADD,
            user_id=f"u{proc_index}",
            started_at=started,
            ended_at=ended,
            status="ok",
        )
    ]


def test_parent_hooks_bracket_every_shards_measured_window():
    boundaries: dict[str, float] = {}

    results = run_shards(
        _synchronized_entry,
        {},
        2,
        on_measure_start=lambda: boundaries.setdefault("start", time.time()),
        on_measure_end=lambda: boundaries.setdefault("end", time.time()),
    )

    assert len(results) == 2
    assert all(result.started_at >= boundaries["start"] for result in results)
    assert all(result.ended_at <= boundaries["end"] for result in results)
