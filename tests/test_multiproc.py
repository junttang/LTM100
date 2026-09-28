"""Tests for multi-process measurement-window coordination."""

from __future__ import annotations

import time

from ltm100.core.multiproc import ShardResult, run_shards
from ltm100.core.op import OpResult, OpType
from ltm100.core.runner import PreingestStats, SessionAdmissionStats


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


def _session_stats_entry(args: dict, proc_index: int) -> ShardResult:
    stats = SessionAdmissionStats()
    for _ in range(proc_index + 1):
        stats.record("offered", "group-a")
        stats.record("admitted", "group-a")
    stats.record("offered", "group-b")
    stats.record("rejected", "group-b")
    return ShardResult([], stats)


def _preingest_stats_entry(args: dict, proc_index: int) -> ShardResult:
    stats = PreingestStats()
    for _ in range(2):
        stats.record((proc_index + 1) * 10)
    return ShardResult([], SessionAdmissionStats(), stats)


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


def test_session_admission_stats_are_pooled_across_shards():
    result = run_shards(_session_stats_entry, {}, 2)

    assert isinstance(result, ShardResult)
    assert result.sessions.as_dict() == {
        "offered": 5,
        "admitted": 3,
        "rejected": 2,
        "rejection_rate": 0.4,
        "by_group": {
            "group-a": {
                "offered": 3,
                "admitted": 3,
                "rejected": 0,
                "rejection_rate": 0.0,
            },
            "group-b": {
                "offered": 2,
                "admitted": 0,
                "rejected": 2,
                "rejection_rate": 1.0,
            },
        },
    }


def test_preingest_stats_are_pooled_across_shards():
    result = run_shards(_preingest_stats_entry, {}, 2)

    assert isinstance(result, ShardResult)
    assert result.preingest.as_dict() == {
        "users": 4,
        "input_items": 60,
        "min_items_per_user": 10,
        "max_items_per_user": 20,
    }
