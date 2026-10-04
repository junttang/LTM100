"""Tests for multi-process measurement-window coordination."""

from __future__ import annotations

import asyncio
import time

import pytest

from ltm100.core.multiproc import ShardResult, run_shards
from ltm100.core.op import OpResult, OpType
from ltm100.core.runner import PreingestStats, SessionAdmissionStats


def _deadline_runner_entry(args: dict, proc_index: int) -> ShardResult:
    from test_duration_boundary import Backend, Dataset, Scenario

    from ltm100.core.config import RunConfig
    from ltm100.core.runner import LoadRunner

    class SlowBackend(Backend):
        async def add(self, user, items):
            self.starts.append(time.monotonic())
            await asyncio.sleep(0.08)
            return ["id"]

    async def run():
        backend = SlowBackend()
        runner = LoadRunner(
            client=backend,
            dataset=Dataset(),
            scenario=Scenario(),
            config=RunConfig(
                users=4,
                procs=2,
                proc_index=proc_index,
                duration=0.03,
                warmup=0.02,
                global_concurrency=1,
                queue_bound=10,
                model=args["model"],
                arrival_rate=1000,
                session_ops=2,
            ),
        )
        results = await runner.run()
        deadline = runner._start_time + runner.config.duration
        assert all(start < deadline for start in backend.starts)
        assert runner._admitted == 0
        assert runner._global_sem._value == 1
        assert sum(result.status == "ok" for result in results) == 1
        assert runner.measurement_ended_at >= max(result.ended_at for result in results)
        return ShardResult(
            results,
            runner.session_stats,
            measurement_started_at=runner.measurement_started_at,
            measurement_ended_at=runner.measurement_ended_at,
        )

    return asyncio.run(run())


@pytest.mark.parametrize("model", ["closed", "open"])
def test_duration_boundaries_hold_in_spawned_runners(model):
    pooled = run_shards(_deadline_runner_entry, {"model": model}, 2)
    assert isinstance(pooled, ShardResult)
    assert sum(result.status == "ok" for result in pooled.results) == 2
    assert all(
        result.ended_at <= pooled.measurement_ended_at for result in pooled.results
    )


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


def _measurement_window_entry(args: dict, proc_index: int) -> ShardResult:
    return ShardResult(
        [],
        SessionAdmissionStats(),
        measurement_started_at=100.0 + proc_index,
        measurement_ended_at=110.0 + proc_index,
    )


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


def test_measurement_window_spans_every_shard():
    result = run_shards(_measurement_window_entry, {}, 2)

    assert isinstance(result, ShardResult)
    assert result.measurement_started_at == 100.0
    assert result.measurement_ended_at == 111.0
