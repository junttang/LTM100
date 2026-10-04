"""Deadline regressions: stop dispatch, drain requests, and retain count budgets."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from ltm100.common import MemoryItem, ResultItem
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpType
from ltm100.core.runner import LoadRunner


class Dataset:
    name = "deadline-test"

    def users(self, n_users, *, seed=0):
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user):
        yield MemoryItem(content=user)


class Scenario:
    def __init__(self, delay=0.0):
        self.delay = delay

    def plan(self, user, dataset, rng_state):
        while True:
            yield Op(OpType.ADD, items=[MemoryItem(content=user)], delay=self.delay)


class Backend:
    name = "deadline-test"

    def __init__(self):
        self.starts = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.gated = False

    async def setup(self, users):
        pass

    async def add(self, user, items):
        self.starts.append(time.monotonic())
        self.entered.set()
        if self.gated:
            await self.release.wait()
        await asyncio.sleep(0)
        return ["id"]

    async def search(self, user, query):
        await self.add(user, [])
        return [ResultItem(content="hit")]


def make_runner(*, model="closed", delay=0.0, **kwargs):
    config = RunConfig(
        users=2,
        duration=0.05,
        model=model,
        arrival_rate=100,
        session_ops=1,
        **kwargs,
    )
    return LoadRunner(
        client=Backend(), dataset=Dataset(), scenario=Scenario(delay), config=config
    )


async def session(runner, deadline, user="u0", active=None):
    if runner.config.model == "closed":
        await runner._user_loop(user, 0, deadline)
    else:
        await runner._open_session(
            user,
            deadline,
            session_id=None,
            session_count=None,
            active_sessions=active or {user: set()},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_delay_reaching_deadline_does_not_dispatch(model):
    runner = make_runner(model=model, delay=10)
    await session(runner, time.monotonic() + 0.02)
    assert runner.client.starts == []
    assert runner.recorder.raw() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_queued_request_expires_while_running_request_drains(model):
    runner = make_runner(model=model, global_concurrency=1, queue_bound=1)
    runner._global_sem = asyncio.Semaphore(1)
    runner.client.gated = True
    deadline = time.monotonic() + 0.04
    running = asyncio.create_task(session(runner, deadline))
    await asyncio.wait_for(runner.client.entered.wait(), 1)
    queued = asyncio.create_task(session(runner, deadline, "u1"))
    try:
        await asyncio.sleep(0.07)
        expired_before_drain = queued.done()
        assert not running.done()
    finally:
        runner.client.release.set()
        await asyncio.wait_for(asyncio.gather(running, queued), 1)
    assert len(runner.client.starts) == 1
    assert expired_before_drain
    assert len(runner.recorder.raw()) == 1
    assert runner.recorder.raw()[0].status == "ok"
    assert runner._global_sem._value == 1
    assert runner._admitted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_slot_acquisition_crossing_deadline_releases_permit(model, monkeypatch):
    import ltm100.core.runner as module

    clock = [0.0]
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )

    class LateSemaphore(asyncio.Semaphore):
        async def acquire(self):
            await super().acquire()
            clock[0] = 2.0
            return True

    runner = make_runner(model=model, global_concurrency=1, queue_bound=1)
    runner._global_sem = LateSemaphore(1)
    await session(runner, 1.0)
    assert runner.client.starts == []
    assert runner.recorder.raw() == []
    assert runner._global_sem._value == 1
    assert runner._admitted == 0


@pytest.mark.asyncio
async def test_open_arrival_waking_after_deadline_is_not_admitted(monkeypatch):
    import ltm100.core.runner as module

    clock = [0.0]
    real_sleep = asyncio.sleep

    async def oversleep(delay):
        clock[0] = 2.0
        await real_sleep(0)

    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )
    monkeypatch.setattr(module.asyncio, "sleep", oversleep)
    runner = make_runner(model="open")
    await runner._open_loop(["u0"], 1.0)
    assert runner.session_stats.offered == 0
    assert runner.client.starts == []


@pytest.mark.asyncio
async def test_rampup_does_not_outlive_duration():
    runner = make_runner()
    deadline = time.monotonic() + 0.02
    task = asyncio.create_task(runner._user_loop("u0", 10, deadline))
    try:
        await asyncio.wait_for(task, 0.3)
    finally:
        task.cancel()
    assert runner.client.starts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 3, 17])
async def test_count_only_runs_execute_every_reserved_operation(budget):
    runner = LoadRunner(
        client=Backend(),
        dataset=Dataset(),
        scenario=Scenario(delay=0.001),
        config=RunConfig(users=12, ops=budget, global_concurrency=1),
    )
    results = await runner.run()
    assert len(results) == budget
    assert len(runner.client.starts) == budget
    assert runner._global_sem._value == 1


@pytest.mark.asyncio
async def test_duration_takes_precedence_over_reserved_count_budget():
    runner = make_runner(delay=10, ops=10)
    assert await runner.run() == []
    assert runner.client.starts == []


@pytest.mark.asyncio
async def test_warmup_expiry_does_not_discard_measured_count_budget():
    runner = LoadRunner(
        client=Backend(),
        dataset=Dataset(),
        scenario=Scenario(delay=0.02),
        config=RunConfig(users=4, ops=7, warmup=0.005, global_concurrency=1),
    )
    results = await runner.run()
    assert len(results) == 7
    assert len(runner.client.starts) == 7
    assert runner._admitted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_cancelled_queue_waiter_releases_reservations(model):
    runner = make_runner(model=model, global_concurrency=1, queue_bound=1)
    runner._global_sem = asyncio.Semaphore(0)
    waiting = asyncio.create_task(session(runner, time.monotonic() + 10))
    await asyncio.sleep(0)
    if model == "open":
        assert runner._admitted == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert runner._admitted == 0
    assert runner._global_sem._value == 0
    assert runner.client.starts == []


@pytest.mark.asyncio
async def test_expired_open_session_releases_profile_lane_without_rejection():
    runner = make_runner(model="open", global_concurrency=1, queue_bound=1)
    runner._global_sem = asyncio.Semaphore(0)
    active = {"u0": {0}}
    await runner._open_session(
        "u0",
        time.monotonic() + 0.02,
        session_id=0,
        session_count=1,
        active_sessions=active,
    )
    assert active == {"u0": set()}
    assert runner._admitted == 0
    assert runner.recorder.raw() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_operation_preparation_crossing_deadline_does_not_dispatch(
    model, monkeypatch
):
    import ltm100.core.runner as module

    clock = [0.0]
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )

    class SlowPlan:
        def plan(self, user, dataset, rng_state):
            clock[0] = 2.0
            yield Op(OpType.ADD, items=[MemoryItem(content=user)])

    runner = make_runner(model=model)
    runner.scenario = SlowPlan()
    await session(runner, 1.0)
    assert runner.recorder.raw() == []
    assert runner.client.starts == []
