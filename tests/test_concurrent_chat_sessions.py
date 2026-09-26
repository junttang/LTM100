"""Concurrent-session execution tests for profiled chat replay."""

from __future__ import annotations

import asyncio

import pytest

from ltm100.common import MemoryItem, ResultItem, Turn, UserId
from ltm100.core.chat_profile import ChatGroup, ChatProfile, ChatSettings
from ltm100.core.config import RunConfig
from ltm100.core.multiproc import run_shards
from ltm100.core.op import OpType
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import ChatReplay


class SessionDataset:
    name = "sessions"

    def __init__(self, sessions: int = 5) -> None:
        self.sessions = sessions

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{index}" for index in range(n_users)]

    def memory_stream(self, user: UserId):
        for session in self.session_stream(user):
            for turn in session:
                yield from turn.items

    def turn_stream(self, user: UserId):
        for session in self.session_stream(user):
            yield from session

    def session_stream(self, user: UserId):
        for index in range(self.sessions):
            yield [
                Turn(
                    role="user",
                    items=[MemoryItem(content=f"{user} session {index} question")],
                ),
                Turn(
                    role="assistant",
                    items=[MemoryItem(content=f"{user} session {index} answer")],
                ),
            ]


class ConcurrentBackend:
    name = "concurrent"

    def __init__(self, delay: float = 0.005) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.users: set[str] = set()
        self.queries: list[str] = []
        self.setup_called = False

    async def setup(self, users):
        self.setup_called = True

    async def _enter(self, user: str) -> None:
        self.users.add(user)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1

    async def add(self, user, items):
        await self._enter(user)
        return ["id"]

    async def search(self, user, query):
        self.queries.append(query.query)
        await self._enter(user)
        return [ResultItem(content="hit")]

    async def teardown(self, users, *, delete):
        return


def _profile(concurrent_sessions: int, *, top_k: int = 20) -> ChatProfile:
    return ChatProfile(
        [
            ChatGroup(
                name="test",
                share=1.0,
                settings=ChatSettings(
                    think=0.0,
                    search_every=1,
                    answer_time=0.0,
                    user_gap=0.0,
                    top_k=top_k,
                    concurrent_sessions=concurrent_sessions,
                ),
            )
        ]
    )


def _concurrent_shard_entry(args: dict, proc_index: int):
    runner = LoadRunner(
        client=ConcurrentBackend(delay=0.001),
        dataset=SessionDataset(sessions=2),
        scenario=ChatReplay(profile=_profile(2)),
        config=RunConfig(
            users=2,
            ops=6,
            seed=args["seed"],
            procs=2,
            proc_index=proc_index,
        ),
    )
    return asyncio.run(runner.run())


@pytest.mark.asyncio
async def test_sessions_overlap_but_each_session_preserves_operation_order():
    backend = ConcurrentBackend()
    runner = LoadRunner(
        client=backend,
        dataset=SessionDataset(sessions=5),
        scenario=ChatReplay(profile=_profile(5)),
        config=RunConfig(users=1, ops=15, seed=0),
    )

    results = await runner.run()

    assert len(results) == 15
    assert backend.max_active == 5
    assert backend.users == {"u0"}
    assert {result.session_id for result in results} == {0, 1, 2, 3, 4}
    for session_id in range(5):
        session_ops = [
            result.type
            for result in results
            if result.session_id == session_id
        ]
        assert session_ops == [OpType.SEARCH, OpType.ADD, OpType.ADD]


@pytest.mark.asyncio
async def test_global_concurrency_remains_the_final_request_cap():
    backend = ConcurrentBackend()
    runner = LoadRunner(
        client=backend,
        dataset=SessionDataset(sessions=5),
        scenario=ChatReplay(profile=_profile(5)),
        config=RunConfig(users=1, ops=15, seed=0, global_concurrency=2),
    )

    await runner.run()

    assert backend.max_active == 2


def test_session_lanes_cover_disjoint_source_conversations():
    dataset = SessionDataset(sessions=7)
    profile = _profile(2)
    scenario = ChatReplay(profile=profile)
    profile.assign(["u0"], seed=0)
    scenario.validate_run(dataset, ["u0"], model="closed")

    lane_zero = scenario.plan(
        "u0", dataset, {"seed": 0, "user": "u0", "session_id": 0}
    )
    lane_one = scenario.plan(
        "u0", dataset, {"seed": 0, "user": "u0", "session_id": 1}
    )
    queries_zero = [
        next(op for op in lane_zero if op.type is OpType.SEARCH).query.query
        for _ in range(4)
    ]
    queries_one = [
        next(op for op in lane_one if op.type is OpType.SEARCH).query.query
        for _ in range(3)
    ]

    assert queries_zero == [
        "u0 session 0 question",
        "u0 session 2 question",
        "u0 session 4 question",
        "u0 session 6 question",
    ]
    assert queries_one == [
        "u0 session 1 question",
        "u0 session 3 question",
        "u0 session 5 question",
    ]


@pytest.mark.asyncio
async def test_insufficient_source_sessions_fail_before_backend_setup():
    backend = ConcurrentBackend()
    runner = LoadRunner(
        client=backend,
        dataset=SessionDataset(sessions=2),
        scenario=ChatReplay(profile=_profile(5)),
        config=RunConfig(users=1, ops=3),
    )

    with pytest.raises(ValueError, match="provides only 2"):
        await runner.run()
    assert backend.setup_called is False


@pytest.mark.asyncio
async def test_open_model_rejects_profiled_concurrent_sessions():
    runner = LoadRunner(
        client=ConcurrentBackend(),
        dataset=SessionDataset(sessions=5),
        scenario=ChatReplay(profile=_profile(2)),
        config=RunConfig(
            users=1,
            duration=0.1,
            model="open",
            arrival_rate=10.0,
            session_ops=3,
        ),
    )

    with pytest.raises(ValueError, match="requires the closed load model"):
        await runner.run()


@pytest.mark.asyncio
async def test_warmup_does_not_consume_concurrent_session_operation_budget():
    backend = ConcurrentBackend(delay=0.002)
    runner = LoadRunner(
        client=backend,
        dataset=SessionDataset(sessions=2),
        scenario=ChatReplay(profile=_profile(2)),
        config=RunConfig(users=1, ops=6, warmup=0.02),
    )

    results = await runner.run()

    assert len(results) == 6
    assert {result.session_id for result in results} == {0, 1}


def test_concurrent_sessions_run_in_spawned_process_shards():
    results = run_shards(_concurrent_shard_entry, {"seed": 7}, 2)

    assert len(results) == 12
    assert {result.user_id for result in results} == {"u0", "u1"}
    for user in ("u0", "u1"):
        assert {
            result.session_id for result in results if result.user_id == user
        } == {0, 1}
