"""Tests for the chat-replay scenario (chatbot-with-LTM workload replay)."""

from __future__ import annotations

import asyncio

import pytest

from ltm100.common import MemoryItem, QueryItem, ResultItem, Turn, UserId
from ltm100.core.config import RunConfig
from ltm100.core.op import OpType
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import ChatReplay


class DialogueDataset:
    """Dataset exposing a turn_stream of user/assistant turns."""

    name = "dialogue"

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user: UserId):
        # Not used by chat-replay, but part of the contract.
        for i in range(3):
            yield MemoryItem(content=f"{user}-mem-{i}", producer=user)

    def query_stream(self, user: UserId):
        # Not used by chat-replay.
        yield QueryItem(query=f"{user}-q")

    def turn_stream(self, user: UserId):
        for i in range(3):
            yield Turn(
                role="user",
                items=[MemoryItem(content=f"{user} user turn {i}", producer=user)],
            )
            yield Turn(
                role="assistant",
                items=[MemoryItem(content=f"{user} assistant turn {i}", producer=user)],
            )


class FlatDataset:
    """Dataset with no turn_stream (e.g. synthetic-like)."""

    name = "flat"

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user: UserId):
        for i in range(3):
            yield MemoryItem(content=f"{user}-mem-{i}", producer=user)

    def query_stream(self, user: UserId):
        yield QueryItem(query=f"{user}-q")


class RecordingBackend:
    name = "record"

    def __init__(self) -> None:
        self.ops: list[tuple[str, str]] = []

    async def setup(self, users):
        return

    async def add(self, user, items):
        self.ops.append((user, "add"))
        return [f"{user}-{i}" for i in range(len(items))]

    async def search(self, user, query):
        self.ops.append((user, "search"))
        return [ResultItem(content="hit")]

    async def teardown(self, users, *, delete):
        return


def _ops_for_user(runner: LoadRunner, user: UserId) -> list[str]:
    return [r.type.value for r in runner.recorder.raw() if r.user_id == user]


@pytest.mark.asyncio
async def test_chat_replay_search_before_user_turn_then_adds():
    ds = DialogueDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, ops=9, seed=0)  # 3 turn-pairs: search + user add + assistant add
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await runner.run()
    seq = _ops_for_user(runner, "u0")
    # Per turn-pair: user turn -> search then add; assistant turn -> add only.
    assert seq == [
        "search", "add", "add",  # user turn 0 + assistant turn 0
        "search", "add", "add",  # user turn 1 + assistant turn 1
        "search", "add", "add",  # user turn 2 + assistant turn 2
    ]


@pytest.mark.asyncio
async def test_chat_replay_search_query_is_user_turn_content():
    """The recall query is derived from the upcoming user turn's content."""
    seen_queries: list[str] = []

    class QueryCapture(RecordingBackend):
        async def search(self, user, query):
            seen_queries.append(query.query)
            return await super().search(user, query)

    ds = DialogueDataset()
    backend = QueryCapture()
    cfg = RunConfig(users=1, ops=9, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await runner.run()
    assert seen_queries == ["u0 user turn 0", "u0 user turn 1", "u0 user turn 2"]


@pytest.mark.asyncio
async def test_chat_replay_terminates_when_stream_exhausted():
    ds = DialogueDataset()
    backend = RecordingBackend()
    # duration-based; the plan is finite (6 turns = 9 ops) so it ends early.
    cfg = RunConfig(users=1, duration=5.0, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await asyncio.wait_for(runner.run(), timeout=10.0)
    seq = _ops_for_user(runner, "u0")
    assert seq == [
        "search", "add", "add",
        "search", "add", "add",
        "search", "add", "add",
    ]


@pytest.mark.asyncio
async def test_chat_replay_requires_turn_stream():
    ds = FlatDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, ops=6, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(), config=cfg)
    with pytest.raises(ValueError, match="turn_stream"):
        await runner.run()
