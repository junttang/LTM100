"""Tests for the chat-replay scenario (chatbot-with-LTM workload replay)."""

from __future__ import annotations

import asyncio

import pytest

from ltm100.common import MemoryItem, ResultItem, Turn, UserId
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
async def test_chat_replay_count_terminates_at_ops():
    """One pass of the dialogue is 3 turn-pairs = 9 ops; with ops=9 the run
    stops exactly after one replay."""
    ds = DialogueDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, ops=9, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await runner.run()
    seq = _ops_for_user(runner, "u0")
    assert seq == [
        "search", "add", "add",
        "search", "add", "add",
        "search", "add", "add",
    ]


@pytest.mark.asyncio
async def test_chat_replay_search_every_2():
    """With search_every=2, only every 2nd user turn triggers a recall search.
    The per-pass counter resets each pass: turns 0 and 2 search, turn 1 does
    not. One pass = 2 searches + 6 adds = 8 ops."""
    ds = DialogueDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, ops=8, seed=0)
    runner = LoadRunner(
        client=backend, dataset=ds, scenario=ChatReplay(think=0.0, search_every=2), config=cfg
    )
    await runner.run()
    seq = _ops_for_user(runner, "u0")
    # turn0: search + user add + assistant add
    # turn1: (no search) + user add + assistant add
    # turn2: search + user add + assistant add
    assert seq == ["search", "add", "add", "add", "add", "search", "add", "add"]


@pytest.mark.asyncio
async def test_chat_replay_wraps_over_duration():
    """The turn stream wraps, so a duration longer than one replay emits more
    than the 9 ops of a single pass."""
    ds = DialogueDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, duration=2.0, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await asyncio.wait_for(runner.run(), timeout=10.0)
    seq = _ops_for_user(runner, "u0")
    assert len(seq) > 9  # the conversation replayed more than once


@pytest.mark.asyncio
async def test_chat_replay_runs_under_open_model():
    """chat-replay works under the open model: Poisson-arriving sessions each
    replay a bounded slice of the conversation. Both op types appear."""
    ds = DialogueDataset()
    backend = RecordingBackend()
    cfg = RunConfig(
        users=2,
        duration=1.5,
        seed=0,
        model="open",
        arrival_rate=20.0,
        session_ops=6,
    )
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg)
    await asyncio.wait_for(runner.run(), timeout=10.0)
    summary = runner.recorder.summary()
    assert summary["total"] > 0
    assert "search" in summary["by_op"]
    assert "add" in summary["by_op"]


@pytest.mark.asyncio
async def test_chat_replay_requires_turn_stream():
    ds = FlatDataset()
    backend = RecordingBackend()
    cfg = RunConfig(users=1, ops=6, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=ChatReplay(), config=cfg)
    with pytest.raises(ValueError, match="turn_stream"):
        await runner.run()


@pytest.mark.asyncio
async def test_chat_replay_answer_time_and_user_gap_add_idle():
    """answer_time attaches a delay after a user turn's last add (LLM answer
    time); user_gap attaches a delay before a user turn's first op (user
    typing time), except the pass's first turn. With both set, the total
    per-pass delay budget grows by ~3*answer_time + 2*user_gap (3 user turns
    => 3 answer_times; user_gap skipped on the first turn => 2 user_gaps).
    Compared deterministically via the plan's Op delays (seeded, not via
    wall-clock, to avoid Exponential-draw flakiness)."""
    import itertools

    ds = DialogueDataset()

    def total_delay(answer_time: float, user_gap: float) -> float:
        plan = ChatReplay(
            think=0.0, answer_time=answer_time, user_gap=user_gap
        ).plan("u0", ds, {"seed": 0})
        return sum(op.delay for op in itertools.islice(plan, 9))

    baseline = total_delay(answer_time=0.0, user_gap=0.0)
    with_gaps = total_delay(answer_time=1.0, user_gap=1.0)
    # Expected extra idle = 3*1.0 + 2*1.0 = 5.0 (Exp means); assert it is
    # substantial and clearly above the zero-gaps baseline.
    assert with_gaps > baseline + 2.0


@pytest.mark.asyncio
async def test_chat_replay_answer_time_gaps_after_user_turn_only():
    """answer_time lands on a user turn's last add (the assistant answer
    follows during that gap); user_gap before a user turn's first op but not
    the pass's first turn. Assistant turns get neither. Inspect the raw Op
    delays from the plan (checks placement + that the delays are the Exponential
    draws, aggregated to avoid single-draw flakiness)."""
    import itertools

    ds = DialogueDataset()
    # 3 user turns interleaved with 3 assistant turns (u,a,u,a,u,a).
    plan = ChatReplay(think=0.0, answer_time=2.0, user_gap=3.0).plan(
        "u0", ds, {"seed": 0}
    )
    ops = list(itertools.islice(plan, 9))  # one pass: 3 turn-pairs

    # Placement is deterministic: each turn yields 1 op here (turn0 user has a
    # search + add = 2 ops, then turns 1..5 have 1 op each = 7, total 9).
    # Layout: [search, add,  add, search, add,  add, search, add,  add]
    #           u0    u0a   a1    u2       u2a  a3    u4       u4a  a5
    types = [op.type.value for op in ops]
    assert types == [
        "search", "add", "add", "search", "add", "add", "search", "add", "add"
    ]

    # First user turn's first op (search) has no user_gap; assistant adds have
    # neither gap. These three must be ~0 (think=0).
    assert ops[0].delay < 0.05  # u0 search: first turn, no user_gap
    assert ops[2].delay < 0.05  # a1 add: assistant, no gap
    assert ops[5].delay < 0.05  # a3 add: assistant, no gap

    # The answer_time draws (3 of them, mean 2.0 each) land on the user-turn
    # adds: ops[1], ops[4], ops[7]. Their sum is ~6 in expectation; assert the
    # aggregate is substantial (single Exponential draws vary, but their sum
    # is robustly large).
    answer_total = ops[1].delay + ops[4].delay + ops[7].delay
    assert answer_total > 2.0
    # The user_gap draws (2 of them, mean 3.0 each) land on the non-first user
    # turns' first ops: ops[3] (u2 search) and ops[6] (u4 search).
    user_gap_total = ops[3].delay + ops[6].delay
    assert user_gap_total > 2.0
    # Sanity: the assistant adds (ops[2], ops[5], ops[8]) carry no gap.
    assert ops[8].delay < 0.05  # a5 add: assistant, no gap


@pytest.mark.asyncio
async def test_chat_replay_rejects_negative_timing_params():
    with pytest.raises(ValueError, match="answer_time"):
        ChatReplay(answer_time=-0.1)
    with pytest.raises(ValueError, match="user_gap"):
        ChatReplay(user_gap=-0.1)


@pytest.mark.asyncio
async def test_chat_replay_top_k_forwarded_to_recall_queries():
    """--top-k controls the recall search depth: the QueryItem.top_k of every
    recall SEARCH reflects the configured top_k (default 20, overridable)."""
    import itertools

    ds = DialogueDataset()
    # default -> 20
    plan = ChatReplay(think=0.0).plan("u0", ds, {"seed": 0})
    default_op = next(op for op in plan if op.type == OpType.SEARCH)
    assert default_op.query.top_k == 20
    # overridden -> 5
    plan = ChatReplay(think=0.0, top_k=5).plan("u0", ds, {"seed": 0})
    override_op = next(op for op in plan if op.type == OpType.SEARCH)
    assert override_op.query.top_k == 5


@pytest.mark.asyncio
async def test_chat_replay_rejects_nonpositive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        ChatReplay(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        ChatReplay(top_k=-1)
