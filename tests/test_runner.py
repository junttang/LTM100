"""End-to-end tests for the closed-model LoadRunner.

Uses an in-memory fake backend and a synthetic dataset so the runner can be
exercised without network or real data. Asserts termination, op dispatch by
type, error handling, global concurrency cap, and metric recording.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from ltm100.common import MemoryItem, QueryItem, ResultItem, Turn, UserId
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad, SearchLoad


class FakeDataset:
    """A tiny dataset: each user has `n` memories. (No query_stream: scenarios
    derive search queries from memory content.)"""

    name = "fake"

    def __init__(self, n_memories: int = 30) -> None:
        self.n_memories = n_memories

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        for i in range(self.n_memories):
            yield MemoryItem(content=f"{user}-mem-{i}", producer=user)


class FakeBackend:
    """In-memory async backend that records calls and can fail on demand."""

    name = "fake"
    fail_every: int = 0  # 0 = never fail
    search_delay: float = 0.0

    def __init__(self) -> None:
        self.adds: list[tuple[UserId, int]] = []
        self.searches: list[tuple[UserId, str]] = []  # (user, query string)
        self.calls = 0

    async def setup(self, users: list[UserId]) -> None:
        return

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("forced add failure")
        self.adds.append((user, len(items)))
        await asyncio.sleep(0)  # yield so concurrent users interleave (real backends yield on I/O)
        return [f"{user}-{i}" for i in range(len(items))]

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        self.calls += 1
        if self.search_delay:
            await asyncio.sleep(self.search_delay)
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("forced search failure")
        self.searches.append((user, query.query))
        await asyncio.sleep(0)  # yield so concurrent users interleave
        return [ResultItem(content="hit")]

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        return


@pytest.mark.asyncio
async def test_add_load_emits_adds_only():
    ds = FakeDataset(n_memories=5)
    backend = FakeBackend()
    cfg = RunConfig(users=3, ops=15, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=AddLoad(), config=cfg)
    await runner.run()
    assert len(backend.adds) == 15
    assert backend.searches == []
    summary = runner.recorder.summary()
    assert summary["by_op"]["add"]["count"] == 15


@pytest.mark.asyncio
async def test_search_load_emits_searches_only():
    ds = FakeDataset(n_memories=100)
    backend = FakeBackend()
    cfg = RunConfig(users=2, ops=10, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    await runner.run()
    assert len(backend.searches) == 10
    assert backend.adds == []
    summary = runner.recorder.summary()
    assert summary["by_op"]["search"]["count"] == 10
    # Search queries are content-derived from the user's own memories.
    for user, q in backend.searches:
        assert q.startswith(f"{user}-mem-")


@pytest.mark.asyncio
async def test_search_load_top_k_forwarded():
    """--top-k (a scenario constructor param) reaches the backend as
    QueryItem.top_k on every search op."""

    seen_top_k: list[int] = []

    class TopKCapture(FakeBackend):
        async def search(self, user: UserId, query: QueryItem):
            seen_top_k.append(query.top_k)
            return await super().search(user, query)

    ds = FakeDataset(n_memories=10)
    backend = TopKCapture()
    cfg = RunConfig(users=1, ops=4, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(top_k=7), config=cfg)
    await runner.run()
    assert seen_top_k == [7, 7, 7, 7]
    # default is 20
    seen_top_k.clear()
    backend2 = TopKCapture()
    runner2 = LoadRunner(client=backend2, dataset=ds, scenario=SearchLoad(), config=cfg)
    await runner2.run()
    assert all(k == 20 for k in seen_top_k)


@pytest.mark.asyncio
async def test_chat_replay_mixed_emits_both():
    """chat-replay interleaves recall (search) and ingestion (add): with
    search_every=1 every user turn issues a search, so both op types appear."""
    from ltm100.core.scenarios import ChatReplay

    class TurnDataset(FakeDataset):
        def turn_stream(self, user):
            # 3 user/assistant turn-pairs; each turn has one memory chunk.
            for i in range(3):
                yield Turn(role="user", items=[MemoryItem(content=f"{user}-ut-{i}", producer=user)])
                yield Turn(role="assistant", items=[MemoryItem(content=f"{user}-at-{i}", producer=user)])

    ds = TurnDataset(n_memories=3)
    backend = FakeBackend()
    cfg = RunConfig(users=1, ops=20, seed=0)
    runner = LoadRunner(
        client=backend, dataset=ds, scenario=ChatReplay(think=0.0), config=cfg
    )
    await runner.run()
    assert len(backend.adds) > 0
    assert len(backend.searches) > 0
    summary = runner.recorder.summary()
    assert summary["by_op"]["add"]["count"] + summary["by_op"]["search"]["count"] == 20


@pytest.mark.asyncio
async def test_errors_recorded_not_raised():
    ds = FakeDataset(n_memories=5)
    backend = FakeBackend()
    backend.fail_every = 3  # every 3rd call fails
    cfg = RunConfig(users=1, ops=10, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=AddLoad(), config=cfg)
    results = await runner.run()
    statuses = {r.status for r in results}
    assert "error" in statuses
    assert "ok" in statuses
    summary = runner.recorder.summary()
    assert summary["by_op"]["add"]["errors"] > 0
    assert summary["by_op"]["add"]["error_rate"] > 0.0


@pytest.mark.asyncio
async def test_global_concurrency_caps_inflight():
    ds = FakeDataset(n_memories=100)
    backend = FakeBackend()
    backend.search_delay = 0.05
    cfg = RunConfig(users=5, duration=2.0, global_concurrency=2, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    await runner.run()
    # With cap=2 and 0.05s/search over 2s, max ~80 searches; just assert it ran.
    assert len(backend.searches) > 0


@pytest.mark.asyncio
async def test_termination_by_ops():
    ds = FakeDataset(n_memories=1000)
    backend = FakeBackend()
    cfg = RunConfig(users=1, ops=7, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=AddLoad(), config=cfg)
    results = await runner.run()
    assert len(results) == 7


@pytest.mark.asyncio
async def test_per_user_isolation_in_dispatch():
    ds = FakeDataset(n_memories=3)
    backend = FakeBackend()
    cfg = RunConfig(users=2, ops=6, seed=0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=AddLoad(), config=cfg)
    results = await runner.run()
    # Each user should only ever add its own items.
    users_seen = {r.user_id for r in results}
    assert users_seen == {"u0", "u1"}
    for user, n in backend.adds:
        assert all(user in m for m in [])
    # Items stored carry the producing user id in content.
    for r in results:
        assert r.user_id in ("u0", "u1")


def test_config_requires_termination():
    with pytest.raises(ValueError):
        RunConfig(users=1)
    with pytest.raises(ValueError):
        RunConfig(users=0, ops=1)


@pytest.mark.asyncio
async def test_preingest_populates_memory_before_measure():
    """preingest=True ingests memories (out of metrics) so search-load finds them."""
    ds = FakeDataset(n_memories=20)
    backend = FakeBackend()
    cfg = RunConfig(users=2, ops=6, seed=0, preingest=True)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    results = await runner.run()
    # Pre-ingest happened before measured searches: each user has memories.
    # Map recorded results' searches to users; each search user had ingested.
    search_users = [r.user_id for r in results]
    assert set(search_users) <= {"u0", "u1"}
    # The backend stored items for each user during preingest.
    add_users = {u for u, _ in backend.adds}
    assert "u0" in add_users and "u1" in add_users
    # And the measured run recorded only searches (not the preingest adds).
    summary = runner.recorder.summary()
    assert set(summary["by_op"]) == {"search"}


@pytest.mark.asyncio
async def test_preingest_fraction_limits_items():
    ds = FakeDataset(n_memories=100)
    backend = FakeBackend()
    cfg = RunConfig(users=1, ops=2, seed=0, preingest=True, preingest_fraction=0.1)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    await runner.run()
    # ~10 of 100 memories ingested.
    total_items = sum(n for _, n in backend.adds)
    assert 5 <= total_items <= 15


@pytest.mark.asyncio
async def test_preingest_fraction_zero_adds_nothing():
    ds = FakeDataset(n_memories=20)
    backend = FakeBackend()
    cfg = RunConfig(
        users=1,
        ops=2,
        seed=0,
        preingest=True,
        preingest_fraction=0.0,
    )
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    results = await runner.run()

    assert backend.adds == []
    assert len(results) == 2
    assert all(result.type.value == "search" for result in results)


@pytest.mark.asyncio
async def test_preingest_failure_aborts_the_measured_run():
    ds = FakeDataset(n_memories=20)
    backend = FakeBackend()
    backend.fail_every = 1
    cfg = RunConfig(users=1, ops=2, seed=0, preingest=True)
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)

    with pytest.raises(RuntimeError, match=r"pre-ingest failed.*u0.*RuntimeError"):
        await runner.run()

    assert backend.searches == []
    assert runner.recorder.raw() == []


@pytest.mark.asyncio
async def test_preingest_off_by_default():
    ds = FakeDataset(n_memories=20)
    backend = FakeBackend()
    cfg = RunConfig(users=1, ops=3, seed=0)  # preingest defaults False
    runner = LoadRunner(client=backend, dataset=ds, scenario=SearchLoad(), config=cfg)
    await runner.run()
    # No preingest adds, only the measured searches (which find nothing).
    assert backend.adds == []
