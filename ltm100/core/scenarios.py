"""Initial scenarios (closed model).

Each Scenario turns a per-user dataset stream into a plan of Ops with delays.
The closed model loops back-to-back (delay=0) unless a scenario adds think time.

Scenarios here:
  - AddLoad:        stream all memories back-to-back, then stop.
  - SearchLoad:     ingest is assumed pre-done (warm-up); loop the query
                    stream repeatedly until the runner stops us.
  - AddSearchMixed: interleave add and search from the same user's streams.

Realistic (open model) and open-model congestion policy are deferred.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from ltm100.common import DatasetAdapter, UserId
from ltm100.core.op import Op, OpType, Scenario


def _seed_for(seed: int, user: str) -> int:
    """Combine the run seed and user id into a stable int seed.

    Avoids passing a str/tuple to random.Random, which Python 3.9+ warns
    about (hash-based seeding)."""
    h = (seed * 1_000_003) & 0xFFFFFFFF
    for ch in user.encode("utf-8"):
        h = (h ^ ch) * 1_000_003 & 0xFFFFFFFF
    return h


class AddLoad:
    """Pure storage throughput: each user adds its memory stream back-to-back."""

    name = "add-load"

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        for item in dataset.memory_stream(user):
            yield Op(type=OpType.ADD, items=[item], delay=0.0)


class SearchLoad:
    """Pure search throughput: each user loops its query stream repeatedly.

    Assumes memories were pre-ingested (warm-up). The plan is infinite; the
    runner terminates it via duration/ops. To avoid every user issuing the
    exact same query in lockstep, the first query is jittered by an op index
    drawn from the seeded state.
    """

    name = "search-load"
    max_iterations = 10_000

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        rng = random.Random(_seed_for(rng_state.get("seed", 0), user))
        queries = list(dataset.query_stream(user))
        if not queries:
            return
        for _ in range(self.max_iterations):
            for q in queries:
                # Small think time so users drift out of lockstep.
                delay = rng.uniform(0.0, 0.02)
                yield Op(type=OpType.SEARCH, query=q, delay=delay)


class AddSearchMixed:
    """Interleaved add and search per user.

    The user ingests its memory stream, but after every `search_every` adds,
    it issues one search from its query stream. Adds and searches are thus
    mixed within one user's lifetime. If the query stream is empty, this
    degenerates to add-load.
    """

    name = "add-search-mixed"

    def __init__(self, search_every: int = 20, add_batch: int = 1) -> None:
        self.search_every = max(1, search_every)
        self.add_batch = max(1, add_batch)

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        queries = list(dataset.query_stream(user))
        q_iter = (queries[i % len(queries)] for i in range(10_000 * len(queries) or 1))
        adds_since_search = 0
        batch: list = []
        for item in dataset.memory_stream(user):
            batch.append(item)
            if len(batch) >= self.add_batch:
                yield Op(type=OpType.ADD, items=batch, delay=0.0)
                batch = []
                adds_since_search += 1
                if queries and adds_since_search % self.search_every == 0:
                    yield Op(type=OpType.SEARCH, query=next(q_iter), delay=0.0)
        if batch:
            yield Op(type=OpType.ADD, items=batch, delay=0.0)


SCENARIOS: dict[str, type] = {
    AddLoad.name: AddLoad,
    SearchLoad.name: SearchLoad,
    AddSearchMixed.name: AddSearchMixed,
}


def get_scenario(name: str) -> Scenario:
    cls = SCENARIOS.get(name)
    if cls is None:
        raise ValueError(f"unknown scenario: {name}")
    return cls()  # type: ignore[return-value]


__all__ = ["AddLoad", "SearchLoad", "AddSearchMixed", "SCENARIOS", "get_scenario"]
