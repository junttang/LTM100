"""Scenarios for the closed and open load models.

Each Scenario turns a per-user dataset stream into a plan of Ops with delays.
The closed model loops back-to-back (delay=0) unless a scenario adds think
time; the open model consumes a bounded number of ops per arriving session
(see the runner's `_open_session`), so an open scenario's plan is infinite.

Scenarios here:
  - AddLoad:        stream all memories back-to-back, then stop. (closed)
  - SearchLoad:     ingest is assumed pre-done (warm-up); loop the query
                    stream repeatedly until the runner stops us. (closed)
  - AddSearchMixed: interleave add and search from the same user's streams. (closed)
  - Realistic:      infinite search-weighted op stream with inter-arrival
                    jitter, intended for the open model. The op mix (search
                    vs add) is owned here via `search_weight`, not by the
                    runner — open and closed share one Scenario interface.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from ltm100.common import DatasetAdapter, QueryItem, UserId
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


class Realistic:
    """Infinite search-weighted op stream for the open model.

    Each arriving user session (see the runner's `_open_session`) consumes up
    to `session_ops` ops from this plan then leaves. Per-op, a SEARCH is drawn
    with probability `search_weight`, else an ADD of one memory item. A small
    think-time gap is attached as `Op.delay` so sessions are not perfectly
    back-to-back; the open model's inter-arrival is driven separately by the
    Poisson generator in the runner.

    The op mix is owned by this scenario (via `search_weight`), not by a
    runner-level weight, so the open and closed models share one Scenario
    interface. Memories are pulled from `memory_stream`; once a user's stream
    is exhausted, adds reuse earlier memories cyclically (an arriving session
    is a returning user who has memories to re-add).
    """

    name = "realistic"

    def __init__(self, search_weight: float = 0.8, think: float = 0.05) -> None:
        if not 0.0 <= search_weight <= 1.0:
            raise ValueError("search_weight must be in [0, 1]")
        self.search_weight = search_weight
        self.think = think

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        rng = random.Random(_seed_for(rng_state.get("seed", 0), user))
        queries = list(dataset.query_stream(user))
        memories = list(dataset.memory_stream(user))
        if not queries and not memories:
            return
        i_add = 0
        i_q = 0
        # Infinite: the open runner bounds consumption per session.
        while True:
            if memories and (not queries or rng.random() < self.search_weight):
                q = queries[i_q % len(queries)]
                i_q += 1
                delay = rng.uniform(0.0, self.think)
                yield Op(type=OpType.SEARCH, query=q, delay=delay)
            else:
                item = memories[i_add % len(memories)]
                i_add += 1
                yield Op(type=OpType.ADD, items=[item], delay=0.0)


class ChatReplay:
    """Replay a chatbot-with-LTM workload over a multi-turn dialogue.

    Models the real integration pattern: a chatbot recalls relevant memories
    before answering a user, then ingests the conversation turn. The plan
    walks the dataset's structured `turn_stream` (user/assistant turns in
    order). For every turn:

      - if the turn is a **user** turn: first issue a SEARCH whose query is the
        user turn's content (the recall step), then ADD the turn's items;
      - otherwise (assistant turn): just ADD the turn's items.

    Adds and the recall search are interleaved exactly as a live chatbot
    session would interleave them. The search query is derived from the
    upcoming user turn (not the dataset's separate evaluation question), so
    recall is driven by the conversation itself.

    Requires a dataset adapter that implements `turn_stream` (LongMemEval
    does). Closed model; the turn stream is finite, so the user stops when
    the conversation is replayed (or on duration/ops). `query_stream` is not
    used by this scenario.

    A small `think` delay between ops mimics the user/assistant think time so
    users drift out of lockstep.
    """

    name = "chat-replay"

    def __init__(self, think: float = 0.05) -> None:
        self.think = think

    def validate(self, dataset: DatasetAdapter) -> None:
        if not hasattr(dataset, "turn_stream"):
            raise ValueError(
                "chat-replay requires a dataset with a turn_stream "
                "(e.g. LongMemEval); the configured dataset does not provide one"
            )

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        return self._plan(user, dataset, rng_state)

    def _plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        rng = random.Random(_seed_for(rng_state.get("seed", 0), user))
        for turn in dataset.turn_stream(user):
            is_user = turn.role.lower() == "user"
            if is_user and turn.items:
                # Recall before answering: query is the user turn's content.
                first_content = turn.items[0].content
                yield Op(
                    type=OpType.SEARCH,
                    query=QueryItem(query=first_content, top_k=20),
                    delay=rng.uniform(0.0, self.think),
                )
            for item in turn.items:
                yield Op(type=OpType.ADD, items=[item], delay=rng.uniform(0.0, self.think))


SCENARIOS: dict[str, type] = {
    AddLoad.name: AddLoad,
    SearchLoad.name: SearchLoad,
    AddSearchMixed.name: AddSearchMixed,
    Realistic.name: Realistic,
    ChatReplay.name: ChatReplay,
}


def get_scenario(name: str, **kwargs: Any) -> Scenario:
    cls = SCENARIOS.get(name)
    if cls is None:
        raise ValueError(f"unknown scenario: {name}")
    return cls(**kwargs)  # type: ignore[return-value]


__all__ = [
    "AddLoad",
    "SearchLoad",
    "AddSearchMixed",
    "Realistic",
    "ChatReplay",
    "SCENARIOS",
    "get_scenario",
]
