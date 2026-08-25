"""Scenarios for the closed and open load models.

Each Scenario turns a per-user dataset stream into a plan of Ops with delays.
The closed model loops back-to-back (delay=0) unless a scenario adds think
time; the open model consumes a bounded number of ops per arriving session
(see the runner's `_open_session`). Every scenario's plan is **infinite**
(it wraps its underlying stream), so a duration run sustains load instead of
going idle once a finite stream is exhausted; the runner bounds consumption
via duration/ops (closed) or session_ops (open).

Search queries are **content-derived**: built from the user's own
`memory_stream` items (one query per stored unit), so the query pool is as
large as the memory stream and cycling it does not naively repeat a single
query (which would warm a server result cache and understate latency).
`chat-replay` instead derives each recall query from the upcoming user turn's
content. No scenario consumes a separate evaluation `query_stream`.

Scenarios here:
  - ChatReplay:   the primary workload — replay a chatbot-with-LTM integration
                  over the dataset's turn_stream (recall before a user turn,
                  then ingest the turn), with a configurable recall cadence.
                  (closed/open)
  - AddLoad:      infinite add stream over the user's memory_stream. (closed/open)
  - SearchLoad:   infinite content-derived search stream; assumes pre-ingest. (closed/open)
  - Mixed:        controllable add/search mixture (op mix via `search_weight`),
                  with think jitter. A simple, no-dialogue workload usable with
                  any dataset (incl. synthetic) — handy for quick congestion
                  probing. (closed/open)
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Iterator

from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, UserId
from ltm100.core.op import Op, OpType, Scenario


def _seed_for(seed: int, user: str) -> int:
    """Combine the run seed and user id into a stable int seed.

    Avoids passing a str/tuple to random.Random, which Python 3.9+ warns
    about (hash-based seeding)."""
    h = (seed * 1_000_003) & 0xFFFFFFFF
    for ch in user.encode("utf-8"):
        h = (h ^ ch) * 1_000_003 & 0xFFFFFFFF
    return h


def _content_queries(memories: list[MemoryItem]) -> list[QueryItem]:
    """Content-derived search queries, one per memory item.

    The pool is as large as the memory stream, so cycling it does not
    naively repeat a single query (which would warm a server result cache and
    understate latency). Each query is the memory item's own content."""
    return [QueryItem(query=m.content, top_k=20) for m in memories]


class AddLoad:
    """Pure storage throughput: each user adds its memory stream forever.

    The plan wraps the memory stream, so a duration run sustains add load
    until the runner stops it (count- or time-based)."""

    name = "add-load"

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        memories = list(dataset.memory_stream(user))
        if not memories:
            return
        i = 0
        while True:
            yield Op(type=OpType.ADD, items=[memories[i % len(memories)]], delay=0.0)
            i += 1


class SearchLoad:
    """Pure search throughput: each user searches content-derived queries forever.

    Assumes memories were pre-ingested (warm-up), so searches run against the
    user's own stored content. The query pool is built from the user's
    `memory_stream`; each pass through the pool starts at a rotating offset
    so passes are not identical, and a small think time per op drifts users
    out of lockstep. The plan is infinite; the runner bounds it via
    duration/ops (closed) or session_ops (open)."""

    name = "search-load"

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        rng = random.Random(_seed_for(rng_state.get("seed", 0), user))
        queries = _content_queries(list(dataset.memory_stream(user)))
        n = len(queries)
        if n == 0:
            return
        stride = max(1, n // 7)  # coprime-ish rotation so passes differ
        pass_i = 0
        while True:
            offset = (pass_i * stride) % n
            for j in range(n):
                q = queries[(offset + j) % n]
                yield Op(type=OpType.SEARCH, query=q, delay=rng.uniform(0.0, 0.02))
            pass_i += 1


class Mixed:
    """A controllable add/search mixture over the user's `memory_stream`.

    A simple, no-dialogue workload: per op, a SEARCH is drawn with probability
    `search_weight`, else an ADD of one memory item. It needs no `turn_stream`,
    so it works with any dataset (including the synthetic one). Search queries
    are content-derived (from the user's `memory_stream`); adds reuse memories
    cyclically once the stream is exhausted (a returning user who has memories
    to re-add). A small think-time gap is attached as `Op.delay` so ops are not
    perfectly back-to-back. The plan is infinite; the runner bounds it via
    duration/ops (closed) or `session_ops` (open).

    The op mix is owned by this scenario (via `search_weight`), not by a
    runner-level weight, so the open and closed models share one Scenario
    interface. Handy as a quick congestion probe — a flat op stream with a
    tunable search/add ratio that runs under either load model.
    """

    name = "mixed"

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
        memories = list(dataset.memory_stream(user))
        if not memories:
            return
        queries = _content_queries(memories)
        i_add = 0
        i_q = 0
        # Infinite: the runner bounds consumption per session / over duration.
        while True:
            if rng.random() < self.search_weight:
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
    order) and wraps it, so a duration run replays the conversation as many
    times as needed. For every turn:

      - if the turn is a **user** turn: issue a SEARCH whose query is the
        user turn's content (the recall step) when this turn's recall cadence
        fires (see `search_every`), then ADD the turn's items;
      - otherwise (assistant turn): just ADD the turn's items.

    Adds and the recall search are interleaved exactly as a live chatbot
    session would interleave them. The search query is derived from the
    upcoming user turn (not a separate evaluation question), so recall is
    driven by the conversation itself. `query_stream` is not used.

    Requires a dataset adapter that implements `turn_stream` (LongMemEval
    does); the run fails loudly at validation time otherwise. The turn stream
    wraps, so the conversation replays until the runner stops it (duration/ops
    closed, or session_ops open). A small `think` delay between ops mimics
    user/assistant think time so users drift out of lockstep.

    `search_every` (default 1) emits a recall search before every Nth user
    turn; 1 = recall before every user turn. The user-turn counter resets each
    replay pass, so each pass is an independent, reproducible chat session.

    **LLM answer time and user think time.** A real chatbot does not loop
    back-to-back: after recalling, the LLM spends time generating an answer,
    and the user spends time typing the next turn. These are modeled as two
    *mean* delays attached to the ops that follow them:

      - `answer_time`: an Exponential(mean=answer_time) delay is attached to
        the **last** ADD of a user turn — the assistant turn's ADDs (the LLM
        answer being written) happen during this gap. Models LLM generation
        time.
      - `user_gap`: an Exponential(mean=user_gap) delay is attached to the
        **first** SEARCH (or first ADD if recall is skipped) of a user turn —
        the user is reading/typing while nothing happens at the LTM. Models
        user think/typing time before the next utterance.

    Both default to 0, which reproduces the original back-to-back loop. They
    apply uniformly to every user (a per-user ratio is a planned follow-up).
    `answer_time` only takes effect when there is a following assistant turn;
    `user_gap` only between turns (never before the very first turn of a
    replay pass, so each pass starts cleanly).
    """

    name = "chat-replay"

    def __init__(
        self,
        think: float = 0.05,
        search_every: int = 1,
        answer_time: float = 0.0,
        user_gap: float = 0.0,
    ) -> None:
        self.think = think
        self.search_every = max(1, int(search_every))
        if answer_time < 0:
            raise ValueError("answer_time must be >= 0")
        if user_gap < 0:
            raise ValueError("user_gap must be >= 0")
        self.answer_time = answer_time
        self.user_gap = user_gap

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
        turns = list(dataset.turn_stream(user))
        if not turns:
            return
        while True:  # wrap the conversation for sustained load
            user_turn_idx = 0
            for t_i, turn in enumerate(turns):
                is_user = turn.role.lower() == "user"
                turn_ops: list[Op] = []
                if is_user and turn.items:
                    if user_turn_idx % self.search_every == 0:
                        # Recall before answering: query is the user turn's content.
                        first_content = turn.items[0].content
                        turn_ops.append(
                            Op(
                                type=OpType.SEARCH,
                                query=QueryItem(query=first_content, top_k=20),
                                delay=rng.uniform(0.0, self.think),
                            )
                        )
                    user_turn_idx += 1
                for item in turn.items:
                    turn_ops.append(
                        Op(
                            type=OpType.ADD,
                            items=[item],
                            delay=rng.uniform(0.0, self.think),
                        )
                )
                if turn_ops:
                    # user_gap: the user reads/typing before the next utterance.
                    # Attached to the first op of a user turn, but not the very
                    # first turn of a replay pass (each pass starts cleanly).
                    if is_user and t_i > 0 and self.user_gap > 0:
                        extra = rng.expovariate(1.0 / self.user_gap)
                        turn_ops[0] = replace(turn_ops[0], delay=turn_ops[0].delay + extra)
                    # answer_time: the LLM generating the answer (assistant turn)
                    # happens during the gap after a user turn's last add.
                    if is_user and self.answer_time > 0:
                        extra = rng.expovariate(1.0 / self.answer_time)
                        turn_ops[-1] = replace(turn_ops[-1], delay=turn_ops[-1].delay + extra)
                    for op in turn_ops:
                        yield op


SCENARIOS: dict[str, type] = {
    ChatReplay.name: ChatReplay,
    AddLoad.name: AddLoad,
    SearchLoad.name: SearchLoad,
    Mixed.name: Mixed,
}


def get_scenario(name: str, **kwargs: Any) -> Scenario:
    cls = SCENARIOS.get(name)
    if cls is None:
        raise ValueError(f"unknown scenario: {name}")
    return cls(**kwargs)  # type: ignore[return-value]


__all__ = [
    "ChatReplay",
    "AddLoad",
    "SearchLoad",
    "Mixed",
    "SCENARIOS",
    "get_scenario",
]
