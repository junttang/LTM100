"""Synthetic dataset adapter for fast, dependency-free load testing.

Generates deterministic per-user content from a seed, so a run can be driven
without any external dataset download. Useful for smoke runs, regression, and
comparing backends on identical synthetic load.

Each virtual user gets:
  - `memories_per_user` memory items (content is deterministic per user+index)
  - `queries_per_user` search queries

It is a real DatasetAdapter (registered as "synthetic"), not a test fixture.
"""

from __future__ import annotations

import random
from typing import Iterator

from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, UserId


class SyntheticAdapter:
    """Deterministic synthetic dataset with no external dependencies."""

    name = "synthetic"

    def __init__(
        self,
        memories_per_user: int = 100,
        queries_per_user: int = 5,
        content_chars: int = 200,
    ) -> None:
        self.memories_per_user = max(1, memories_per_user)
        self.queries_per_user = max(1, queries_per_user)
        self.content_chars = max(1, content_chars)

    def _user_rng(self, user: UserId, seed: int) -> random.Random:
        h = (seed * 1_000_003) & 0xFFFFFFFF
        for ch in user.encode("utf-8"):
            h = (h ^ ch) * 1_000_003 & 0xFFFFFFFF
        return random.Random(h)

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"syn_user_{i:05d}" for i in range(n_users)]

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        rng = self._user_rng(user + "::mem", 0)
        for i in range(self.memories_per_user):
            content = f"{user} memory {i}: " + _filler(rng, self.content_chars)
            yield MemoryItem(content=content, producer=user)

    def query_stream(self, user: UserId) -> Iterator[QueryItem]:
        rng = self._user_rng(user + "::qry", 0)
        for i in range(self.queries_per_user):
            yield QueryItem(query=f"{user} query {i}: " + _filler(rng, 40))


def _filler(rng: random.Random, n: int) -> str:
    words = "the quick brown fox jumps over a lazy dog while memory systems store and retrieve".split()
    out = []
    while len(out) < n // 4 + 1:
        out.append(rng.choice(words))
    return " ".join(out)[:n]


__all__ = ["SyntheticAdapter"]
