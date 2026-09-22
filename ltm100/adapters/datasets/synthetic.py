"""Synthetic dataset adapter for fast, dependency-free load testing.

Generates deterministic per-user content from a seed, so a run can be driven
without any external dataset download. Useful for smoke runs, regression, and
comparing backends on identical synthetic load.

Each virtual user gets `memories_per_user` memory items (content is
deterministic per user+index). Search queries are content-derived by the
scenarios from these memory items, so no separate query stream is exposed.

It is a real DatasetAdapter (registered as "synthetic"), not a test fixture.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ltm100.common import MemoryItem, UserId


class SyntheticAdapter:
    """Deterministic synthetic dataset with no external dependencies."""

    name = "synthetic"

    def __init__(
        self,
        memories_per_user: int = 100,
        content_chars: int = 200,
        categories: int = 0,
    ) -> None:
        self.memories_per_user = max(1, memories_per_user)
        self.content_chars = max(1, content_chars)
        # A field for --filter to select on. 0 leaves metadata empty, so the
        # corpus is byte-identical to one built without this option; N spreads
        # items over N values, so a single-value filter selects about 1/N.
        self.categories = max(0, categories)

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
            # cat_<i mod N>, the same naming the REST bench uses, so a filter
            # expression is comparable between the two harnesses.
            metadata = (
                {"category": f"cat_{i % self.categories}"} if self.categories else {}
            )
            yield MemoryItem(content=content, producer=user, metadata=metadata)


def _filler(rng: random.Random, n: int) -> str:
    words = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "memory", "systems", "store", "and", "retrieve"]
    out = []
    while len(out) < n // 4 + 1:
        out.append(rng.choice(words))
    return " ".join(out)[:n]


__all__ = ["SyntheticAdapter"]
