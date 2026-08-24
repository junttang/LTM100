"""Common data types shared across LTM100.

These types are the lingua franca between the three pluggable axes
(DatasetAdapter, LTMClient, Transport) and the load core. They intentionally
carry no backend-specific or dataset-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

# A virtual-user identifier. Opaque to the core; backends map it to a tenant key.
UserId = str


@dataclass(frozen=True)
class MemoryItem:
    """A single unit of memory to store (an `add` payload)."""

    content: str
    # Optional provenance / metadata. Backends that don't support a field
    # should silently ignore it.
    timestamp: str | None = None
    producer: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryItem:
    """A single search query.

    `expected` fields (gold answer, facts, ...) are kept only for optional
    traceability/debugging — they are NEVER scored by LTM100.
    """

    query: str
    top_k: int = 20
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultItem:
    """A single retrieved memory item returned by a backend `search`."""

    content: str
    score: float | None = None
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Turns a raw dataset into per-user add/search streams.

    The adapter knows nothing about backends or transports. It is responsible
    for mapping virtual users onto dataset samples, including replication when
    `n_users` exceeds the number of available samples.
    """

    name: str

    def users(self, n_users: int, *, seed: int) -> list[UserId]:
        """Return `n_users` virtual-user identifiers (may replicate samples)."""
        ...

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        """Yield add payloads for this user, in ingestion order."""
        ...

    def query_stream(self, user: UserId) -> Iterator[QueryItem]:
        """Yield search queries for this user. May repeat or interleave."""
        ...


@runtime_checkable
class LTMClient(Protocol):
    """Backend adapter: async add/search against one LTM backend.

    Per-user scoping is the adapter's responsibility: it maps `UserId` to the
    backend's tenant key so that a user only ever sees their own memories.
    `setup` and `teardown` are out-of-measurement phases.
    """

    name: str

    async def setup(self, users: list[UserId]) -> None:
        """Per-run provisioning (e.g. create a tenant/project per user)."""
        ...

    async def add(
        self, user: UserId, items: list[MemoryItem]
    ) -> list[str]:
        """Store `items` for `user`; return backend ids (one per item)."""
        ...

    async def search(
        self, user: UserId, query: QueryItem
    ) -> list[ResultItem]:
        """Retrieve memories for `user`, scoped to this user only."""
        ...

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        """Optional cleanup (delete per-user state) per run."""
        ...


@runtime_checkable
class Transport(Protocol):
    """Wire protocol under a backend adapter (REST first, MCP later).

    Backends that ship only an SDK (no server) implement LTMClient directly
    and ignore this layer.
    """

    async def request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform a single request and return the parsed response body."""
        ...
