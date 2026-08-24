"""Metrics recording interface.

All metrics are client-observable and recorded per request, tagged by op
type. The default implementation keeps an in-memory list and aggregates
post-run; a streaming variant (NDJSON) is available for large-scale runs.
Server-side resource metrics are out of scope (collected by the server).
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from ltm100.core.op import OpResult


class MetricsRecorder(Protocol):
    """Collects per-request results and produces an aggregated report."""

    async def record(self, result: OpResult) -> None:
        """Record one completed request. Must be safe to call concurrently."""
        ...

    async def flush(self) -> None:
        """Flush any buffered records (for streaming backends)."""
        ...

    def summary(self) -> dict:
        """Return the aggregated summary dict (count, throughput, QPS,
        latency percentiles, error rate), broken down by op type."""
        ...

    def raw(self) -> list[OpResult]:
        """Return all recorded per-request results (in-memory mode only)."""
        ...


class InMemoryRecorder:
    """Default recorder: append results to a list under a lock."""

    def __init__(self) -> None:
        self._results: list[OpResult] = []
        self._lock = asyncio.Lock()

    async def record(self, result: OpResult) -> None:
        async with self._lock:
            self._results.append(result)

    async def flush(self) -> None:
        return

    def summary(self) -> dict:
        from ltm100.metrics.aggregate import aggregate

        return aggregate(self._results)

    def raw(self) -> list[OpResult]:
        return list(self._results)


__all__ = ["MetricsRecorder", "InMemoryRecorder"]
