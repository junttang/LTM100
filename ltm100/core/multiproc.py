"""Multi-process load generation.

One asyncio event loop saturates a single core long before a healthy server
saturates, so a single-process run past a few dozen users reports the
generator's ceiling rather than the target's. This module runs the same
LoadRunner in several OS processes, each driving a disjoint slice of the
virtual users, and pools their raw results.

Pooling raw OpResults rather than per-shard summaries matters: percentiles are
then computed over the whole population by the same `aggregate` used for a
single-process run, instead of being approximated from per-shard percentiles.

Workers are spawned, not forked: a forked child inherits the parent's event
loop and open sockets, which asyncio does not support.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

from ltm100.core.op import OpResult

# Set by the parent before spawning; the child rebuilds its own run from it.
_ShardEntry = Callable[[dict[str, Any], int], list[OpResult]]


def run_shards(
    entry: _ShardEntry, args: dict[str, Any], procs: int
) -> list[OpResult]:
    """Run `procs` shards concurrently and return their pooled raw results.

    `entry(args, proc_index)` must build and run one shard, returning its raw
    OpResults. It runs in a spawned child, so it has to be importable by name.
    """
    if procs == 1:
        return entry(args, 0)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=procs) as pool:
        parts = pool.starmap(entry, [(args, i) for i in range(procs)])

    pooled: list[OpResult] = []
    for part in parts:
        pooled.extend(part)
    return pooled


__all__ = ["run_shards"]
