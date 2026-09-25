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
from queue import Empty
from typing import Any

from ltm100.core.op import OpResult

# Set by the parent before spawning; the child rebuilds its own run from it.
_ShardEntry = Callable[[dict[str, Any], int], list[OpResult]]
_worker_measure_sync: dict[str, Any] | None = None


def _init_worker(measure_sync: dict[str, Any] | None) -> None:
    global _worker_measure_sync
    _worker_measure_sync = measure_sync


def _run_entry(
    entry: _ShardEntry, args: dict[str, Any], proc_index: int
) -> list[OpResult]:
    worker_args = dict(args)
    if _worker_measure_sync is not None:
        worker_args["_measure_sync"] = _worker_measure_sync
    return entry(worker_args, proc_index)


def run_shards(
    entry: _ShardEntry,
    args: dict[str, Any],
    procs: int,
    *,
    on_measure_start: Callable[[], None] | None = None,
    on_measure_end: Callable[[], None] | None = None,
) -> list[OpResult]:
    """Run `procs` shards concurrently and return their pooled raw results.

    `entry(args, proc_index)` must build and run one shard, returning its raw
    OpResults. It runs in a spawned child, so it has to be importable by name.
    """
    if procs == 1:
        return entry(args, 0)

    ctx = mp.get_context("spawn")
    synchronize = on_measure_start is not None or on_measure_end is not None
    sync = (
        {
            "queue": ctx.Queue(),
            "start_release": ctx.Event(),
            "end_release": ctx.Event(),
        }
        if synchronize
        else None
    )
    with ctx.Pool(
        processes=procs,
        initializer=_init_worker,
        initargs=(sync,),
    ) as pool:
        pending = pool.starmap_async(
            _run_entry, [(entry, args, i) for i in range(procs)]
        )
        if sync is not None:
            _wait_for_workers(sync["queue"], "start", procs, pending)
            try:
                if on_measure_start is not None:
                    on_measure_start()
            finally:
                sync["start_release"].set()

            _wait_for_workers(sync["queue"], "end", procs, pending)
            try:
                if on_measure_end is not None:
                    on_measure_end()
            finally:
                sync["end_release"].set()

        parts = pending.get()

    pooled: list[OpResult] = []
    for part in parts:
        pooled.extend(part)
    return pooled


def _wait_for_workers(queue, stage: str, procs: int, pending) -> None:
    """Wait until every worker reaches one measurement boundary.

    Workers report an exception through the same queue so the parent does not
    wait forever when setup, pre-ingest, or a runner hook fails before the
    boundary.
    """
    arrived: set[int] = set()
    while len(arrived) < procs:
        try:
            kind, index, detail = queue.get(timeout=0.1)
        except Empty:
            if pending.ready():
                pending.get()
                raise RuntimeError(
                    f"workers exited before the {stage} measurement boundary"
                )
            continue
        if kind == "error":
            raise RuntimeError(
                f"worker {index} failed before the {stage} measurement "
                f"boundary: {detail}"
            )
        if kind == stage:
            arrived.add(index)


__all__ = ["run_shards"]
