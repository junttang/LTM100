"""Load core: orchestrates virtual users and drives a backend.

This is the closed-model runner: a fixed number of virtual users, each an
asyncio task that loops over its scenario plan emitting one request at a time
(in-flight = 1 per user). An optional global semaphore caps total concurrency
independently of the user count.

Termination is either time-based (duration) or count-based (total ops). The
runner drains in-flight requests at termination, then returns all recorded
OpResults.

Open-model (arrival-rate driven) behavior shares this same runner: a Scenario
expresses inter-arrival via `Op.delay`, so an open scenario is just a plan
whose delays follow an arrival process. Congestion policy for the open model
(rejection under overload) is handled at the Scenario/runner boundary and
recorded as status="rejected".
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ltm100.common import DatasetAdapter, LTMClient, UserId
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpResult, OpType, Scenario
from ltm100.metrics.recorder import InMemoryRecorder, MetricsRecorder

logger = logging.getLogger(__name__)


class LoadRunner:
    """Drives N virtual users through a Scenario against an LTMClient."""

    def __init__(
        self,
        *,
        client: LTMClient,
        dataset: DatasetAdapter,
        scenario: Scenario,
        config: RunConfig,
        recorder: MetricsRecorder | None = None,
    ) -> None:
        self.client = client
        self.dataset = dataset
        self.scenario = scenario
        self.config = config
        self.recorder = recorder or InMemoryRecorder()
        self._global_sem: asyncio.Semaphore | None = None
        self._stop = asyncio.Event()
        self._ops_done = 0
        self._ops_lock = asyncio.Lock()
        self._start_time = 0.0

    async def run(self) -> list[OpResult]:
        users = self.dataset.users(self.config.users, seed=self.config.seed)
        await self.client.setup(users)

        if self.config.global_concurrency > 0:
            self._global_sem = asyncio.Semaphore(self.config.global_concurrency)

        self._start_time = time.monotonic()
        deadline = (
            self._start_time + self.config.duration
            if self.config.duration > 0
            else None
        )

        tasks = []
        for i, user in enumerate(users):
            # Staggered start for ramp-up: user i starts at i*ramp_step.
            start_delay = self._ramp_delay(i, len(users))
            tasks.append(
                asyncio.create_task(self._user_loop(user, start_delay, deadline))
            )

        # Time-based stop signal.
        if deadline is not None:
            asyncio.create_task(self._timer(deadline))

        await asyncio.gather(*tasks, return_exceptions=True)
        return self.recorder.raw()

    def _ramp_delay(self, index: int, total: int) -> float:
        if self.config.rampup <= 0 or total <= 1:
            return 0.0
        step = self.config.rampup / total
        return step * index

    async def _timer(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self._stop.set()

    def _should_stop(self) -> bool:
        return self._stop.is_set()

    async def _reserve_op(self) -> bool:
        """Reserve budget for one op before emitting it (count-based).

        A successful reservation is a promise to run the op, so count-based
        termination records exactly `ops` results."""
        if self.config.ops <= 0:
            return True
        async with self._ops_lock:
            if self._ops_done >= self.config.ops:
                return False
            self._ops_done += 1
            return True

    async def _user_loop(
        self, user: UserId, start_delay: float, deadline: float | None
    ) -> None:
        if start_delay > 0:
            await asyncio.sleep(start_delay)

        rng_state = {"seed": self.config.seed, "user": user}
        plan = self.scenario.plan(user, self.dataset, rng_state)

        for op in plan:
            if self._should_stop():
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            if not await self._reserve_op():
                self._stop.set()
                return
            # A successful reservation is a promise to run this op, so we
            # don't check _should_stop() here: count-based termination is
            # exact (exactly `ops` results are recorded).
            if op.delay > 0:
                await asyncio.sleep(min(op.delay, self._remaining_until(deadline)))

            async with self._maybe_global_slot():
                await self._execute(user, op)

    def _remaining_until(self, deadline: float | None) -> float:
        if deadline is None:
            return 1e9  # effectively unbounded
        return max(deadline - time.monotonic(), 0.0)

    def _maybe_global_slot(self):
        if self._global_sem is None:
            class _Null:
                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, *a):
                    return False

            return _Null()
        return self._global_sem

    async def _execute(self, user: UserId, op: Op) -> None:
        started = time.time()
        status = "ok"
        error_kind = ""
        n_items = 0
        try:
            if op.type is OpType.ADD:
                uids = await self.client.add(user, op.items)
                n_items = len(uids)
            else:
                results = await self.client.search(user, op.query)
                n_items = len(results)
        except Exception as e:  # noqa: BLE001 - record any failure as op error
            status = "error"
            error_kind = type(e).__name__
            logger.debug("op %s for %s failed: %s", op.type.value, user, e)
        ended = time.time()
        result = OpResult(
            type=op.type,
            user_id=user,
            started_at=started,
            ended_at=ended,
            status=status,
            error_kind=error_kind,
            n_items=n_items,
        )
        await self.recorder.record(result)


__all__ = ["LoadRunner"]
