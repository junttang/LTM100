"""Run configuration for the load core.

A single RunConfig describes one benchmark run: how many virtual users, which
scenario, and how/when it terminates. Backend endpoint/auth and adapter
choices live in the YAML config, not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    # Number of concurrent virtual users.
    users: int = 10
    # Seeded RNG for reproducible load shapes.
    seed: int = 0
    # Time-based termination: run for this many seconds (0 = disabled).
    duration: float = 0.0
    # Count-based termination: stop after this many total ops (0 = disabled).
    ops: int = 0
    # Global concurrency cap (0 = no cap, only per-user in-flight=1 applies).
    global_concurrency: int = 0
    # Warm-up seconds run before measurement. Requests execute against the
    # backend but do not consume duration/ops or enter reports.
    warmup: float = 0.0
    # Pre-ingest each user's memory stream before the measured run so search
    # scenarios run against populated memory. Excluded from metrics.
    preingest: bool = False
    # Fraction of each user's memory stream to pre-ingest (1.0 = all).
    preingest_fraction: float = 1.0
    # Ramp-up seconds over which users start (staggered, avoids thundering herd).
    rampup: float = 0.0
    # Delete per-user state on exit.
    delete_on_exit: bool = True
    # Load model: "closed" (fixed N users looping) or "open" (users arrive
    # per a Poisson process, run a bounded session, then leave).
    model: str = "closed"
    # Open-model params (ignored when model == "closed"):
    #   arrival_rate  - user arrivals per second (Poisson lambda)
    #   session_ops  - ops each arriving user performs before leaving
    #   queue_bound  - max requests queued beyond the global concurrency cap
    #                  before rejection (0 = reject immediately on cap)
    # The op mix (add vs search) for open sessions is owned by the Scenario
    # plan, not by a runner-level weight — open and closed share one Scenario
    # interface. The `mixed` scenario takes a `search_weight` constructor
    # param for that mix.
    arrival_rate: float = 0.0
    session_ops: int = 0
    queue_bound: int = 0
    # Multi-process load generation. One asyncio process saturates a core well
    # before the server does, so beyond a few dozen users a single-process run
    # measures the generator rather than the target. procs > 1 shards the users
    # across OS processes; procs == 1 is the original single-process topology.
    procs: int = 1
    proc_index: int = 0

    def __post_init__(self) -> None:
        if self.duration <= 0 and self.ops <= 0:
            raise ValueError("either duration or ops must be > 0")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.users <= 0:
            raise ValueError("users must be > 0")
        if self.procs < 1:
            raise ValueError(f"procs must be >= 1, got {self.procs}")
        if not 0 <= self.proc_index < self.procs:
            raise ValueError(
                f"proc_index must be in [0, {self.procs}), got {self.proc_index}"
            )
        if self.procs > self.users:
            raise ValueError(
                f"procs ({self.procs}) exceeds users ({self.users}); "
                "some shards would have no work"
            )
        if self.model not in ("closed", "open"):
            raise ValueError(f"model must be 'closed' or 'open', got {self.model!r}")
        if self.model == "open":
            if self.duration <= 0:
                raise ValueError("open model requires duration > 0")
            if self.arrival_rate <= 0:
                raise ValueError("open model requires arrival_rate > 0")
            if self.session_ops <= 0:
                raise ValueError("open model requires session_ops > 0")
