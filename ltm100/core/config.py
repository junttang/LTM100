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
    # Warm-up seconds excluded from metrics (results recorded but filtered out).
    warmup: float = 0.0
    # Ramp-up seconds over which users start (staggered, avoids thundering herd).
    rampup: float = 0.0
    # Delete per-user state on exit.
    delete_on_exit: bool = True

    def __post_init__(self) -> None:
        if self.duration <= 0 and self.ops <= 0:
            raise ValueError("either duration or ops must be > 0")
        if self.users <= 0:
            raise ValueError("users must be > 0")
