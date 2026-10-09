"""Illustrate chat-replay dispatch patterns with a fixed-delay backend.

This example uses LTM100's real scenario, runner, and time-series aggregator.
It neither provisions an LTM server nor stores or retrieves real memories.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ltm100.common import MemoryItem, QueryItem, ResultItem, Turn, UserId
from ltm100.core.config import RunConfig
from ltm100.core.op import OpResult
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import ChatReplay
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import (
    write_raw_ndjson,
    write_summary_json,
    write_timeseries_csv,
)
from ltm100.metrics.timeseries import aggregate_timeseries


@dataclass(frozen=True)
class PatternConfig:
    users: tuple[int, ...] = (1, 16, 64, 256)
    duration: float = 1800
    user_start_interval: float = 4
    seed: int = 0
    search_latency: float = 0.6
    add_latency: float = 0.1
    answer_time: float = 18
    answer_time_variation_seconds: float = 3
    user_gap: float = 84
    user_gap_variation_seconds: float = 40
    top_k: int = 20
    turn_pairs: int = 128
    bucket_seconds: float = 5

    def __post_init__(self) -> None:
        if not self.users or len(set(self.users)) != len(self.users):
            raise ValueError("users must contain distinct positive integers")
        for value in (*self.users, self.top_k, self.turn_pairs):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "users, top_k, and turn_pairs must be positive integers"
                )
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in (
            "duration",
            "user_start_interval",
            "search_latency",
            "add_latency",
            "answer_time",
            "answer_time_variation_seconds",
            "user_gap",
            "user_gap_variation_seconds",
            "bucket_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.duration == 0 or self.bucket_seconds == 0:
            raise ValueError("duration and bucket_seconds must be > 0")
        for mean, variation in (
            (self.answer_time, self.answer_time_variation_seconds),
            (self.user_gap, self.user_gap_variation_seconds),
        ):
            if variation > mean:
                raise ValueError("timing variation must not exceed its mean")

    @classmethod
    def load(cls, path: Path) -> PatternConfig:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("example config must be a YAML mapping")
        if "users" in values:
            values["users"] = tuple(values["users"])
        return cls(**values)


class ExampleDialogue:
    """Short alternating turns, each represented by exactly one memory item."""

    name = "example-dialogue"

    def __init__(self, turn_pairs: int) -> None:
        self.turn_pairs = turn_pairs

    def users(self, n_users: int, *, seed: int) -> list[UserId]:
        return [f"user_{index}" for index in range(n_users)]

    def turn_stream(self, user: UserId) -> Iterator[Turn]:
        for index in range(self.turn_pairs):
            for role in ("user", "assistant"):
                yield Turn(
                    role=role,
                    items=[
                        MemoryItem(
                            content=f"{user}: {role} message {index}",
                            producer=user,
                            role=role,
                        )
                    ],
                )

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        for turn in self.turn_stream(user):
            yield from turn.items


class FixedDelayBackend:
    """Wait independently per call, without a backend queue or resource cap."""

    name = "fixed-delay-example"

    def __init__(self, *, search_latency: float, add_latency: float) -> None:
        self.search_latency = search_latency
        self.add_latency = add_latency
        self._next_id = 0

    async def setup(self, users: list[UserId]) -> None:
        pass

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        await asyncio.sleep(self.add_latency)
        ids = [f"example-{self._next_id + index}" for index in range(len(items))]
        self._next_id += len(items)
        return ids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        await asyncio.sleep(self.search_latency)
        return [ResultItem(content="Simulated result; no retrieval was performed.")]

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        pass


def request_counts(rows: list[dict]) -> list[dict]:
    """Project the existing time series onto per-bin backend dispatch counts."""
    counts: dict[int, dict] = {}
    for row in rows:
        index = row["bucket_index"]
        bucket = counts.setdefault(
            index,
            {
                "elapsed_start_s": row["elapsed_start_s"],
                "interval_seconds": row["interval_seconds"],
                "add_started": 0,
                "search_started": 0,
            },
        )
        if row["op_type"] != "all":
            bucket[f"{row['op_type']}_started"] = row["started"]
    return [counts[index] for index in sorted(counts)]


async def run_case(config: PatternConfig, users: int, output: Path) -> list[OpResult]:
    """Run one real-time case, then write reports after all requests drain."""
    case_output = output / f"users-{users}"
    if case_output.exists():
        raise FileExistsError(f"refusing to overwrite {case_output}")
    case_output.mkdir(parents=True)
    rampup = users * config.user_start_interval if users > 1 else 0.0
    runner = LoadRunner(
        client=FixedDelayBackend(
            search_latency=config.search_latency, add_latency=config.add_latency
        ),
        dataset=ExampleDialogue(config.turn_pairs),
        scenario=ChatReplay(
            think=0,
            search_every=1,
            answer_time=config.answer_time,
            answer_time_variation=(
                config.answer_time_variation_seconds / config.answer_time
                if config.answer_time
                else 0
            ),
            user_gap=config.user_gap,
            user_gap_variation=(
                config.user_gap_variation_seconds / config.user_gap
                if config.user_gap
                else 0
            ),
            top_k=config.top_k,
        ),
        config=RunConfig(
            users=users,
            seed=config.seed,
            duration=config.duration,
            rampup=rampup,
            global_concurrency=0,
            model="closed",
        ),
    )
    print(f"Starting {users} users: duration={config.duration}s, rampup={rampup}s")
    results = await runner.run()
    await runner.client.teardown(runner.users, delete=True)
    assert runner.measurement_started_at is not None
    assert runner.measurement_ended_at is not None
    rows = aggregate_timeseries(
        results,
        interval_seconds=config.bucket_seconds,
        measurement_started_at=runner.measurement_started_at,
        measurement_ended_at=runner.measurement_ended_at,
    )
    write_timeseries_csv(rows, case_output / "timeseries.csv")
    write_raw_ndjson(results, case_output / "raw.ndjson")
    meta = {
        "example": "fixed-delay chat-replay; no real LTM server",
        "dataset": "generated alternating dialogue; one item per turn",
        "config": {**asdict(config), "users": users},
        "model": "closed",
        "global_concurrency": 0,
        "concurrent_sessions": 1,
        "rampup": rampup,
        "measurement_started_at": runner.measurement_started_at,
        "measurement_ended_at": runner.measurement_ended_at,
    }
    write_summary_json(aggregate(results), case_output / "summary.json", meta=meta)
    counts = request_counts(rows)
    if counts:
        with (case_output / "request-counts.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(counts[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(counts)
    print(f"Finished {users} users: {len(results)} requests -> {case_output}")
    return results


async def run_all(config: PatternConfig, output: Path) -> None:
    # Check every destination before starting any of the independent runners.
    for users in config.users:
        if (output / f"users-{users}").exists():
            raise FileExistsError(f"refusing to overwrite {output / f'users-{users}'}")
    await asyncio.gather(*(run_case(config, users, output) for users in config.users))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("chat-replay.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_all(PatternConfig.load(args.config), args.output))


if __name__ == "__main__":
    main()
