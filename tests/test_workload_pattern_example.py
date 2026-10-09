"""Fixed-delay examples use real scheduling and preserve dispatch counts."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from ltm100.common import MemoryItem, QueryItem
from ltm100.core.config import RunConfig
from ltm100.core.op import OpResult, OpType
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import ChatReplay
from ltm100.metrics.timeseries import aggregate_timeseries

EXAMPLES = Path(__file__).parents[1] / "examples" / "workload-patterns"


def _load_example(name):
    spec = importlib.util.spec_from_file_location(
        f"pattern_example_{name}", EXAMPLES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example("run")
plot = _load_example("plot")


def test_preset_uses_requested_real_time_settings():
    config = example.PatternConfig.load(EXAMPLES / "chat-replay.yaml")
    assert config.users == (1, 16, 64, 256)
    assert config.duration == 1800
    assert config.user_start_interval == 4
    assert (config.search_latency, config.add_latency) == (0.6, 0.1)
    assert (config.answer_time, config.answer_time_variation_seconds) == (18, 3)
    assert (config.user_gap, config.user_gap_variation_seconds) == (84, 40)
    assert config.bucket_seconds == 5
    for users in config.users:
        runner = LoadRunner(
            client=example.FixedDelayBackend(search_latency=0.6, add_latency=0.1),
            dataset=example.ExampleDialogue(config.turn_pairs),
            scenario=ChatReplay(),
            config=RunConfig(users=users, duration=1800, rampup=users * 4),
        )
        assert [runner._ramp_delay(i, users) for i in range(users)] == [
            i * 4 for i in range(users)
        ]


@pytest.mark.parametrize(
    "changes",
    [
        {"users": ()},
        {"users": (1, 1)},
        {"users": (0,)},
        {"users": (1.5,)},
        {"duration": 0},
        {"duration": float("nan")},
        {"search_latency": float("inf")},
        {"add_latency": -0.1},
        {"answer_time_variation_seconds": 19},
        {"user_gap_variation_seconds": 85},
        {"bucket_seconds": 0},
        {"turn_pairs": 0},
    ],
)
def test_invalid_example_settings_fail_early(changes):
    with pytest.raises(ValueError):
        replace(example.PatternConfig(), **changes)


@pytest.mark.asyncio
async def test_fixed_delays_are_awaited_once_per_call(monkeypatch):
    delays = []

    async def capture(delay):
        delays.append(delay)

    monkeypatch.setattr(example.asyncio, "sleep", capture)
    backend = example.FixedDelayBackend(search_latency=0.6, add_latency=0.1)
    await backend.setup(["u0"])
    ids = await backend.add("u0", [MemoryItem("a"), MemoryItem("b")])
    more_ids = await backend.add("u0", [MemoryItem("c")])
    hits = await backend.search("u0", QueryItem("a"))
    await backend.teardown(["u0"], delete=True)
    assert delays == [0.1, 0.1, 0.6]
    assert len(set(ids + more_ids)) == 3
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_backend_requests_wait_concurrently(monkeypatch):
    entered = 0
    release = asyncio.Event()

    async def wait(delay):
        nonlocal entered
        entered += 1
        if entered == 3:
            release.set()
        await release.wait()

    monkeypatch.setattr(example.asyncio, "sleep", wait)
    backend = example.FixedDelayBackend(search_latency=0.6, add_latency=0.1)
    await asyncio.wait_for(
        asyncio.gather(
            backend.search("u0", QueryItem("q")),
            backend.search("u1", QueryItem("q")),
            backend.add("u2", [MemoryItem("m")]),
        ),
        timeout=1,
    )
    assert entered == 3


def test_short_turns_keep_chat_order_and_timing_ranges():
    dataset = example.ExampleDialogue(128)
    scenario = ChatReplay(
        think=0,
        answer_time=18,
        answer_time_variation=3 / 18,
        user_gap=84,
        user_gap_variation=40 / 84,
    )
    ops = list(islice(scenario.plan("user_0", dataset, {"seed": 0}), 128 * 3))
    assert [op.type for op in ops] == [OpType.SEARCH, OpType.ADD, OpType.ADD] * 128
    assert ops[0].delay == 0
    for index in range(128):
        search, user_add, assistant_add = ops[index * 3 : index * 3 + 3]
        assert search.query.top_k == 20
        assert len(user_add.items) == len(assistant_add.items) == 1
        assert user_add.items[0].role == "user"
        assert assistant_add.items[0].role == "assistant"
        assert user_add.delay == 0
        assert 15 <= assistant_add.delay <= 21
        if index > 0:
            assert 44 <= search.delay <= 124


def test_counts_use_starts_not_completions_and_keep_empty_bins():
    results = [
        OpResult(OpType.SEARCH, "u0", 100.1, 100.7, "ok"),
        OpResult(OpType.ADD, "u0", 100.7, 100.8, "ok"),
        OpResult(OpType.SEARCH, "u1", 101.1, 101.1, "rejected"),
    ]
    rows = aggregate_timeseries(
        results,
        interval_seconds=0.5,
        measurement_started_at=100,
        measurement_ended_at=102.2,
    )
    counts = example.request_counts(rows)
    assert len(counts) == 5
    assert [(row["add_started"], row["search_started"]) for row in counts] == [
        (0, 1),
        (1, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]
    assert counts[-1]["interval_seconds"] == pytest.approx(0.2)


def test_zero_requests_still_produce_empty_count_bins():
    rows = aggregate_timeseries(
        [],
        interval_seconds=5,
        measurement_started_at=0,
        measurement_ended_at=15,
    )
    assert example.request_counts(rows) == [
        {
            "elapsed_start_s": index * 5,
            "interval_seconds": 5,
            "add_started": 0,
            "search_started": 0,
        }
        for index in range(3)
    ]


@pytest.mark.asyncio
async def test_default_backend_latencies_are_real_waits(tmp_path):
    config = replace(
        example.PatternConfig(), users=(2,), duration=1.2, user_start_interval=0.1
    )
    results = await example.run_case(config, 2, tmp_path)
    assert [result.type for result in results].count(OpType.SEARCH) == 2
    assert [result.type for result in results].count(OpType.ADD) == 2
    for result in results:
        expected = 0.6 if result.type == OpType.SEARCH else 0.1
        assert result.ended_at - result.started_at >= expected - 0.001


@pytest.mark.asyncio
async def test_real_runner_reports_and_independent_cases(tmp_path):
    config = example.PatternConfig(
        users=(1, 3),
        duration=0.25,
        user_start_interval=0.03,
        search_latency=0.01,
        add_latency=0.002,
        answer_time=0.01,
        answer_time_variation_seconds=0.003,
        user_gap=0.02,
        user_gap_variation_seconds=0.005,
        bucket_seconds=0.05,
    )
    await example.run_all(config, tmp_path)
    for users in config.users:
        output = tmp_path / f"users-{users}"
        payload = json.loads((output / "summary.json").read_text())
        meta, summary = payload["meta"], payload["summary"]
        raw = [
            json.loads(line)
            for line in (output / "raw.ndjson").read_text().splitlines()
        ]
        with (output / "request-counts.csv").open() as file:
            counts = list(csv.DictReader(file))
        assert summary["total"] == len(raw)
        assert summary["errors"] == summary["rejected"] == 0
        assert {row["user_id"] for row in raw} == {f"user_{i}" for i in range(users)}
        assert meta["measurement_ended_at"] - meta["measurement_started_at"] >= 0.25
        for op in ("add", "search"):
            assert sum(int(row[f"{op}_started"]) for row in counts) == sum(
                row["op_type"] == op for row in raw
            )
        for index in range(users):
            first = min(
                row["started_at"] for row in raw if row["user_id"] == f"user_{index}"
            )
            assert first - meta["measurement_started_at"] >= index * 0.03 - 0.002
        assert (
            max(row["started_at"] for row in raw)
            < meta["measurement_started_at"] + config.duration + 0.002
        )


@pytest.mark.asyncio
async def test_existing_output_is_not_overwritten(tmp_path):
    (tmp_path / "users-16").mkdir()
    with pytest.raises(FileExistsError):
        await example.run_all(example.PatternConfig(), tmp_path)
    assert not (tmp_path / "users-1").exists()


def test_plot_outputs_parseable_svg_from_collected_counts(tmp_path):
    pytest.importorskip("matplotlib")
    from xml.etree import ElementTree

    output = tmp_path / "users-1"
    output.mkdir()
    counts = output / "request-counts.csv"
    counts.write_text(
        "elapsed_start_s,interval_seconds,add_started,search_started\n"
        "0,5,1,0\n5,5,0,2\n10,5,0,0\n",
        encoding="utf-8",
    )
    image = tmp_path / "pattern.svg"
    plot.plot_counts([counts], image)
    assert ElementTree.parse(image).getroot().tag.endswith("svg")
    assert "Requests / 5 s bin" in image.read_text()
