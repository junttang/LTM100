"""Opt-in integration tests against the official Supermemory server.

Use a disposable server and an owner/admin key. These tests create unique
container tags and delete their contents; they never reset organization data.
See docs/supermemory.md for invocation and the limits of mock-model validation.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import uuid
from datetime import datetime

import pytest
import yaml

from ltm100.adapters.backends.supermemory import SupermemoryClient
from ltm100.common import MemoryItem, QueryItem

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_SUPERMEMORY_E2E_URL"),
    reason="requires an explicitly configured disposable official Supermemory server",
)


@pytest.fixture
def live_options():
    return {
        "base_url": os.environ["LTM100_SUPERMEMORY_E2E_URL"],
        "user_prefix": f"e2e_{uuid.uuid4().hex[:20]}",
    }


async def test_live_direct_memory_search_filter_isolation_and_cleanup(live_options):
    async with SupermemoryClient(**live_options, add_batch_size=100) as client:
        users = ["user/한글", "other", "never-added"]
        try:
            await client.setup(users)
            assert await client.search(users[2], QueryItem("nothing", top_k=1)) == []
            items = [
                MemoryItem(
                    f"unique benchmark fact {i}",
                    role="assistant",
                    producer="agent",
                    timestamp="2026-10-03T00:00:00Z",
                    metadata={"category": f"cat_{i % 2}"},
                )
                for i in range(6)
            ]
            ids = await client.add(users[0], items)
            assert len(ids) == len(items) and len(set(ids)) == len(items)
            await client.add(users[1], [MemoryItem("private other user fact")])
            rows = await client.search(users[0], QueryItem(items[0].content, top_k=3))
            assert len(rows) == 3
            assert all(r.uid in ids for r in rows)
            assert all(r.metadata["ltm100_role"] == "assistant" for r in rows)
            assert all(r.metadata["ltm100_producer"] == "agent" for r in rows)
            assert all(
                r.metadata["ltm100_timestamp"] == "2026-10-03T00:00:00Z" for r in rows
            )
            filtered = await client.search(
                users[0],
                QueryItem(items[0].content, top_k=3, filter="metadata.category=cat_0"),
            )
            assert len(filtered) == 3
            assert all(r.metadata["category"] == "cat_0" for r in filtered)
            one = await client.search(users[0], QueryItem(items[0].content, top_k=1))
            assert len(one) == 1
            wide = await client.search(
                users[0], QueryItem(items[0].content, top_k=100, expand_context=1)
            )
            assert len(wide) == len(items)
            await client.teardown([users[0]], delete=False)
            assert await client.search(users[0], QueryItem(items[0].content))
            await client.teardown([users[0]], delete=True)
            assert await client.search(users[0], QueryItem(items[0].content)) == []
            assert await client.search(users[1], QueryItem("private other user fact"))
            await client.teardown([users[0]], delete=True)
        finally:
            await client.teardown(users, delete=True)


def _write_config(tmp_path, live_options, *, chat=False):
    dataset = {
        "name": "synthetic",
        "memories_per_user": 12,
        "content_chars": 200,
        "categories": 2,
    }
    if chat:
        records = [
            {
                "question_id": f"q{i}",
                "question": "unused",
                "answer": "unused",
                "haystack_sessions": [
                    [
                        {"role": "user", "content": f"user fact {i} session {s}"},
                        {
                            "role": "assistant",
                            "content": f"assistant fact {i} session {s}",
                        },
                    ]
                    for s in range(6)
                ],
            }
            for i in range(4)
        ]
        data = tmp_path / "longmemeval.json"
        data.write_text(json.dumps(records), encoding="utf-8")
        dataset = {"name": "longmemeval", "path": str(data)}
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": dataset,
                "backend": {"name": "supermemory", **live_options, "add_batch_size": 6},
            }
        )
    )
    return config


async def _cli(*args):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "ltm100.cli",
        *map(str, args),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=90)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == 0, (stdout.decode(), stderr.decode())
    return stderr.decode()


@pytest.mark.parametrize(
    ("scenario", "procs", "model"),
    [
        ("add-load", 1, "closed"),
        ("search-load", 1, "closed"),
        ("mixed", 2, "closed"),
        ("chat-replay", 1, "closed"),
        ("chat-replay", 2, "closed"),
        ("chat-replay", 2, "open"),
    ],
)
async def test_live_cli_workloads_and_observability(
    tmp_path, live_options, scenario, procs, model
):
    chat = scenario == "chat-replay"
    config = _write_config(tmp_path, live_options, chat=chat)
    output = tmp_path / "out"
    args = [
        "run",
        "--config",
        config,
        "--scenario",
        scenario,
        "--users",
        4,
        "--procs",
        procs,
        "--model",
        model,
        "--seed",
        0,
        "--top-k",
        3,
        "--think",
        0,
        "--global-concurrency",
        4,
        "--preingest",
        "--preingest-items-per-user",
        6,
        "--raw",
        "--time-series-interval",
        0.2,
        "--server-metrics",
        "--server-metrics-interval",
        0.2,
        "--output",
        output,
    ]
    if chat:
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "defaults": {
                        "think": 0,
                        "answer_time": 0.01,
                        "answer_time_variation": 0.3,
                        "user_gap": 0.01,
                        "user_gap_variation": 0.5,
                    },
                    "groups": [
                        {
                            "name": "standard",
                            "share": 1,
                            "top_k": 3,
                            "concurrent_sessions": 2,
                            "max_sessions_per_user": 2,
                        }
                    ],
                }
            )
        )
        args += ["--chat-profile", profile]
    if model == "open":
        args += [
            "--duration",
            0.8,
            "--arrival-rate",
            20,
            "--session-ops",
            6,
            "--queue-bound",
            20,
            "--warmup",
            0.2,
        ]
    else:
        args += ["--ops", 24, "--warmup", 0.15]
    await _cli(*args)
    payload = json.loads((output / "summary.json").read_text())
    summary = payload["summary"]
    assert summary["total"] > 0 and summary["errors"] == 0
    if model == "closed":
        assert summary["total"] == 24
    assert payload["meta"]["preingest_stats"]["input_items"] == 24
    assert payload["server_metrics"]["status"] == "unsupported"
    raw = [
        json.loads(line) for line in (output / "raw.ndjson").read_text().splitlines()
    ]
    assert len(raw) == summary["total"]
    assert all(row["status"] in ("ok", "rejected") for row in raw)
    meta = payload["meta"]
    start = datetime.fromisoformat(meta["measurement_started_at"]).timestamp()
    end = datetime.fromisoformat(meta["measurement_ended_at"]).timestamp()
    assert all(start <= row["started_at"] <= row["ended_at"] <= end for row in raw)
    assert datetime.fromisoformat(meta["started_at"]).timestamp() < start < end
    assert datetime.fromisoformat(meta["ended_at"]).timestamp() >= end
    if chat:
        assert summary["by_group"]["standard"]["total"] == summary["total"]
        assert all(
            row["group"] == "standard" and row["session_id"] is not None for row in raw
        )
    with (output / "timeseries.csv").open() as stream:
        series = list(csv.DictReader(stream))
    assert (
        sum(int(row["completed"]) for row in series if row["op_type"] == "all")
        == summary["accepted"]
    )
    assert (
        sum(int(row["successful"]) for row in series if row["op_type"] == "all")
        == summary["successful"]
    )
    assert sum(int(row["errors"]) for row in series if row["op_type"] == "all") == 0
    assert all(
        int(row["in_flight_end"]) == 0
        for row in series
        if row["bucket_index"] == series[-1]["bucket_index"]
    )
    with (output / "server_metrics_timeseries.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    async with SupermemoryClient(**live_options) as client:
        # Reconstruct the exact dataset users; cleanup must be done by the CLI.
        from ltm100.config import build_dataset, load_config

        users = build_dataset(load_config(config).dataset).users(4, seed=0)
        for user in users:
            assert await client.search(user, QueryItem("fact")) == []


async def test_live_memory_growth_sweep(tmp_path, live_options):
    config = _write_config(tmp_path, live_options)
    output = tmp_path / "sweep"
    await _cli(
        "sweep",
        "memory-growth",
        "--config",
        config,
        "--memory-counts",
        "3,6",
        "--queries-per-user",
        2,
        "--users",
        2,
        "--ops",
        8,
        "--repetitions",
        1,
        "--top-k",
        3,
        "--output",
        output,
    )
    reports = list(output.rglob("summary.json"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    namespaces = [
        repeat["namespace"]
        for point in manifest["points"]
        for repeat in point["repetitions"]
    ]
    assert len(set(namespaces)) == 2
    assert len(reports) == 2
    for path in reports:
        payload = json.loads(path.read_text())
        assert payload["summary"]["total"] == 8
        assert payload["summary"]["errors"] == 0
        assert payload["meta"]["preingest_stats"]["input_items"] in (6, 12)
    from ltm100.config import build_dataset, load_config

    source_users = build_dataset(load_config(config).dataset).users(2, seed=0)
    async with SupermemoryClient(**live_options) as client:
        for namespace in namespaces:
            for user in source_users:
                assert (
                    await client.search(f"{namespace}{user}", QueryItem("fact")) == []
                )
