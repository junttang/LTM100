"""Opt-in deadline checks against a disposable standalone MemMachine server.

Set LTM100_MEMMACHINE_E2E_URL to run. Each test uses its own organization and
deletes its projects. A slow local embedder also exercises in-flight draining;
these tests verify functionality, not production latency or search quality.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
import uuid

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.common import MemoryItem, Turn
from ltm100.core.chat_profile import ChatGroup, ChatProfile, ChatSettings
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad, ChatReplay, SearchLoad

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_MEMMACHINE_E2E_URL"),
    reason="requires an explicitly configured disposable MemMachine server",
)


class DialogueDataset:
    name = "deadline-dialogue"

    def users(self, n_users, *, seed=0):
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user):
        for turn in self.turn_stream(user):
            yield from turn.items

    def turn_stream(self, user):
        for session in self.session_stream(user):
            yield from session

    def session_stream(self, user):
        for i in range(3):
            yield [
                Turn(role, [MemoryItem(f"{user} {role} fact {i}", role=role)])
                for role in ("user", "assistant")
            ]


class ObservedClient(MemMachineClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.starts = []
        self.added_roles = []

    async def add(self, user, items):
        self.starts.append(time.monotonic())
        self.added_roles.extend(item.role for item in items)
        return await super().add(user, items)

    async def search(self, user, query):
        self.starts.append(time.monotonic())
        return await super().search(user, query)


class ObservedRunner(LoadRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phases = []

    async def _run_workload(self, *, duration):
        offset = len(self.client.starts)
        await super()._run_workload(duration=duration)
        self.phases.append((self._start_time, duration, self.client.starts[offset:]))


@pytest.fixture
def options():
    return {
        "base_url": os.environ["LTM100_MEMMACHINE_E2E_URL"],
        "org_prefix": f"deadline_{uuid.uuid4().hex}",
        "timeout": 30,
    }


@pytest.mark.parametrize("model", ["closed", "open"])
@pytest.mark.parametrize("workload", ["add-load", "search-load", "chat-replay"])
async def test_live_dispatch_boundaries_and_drain(options, model, workload):
    scenario = {
        "add-load": AddLoad(),
        "search-load": SearchLoad(top_k=2),
        "chat-replay": ChatReplay(think=0, top_k=2),
    }[workload]
    dataset = DialogueDataset()
    async with ObservedClient(**options) as client:
        runner = ObservedRunner(
            client=client,
            dataset=dataset,
            scenario=scenario,
            config=RunConfig(
                users=4,
                duration=0.04,
                warmup=0.03,
                global_concurrency=1,
                queue_bound=100,
                model=model,
                arrival_rate=1000,
                session_ops=10,
                preingest=True,
                preingest_items_per_user=2,
            ),
        )
        try:
            results = await runner.run()
            assert results
            assert all(result.status == "ok" for result in results)
            for start, duration, calls in runner.phases:
                assert calls
                assert all(start <= call < start + duration for call in calls)
            assert runner._admitted == 0
            assert runner._global_sem._value == 1
            assert runner.measurement_ended_at >= max(r.ended_at for r in results)
            assert runner.measurement_started_at <= min(r.started_at for r in results)
            assert len(results) == len(runner.phases[-1][2])
        finally:
            await client.teardown(dataset.users(4), delete=True)


@pytest.mark.parametrize("model", ["closed", "open"])
async def test_live_chat_delay_drops_pending_assistant_add(options, model):
    dataset = DialogueDataset()
    async with ObservedClient(**options) as client:
        profile = ChatProfile(
            [
                ChatGroup(
                    "chat",
                    1.0,
                    ChatSettings(
                        think=0,
                        search_every=1,
                        top_k=2,
                        answer_time=10,
                        answer_time_variation=0,
                        user_gap=0,
                        concurrent_sessions=3,
                        max_sessions_per_user=3,
                    ),
                )
            ]
        )
        runner = ObservedRunner(
            client=client,
            dataset=dataset,
            scenario=ChatReplay(
                think=0,
                top_k=2,
                answer_time=10,
                answer_time_variation=0,
                profile=profile,
            ),
            config=RunConfig(
                users=1,
                duration=1,
                model=model,
                arrival_rate=10,
                session_ops=3,
                global_concurrency=1,
                queue_bound=100,
            ),
        )
        try:
            results = await runner.run()
            assert results and all(r.status == "ok" for r in results)
            assert all(
                call < start + duration
                for start, duration, calls in runner.phases
                for call in calls
            )
            # Every admitted session can search and add its user turn, but the
            # fixed 10-second answer delay exceeds this run's dispatch window.
            assert sum(r.type.value == "add" for r in results) <= sum(
                r.type.value == "search" for r in results
            )
            assert "assistant" not in client.added_roles
            assert all(result.group == "chat" for result in results)
        finally:
            await client.teardown(dataset.users(1), delete=True)


@pytest.mark.parametrize("model", ["closed", "open", "count"])
async def test_live_multiprocess_cli_reports(tmp_path, options, model):
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "synthetic",
                    "memories_per_user": 4,
                    "content_chars": 200,
                },
                "backend": {"name": "memmachine", **options},
            }
        )
    )
    output = tmp_path / "out"
    args = [
        sys.executable,
        "-m",
        "ltm100.cli",
        "run",
        "--config",
        str(config),
        "--scenario",
        "search-load",
        "--users",
        "4",
        "--procs",
        "2",
        "--global-concurrency",
        "2",
        "--preingest",
        "--warmup",
        "0.03",
        "--top-k",
        "2",
        "--output",
        str(output),
        "--raw",
        "--time-series-interval",
        "0.02",
        "--server-metrics",
        "--server-metrics-interval",
        "0.02",
    ]
    if model == "count":
        args.extend(["--ops", "17"])
    else:
        args.extend(["--duration", "0.04", "--model", model])
        if model == "open":
            args.extend(
                [
                    "--arrival-rate",
                    "1000",
                    "--session-ops",
                    "10",
                    "--queue-bound",
                    "100",
                ]
            )
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == 0, (stdout.decode(), stderr.decode())
    raw = [
        json.loads(line) for line in (output / "raw.ndjson").read_text().splitlines()
    ]
    assert raw and all(r["status"] == "ok" for r in raw)
    if model == "count":
        assert len(raw) == 17
    with (output / "timeseries.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert sum(int(row["started"]) for row in rows if row["op_type"] == "all") == len(
        raw
    )
    assert sum(int(row["completed"]) for row in rows if row["op_type"] == "all") == len(
        raw
    )
    with (output / "server_metrics_timeseries.csv").open() as stream:
        server_rows = list(csv.DictReader(stream))
    assert server_rows
    summary = json.loads((output / "summary.json").read_text())
    server = summary["server_metrics"]
    assert server["status"] == "ok"
    assert server["timeseries"]["failed_scrapes"] == 0
    http_series = next(
        row
        for row in server["rows"]
        if row["series"].startswith("http_request")
        and "/memories/search" in row["series"]
    )
    assert http_series["delta_count"] == len(raw)
    assert sum(
        float(row["delta_count"] or 0)
        for row in server_rows
        if row["series"] == http_series["series"]
    ) == len(raw)
