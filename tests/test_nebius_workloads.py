"""Nebius corpus contracts through real scenarios and spawned runners."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ltm100.adapters.datasets.nebius import NebiusAdapter
from ltm100.common import ResultItem
from ltm100.core.config import RunConfig
from ltm100.core.memory_growth import NamespacedDataset
from ltm100.core.multiproc import run_shards
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad, ChatReplay, Mixed, SearchLoad


def _write_source(path: Path, count: int = 8) -> None:
    records = []
    for i in range(count):
        records.append(
            {
                "trajectory_id": f"trace-{i}",
                "instance_id": f"example/repo-{i}",
                "repo": "example/repo",
                "exit_status": "submit",
                "resolved": i % 2,
                "trajectory": [
                    {"role": "user", "content": f"Fix issue {i}"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-0",
                                "type": "function",
                                "function": {
                                    "name": "execute_bash",
                                    "arguments": json.dumps(
                                        {"command": f"echo recorded-{i}"}
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "execute_bash",
                        "tool_call_id": "call-0",
                        "content": f"recorded-{i}",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "finish-0",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps(
                                        {"message": f"Fixed issue {i}"}
                                    ),
                                },
                            }
                        ],
                    },
                ],
            }
        )
    path.write_text(json.dumps(records))


class _CaptureBackend:
    name = "nebius-capture"

    def __init__(self):
        self.adds = {}
        self.queries = {}
        self.setup_users = []

    async def setup(self, users):
        self.setup_users = users

    async def add(self, user, items):
        self.adds.setdefault(user, []).extend(asdict(item) for item in items)
        await asyncio.sleep(0)
        return ["id"] * len(items)

    async def search(self, user, query):
        self.queries.setdefault(user, []).append(query.query)
        await asyncio.sleep(0)
        return [ResultItem(content="hit")]


def _nebius_entry(args: dict, proc_index: int):
    async def run():
        dataset = NebiusAdapter(path=args["source"])
        backend = _CaptureBackend()
        scenario = {
            "add-load": AddLoad(),
            "search-load": SearchLoad(query_limit=3),
            "mixed": Mixed(think=0),
        }[args["scenario"]]
        runner = LoadRunner(
            client=backend,
            dataset=dataset,
            scenario=scenario,
            config=RunConfig(
                users=4,
                seed=args["seed"],
                ops=64 // args["procs"],
                procs=args["procs"],
                proc_index=proc_index,
                global_concurrency=2,
                preingest=args["scenario"] != "add-load",
                preingest_items_per_user=4 if args["scenario"] != "add-load" else None,
            ),
        )
        results = await runner.run()
        Path(args["traces"], f"{proc_index}.json").write_text(
            json.dumps(
                {
                    "corpus": {
                        user: [asdict(item) for item in dataset.memory_stream(user)]
                        for user in runner.users
                    },
                    "tasks": {
                        user: [task.trajectory_id for task in dataset.task_stream(user)]
                        for user in runner.users
                    },
                    "adds": backend.adds,
                    "queries": backend.queries,
                }
            )
        )
        return results

    return asyncio.run(run())


@pytest.mark.parametrize("scenario", ["add-load", "search-load", "mixed"])
@pytest.mark.parametrize("seed", [0, 42, -7])
@pytest.mark.parametrize("task_count", [2, 8])
def test_assignment_and_workload_corpus_match_across_process_counts(
    tmp_path, scenario, seed, task_count
):
    source = tmp_path / "source.json"
    _write_source(source, task_count)
    reference = NebiusAdapter(path=str(source))
    users = reference.users(4, seed=seed)
    expected = {
        user: [asdict(item) for item in reference.memory_stream(user)] for user in users
    }
    expected_tasks = {
        user: [task.trajectory_id for task in reference.task_stream(user)]
        for user in users
    }
    for procs in (1, 2):
        traces = tmp_path / str(procs)
        traces.mkdir()
        results = run_shards(
            _nebius_entry,
            {
                "source": str(source),
                "scenario": scenario,
                "seed": seed,
                "procs": procs,
                "traces": str(traces),
            },
            procs,
        )
        assert len(results) == 64
        assert all(result.status == "ok" for result in results)
        actual = {}
        actual_tasks = {}
        for proc in range(procs):
            trace = json.loads((traces / f"{proc}.json").read_text())
            assert not actual.keys() & trace["corpus"].keys()
            actual.update(trace["corpus"])
            actual_tasks.update(trace["tasks"])
            for user, adds in trace["adds"].items():
                assert all(item in expected[user] for item in adds)
            for user, queries in trace["queries"].items():
                pool = (
                    expected[user][:3] if scenario == "search-load" else expected[user]
                )
                assert all(
                    query in {item["content"] for item in pool} for query in queries
                )
        assert actual == expected
        assert actual_tasks == expected_tasks


async def test_malformed_source_fails_before_backend_setup(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('[{"trajectory_id": "bad"}]')
    backend = _CaptureBackend()
    runner = LoadRunner(
        client=backend,
        dataset=NebiusAdapter(path=str(source)),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=3),
    )
    with pytest.raises(ValueError, match="row 0"):
        await runner.run()
    assert backend.setup_users == []
    assert backend.adds == {}


async def test_chat_replay_rejects_nebius_before_setup(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source, 1)
    backend = _CaptureBackend()
    runner = LoadRunner(
        client=backend,
        dataset=NebiusAdapter(path=str(source)),
        scenario=ChatReplay(),
        config=RunConfig(users=1, ops=3),
    )
    with pytest.raises(ValueError, match="turn_stream"):
        await runner.run()
    assert backend.setup_users == []


def test_memory_growth_namespace_preserves_corpus_and_producer(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source, 2)
    original = NebiusAdapter(path=str(source))
    users = original.users(2, seed=42)
    namespaced = NamespacedDataset(NebiusAdapter(path=str(source)), "point0")
    scoped = namespaced.users(2, seed=42)
    for user, scoped_user in zip(users, scoped, strict=True):
        baseline = list(original.memory_stream(user))
        items = list(namespaced.memory_stream(scoped_user))
        assert [item.content for item in items] == [item.content for item in baseline]
        assert all(item.producer == scoped_user for item in items)


async def test_warmup_preserves_exact_measured_count(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source, 2)
    backend = _CaptureBackend()
    runner = LoadRunner(
        client=backend,
        dataset=NebiusAdapter(path=str(source)),
        scenario=AddLoad(),
        config=RunConfig(users=2, ops=20, warmup=0.02),
    )
    results = await runner.run()
    assert len(results) == 20
    assert sum(map(len, backend.adds.values())) > 20


async def test_open_model_uses_existing_session_budget(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source, 2)
    runner = LoadRunner(
        client=_CaptureBackend(),
        dataset=NebiusAdapter(path=str(source)),
        scenario=AddLoad(),
        config=RunConfig(
            users=2,
            duration=0.15,
            model="open",
            arrival_rate=100,
            session_ops=3,
            seed=0,
        ),
    )
    results = await runner.run()
    assert len(results) > 0
    assert all(result.status == "ok" for result in results)
    assert {result.user_id for result in results} <= {
        "neb_user_00000",
        "neb_user_00001",
    }


def test_real_local_parquet_matches_json(tmp_path):
    pytest.importorskip("datasets")
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.json"
    _write_source(source, 3)
    path = tmp_path / "source.parquet"
    parquet.write_table(arrow.Table.from_pylist(json.loads(source.read_text())), path)
    snapshots = []
    for adapter in (
        NebiusAdapter(path=str(source)),
        NebiusAdapter(path=str(path), cache_dir=str(tmp_path / "cache")),
    ):
        users = adapter.users(2, seed=42)
        snapshots.append(
            {
                user: {
                    "tasks": list(adapter.task_stream(user)),
                    "memories": list(adapter.memory_stream(user)),
                }
                for user in users
            }
        )
    assert snapshots[0] == snapshots[1]
