"""Opt-in seeded workloads against a disposable standalone MemMachine server.

Set LTM100_MEMMACHINE_E2E_URL to run. Stored contents are checked directly;
mock embeddings suffice because these tests do not evaluate search quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.common import QueryItem

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_MEMMACHINE_E2E_URL"),
    reason="requires an explicitly configured disposable MemMachine server",
)


async def _cli(*args):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "ltm100.cli",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == 0, (stdout.decode(), stderr.decode())


def _config(path, *, memories=3):
    options = {
        "base_url": os.environ["LTM100_MEMMACHINE_E2E_URL"],
        "org_prefix": f"seed_{uuid.uuid4().hex}",
        "timeout": 30,
    }
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "synthetic",
                    "memories_per_user": memories,
                    "content_chars": 80,
                    "categories": 3,
                },
                "backend": {"name": "memmachine", **options},
            }
        )
    )
    return options


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("scenario", ["add-load", "search-load"])
async def test_live_seeded_corpus_matches_single_and_multiprocess_runs(
    tmp_path, seed, scenario
):
    dataset = SyntheticAdapter(memories_per_user=3, content_chars=80, categories=3)
    expected = {
        user: {item.content for item in dataset.memory_stream(user)}
        for user in dataset.users(2, seed=seed)
    }
    snapshots = []
    for procs in (1, 2):
        config = tmp_path / f"config_{procs}.yaml"
        options = _config(config)
        output = tmp_path / f"out_{procs}"
        args = [
            "run",
            "--config",
            str(config),
            "--scenario",
            scenario,
            "--users",
            "2",
            "--procs",
            str(procs),
            "--seed",
            str(seed),
            "--ops",
            "12",
            "--top-k",
            "3",
            "--raw",
            "--output",
            str(output),
            "--no-delete-on-exit",
        ]
        if scenario == "search-load":
            args.extend(
                ["--preingest", "--preingest-items-per-user", "3", "--query-limit", "3"]
            )
        async with MemMachineClient(**options) as client:
            try:
                await _cli(*args)
                summary = json.loads((output / "summary.json").read_text())
                assert summary["meta"]["seed"] == seed
                raw = [
                    json.loads(line)
                    for line in (output / "raw.ndjson").read_text().splitlines()
                ]
                assert len(raw) == 12
                assert all(row["status"] == "ok" for row in raw)
                if scenario == "search-load":
                    assert summary["meta"]["preingest_stats"]["input_items"] == 6
                stored = {}
                for user, contents in expected.items():
                    stored[user] = set()
                    for content in contents:
                        hits = await client.search(user, QueryItem(content, top_k=20))
                        returned = {hit.content for hit in hits}
                        assert content in returned
                        assert returned <= contents
                        stored[user].update(returned)
                assert stored == expected
                snapshots.append(stored)
            finally:
                await client.teardown(list(expected), delete=True)
    assert snapshots[0] == snapshots[1]


async def test_live_seeded_memory_growth_sweep(tmp_path):
    config = tmp_path / "config.yaml"
    _config(config, memories=6)
    output = tmp_path / "sweep"
    await _cli(
        "sweep",
        "memory-growth",
        "--config",
        str(config),
        "--memory-counts",
        "3,6",
        "--queries-per-user",
        "3",
        "--users",
        "2",
        "--procs",
        "2",
        "--seed",
        "42",
        "--ops",
        "12",
        "--repetitions",
        "2",
        "--top-k",
        "3",
        "--raw",
        "--output",
        str(output),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["config"]["seed"] == 42
    for point in manifest["points"]:
        for repeat in point["repetitions"]:
            assert repeat["status"] == "ok"
            assert repeat["metrics"]["input_items"] == point["memory_count"] * 2
            raw = [
                json.loads(line)
                for line in (output / repeat["output"] / "raw.ndjson")
                .read_text()
                .splitlines()
            ]
            assert len(raw) == 12
            assert all(row["status"] == "ok" and row["n_items"] > 0 for row in raw)
