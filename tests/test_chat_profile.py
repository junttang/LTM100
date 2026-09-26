"""Tests for chat-replay user-group workload profiles."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
import yaml

from ltm100.cli import _build_scenario, _run_metadata, build_parser
from ltm100.common import MemoryItem, ResultItem, Turn, UserId
from ltm100.core.chat_profile import load_chat_profile
from ltm100.core.config import RunConfig
from ltm100.core.multiproc import run_shards
from ltm100.core.op import OpResult, OpType
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import ChatReplay
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import write_raw_ndjson


def _write_profile(tmp_path, raw: dict | None = None) -> str:
    profile = raw or {
        "version": 1,
        "defaults": {
            "think": 0.0,
            "search_every": 1,
            "answer_time": 0.0,
            "user_gap": 0.0,
        },
        "groups": [
            {"name": "standard", "share": 0.8, "top_k": 20},
            {"name": "power", "share": 0.15, "top_k": 50},
            {"name": "intensive", "share": 0.05, "top_k": 100},
        ],
    }
    path = tmp_path / "chat-profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return str(path)


def _load(path: str):
    return load_chat_profile(
        path,
        think=0.05,
        search_every=1,
        answer_time=0.0,
        user_gap=0.0,
        top_k=20,
    )


def _profile_shard_entry(args: dict, proc_index: int) -> list[OpResult]:
    users = [f"u{index}" for index in range(args["users"])]
    profile = _load(args["profile_path"])
    profile.assign(users, seed=args["seed"])
    now = float(proc_index)
    return [
        OpResult(
            OpType.SEARCH,
            user,
            now,
            now,
            "ok",
            group=profile.group_for(user).name,
        )
        for user in users[proc_index :: args["procs"]]
    ]


class DialogueDataset:
    name = "dialogue"

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{index}" for index in range(n_users)]

    def memory_stream(self, user: UserId):
        yield MemoryItem(content=f"{user} memory")

    def turn_stream(self, user: UserId):
        yield Turn(role="user", items=[MemoryItem(content=f"{user} question")])
        yield Turn(role="assistant", items=[MemoryItem(content=f"{user} answer")])

    def session_stream(self, user: UserId):
        yield list(self.turn_stream(user))


class QueryBackend:
    name = "query"

    def __init__(self) -> None:
        self.top_k_by_user: dict[str, int] = {}

    async def setup(self, users):
        return

    async def add(self, user, items):
        await asyncio.sleep(0)
        return ["id"]

    async def search(self, user, query):
        await asyncio.sleep(0)
        self.top_k_by_user[user] = query.top_k
        return [ResultItem(content="hit")]

    async def teardown(self, users, *, delete):
        return


def test_profile_assigns_exact_counts_deterministically(tmp_path):
    users = [f"u{index}" for index in range(20)]
    first = _load(_write_profile(tmp_path))
    second = _load(_write_profile(tmp_path))

    first.assign(users, seed=7)
    second.assign(users, seed=7)

    first_groups = {user: first.group_for(user).name for user in users}
    second_groups = {user: second.group_for(user).name for user in users}
    assert first_groups == second_groups
    assert list(first_groups.values()).count("standard") == 16
    assert list(first_groups.values()).count("power") == 3
    assert list(first_groups.values()).count("intensive") == 1


def test_profile_assignment_is_resolved_before_process_sharding(tmp_path):
    users = [f"u{index}" for index in range(20)]
    assignments = []
    for proc_index in range(2):
        profile = _load(_write_profile(tmp_path))
        profile.assign(users, seed=11)
        shard = users[proc_index::2]
        assignments.extend((user, profile.group_for(user).name) for user in shard)

    reference = _load(_write_profile(tmp_path))
    reference.assign(users, seed=11)
    assert dict(assignments) == {
        user: reference.group_for(user).name for user in users
    }


def test_profile_assignment_survives_spawned_process_sharding(tmp_path):
    profile_path = _write_profile(tmp_path)
    raw = run_shards(
        _profile_shard_entry,
        {"users": 20, "seed": 11, "procs": 2, "profile_path": profile_path},
        2,
    )
    reference = _load(profile_path)
    users = [f"u{index}" for index in range(20)]
    reference.assign(users, seed=11)

    assert len(raw) == 20
    assert {result.user_id: result.group for result in raw} == {
        user: reference.group_for(user).name for user in users
    }


@pytest.mark.asyncio
async def test_group_top_k_reaches_queries_and_results_are_tagged(tmp_path):
    profile = _load(_write_profile(tmp_path))
    backend = QueryBackend()
    runner = LoadRunner(
        client=backend,
        dataset=DialogueDataset(),
        scenario=ChatReplay(profile=profile),
        config=RunConfig(users=20, ops=60, seed=3),
    )

    await runner.run()

    assert set(backend.top_k_by_user.values()) == {20, 50, 100}
    raw = runner.recorder.raw()
    assert {result.group for result in raw} == {"standard", "power", "intensive"}
    summary = aggregate(raw)
    assert set(summary["by_group"]) == {"standard", "power", "intensive"}
    assert sum(group["total"] for group in summary["by_group"].values()) == 60


def test_profile_defaults_and_group_overrides(tmp_path):
    profile = _load(
        _write_profile(
            tmp_path,
            {
                "version": 1,
                "defaults": {"top_k": 30, "search_every": 2},
                "groups": [
                    {"name": "standard", "share": 0.75},
                    {"name": "power", "share": 0.25, "top_k": 80},
                ],
            },
        )
    )
    standard, power = profile.groups
    assert standard.settings.top_k == 30
    assert standard.settings.search_every == 2
    assert standard.settings.think == 0.05  # CLI fallback
    assert power.settings.top_k == 80


def test_profile_accepts_concurrent_session_counts(tmp_path):
    profile = _load(
        _write_profile(
            tmp_path,
            {
                "version": 1,
                "groups": [
                    {
                        "name": "intensive",
                        "share": 1.0,
                        "top_k": 100,
                        "concurrent_sessions": 5,
                    }
                ],
            },
        )
    )

    assert profile.groups[0].settings.concurrent_sessions == 5
    assert profile.metadata(10)["groups"][0]["concurrent_sessions"] == 5


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 2}, "version"),
        ({"groups": []}, "non-empty"),
        (
            {
                "groups": [
                    {"name": "a", "share": 0.4},
                    {"name": "b", "share": 0.4},
                ]
            },
            "sum to 1.0",
        ),
        (
            {"groups": [{"name": "a", "share": 1.0, "topk": 20}]},
            "unknown field",
        ),
    ],
)
def test_profile_rejects_invalid_config(tmp_path, change, message):
    raw = {"version": 1, "groups": [{"name": "only", "share": 1.0}]}
    raw.update(change)
    with pytest.raises(ValueError, match=message):
        _load(_write_profile(tmp_path, raw))


def test_chat_profile_cli_scope_and_metadata(tmp_path):
    profile_path = _write_profile(tmp_path)
    parser = build_parser()
    common = ["run", "--config", "unused.yaml", "--duration", "1"]
    args = parser.parse_args(
        [*common, "--scenario", "chat-replay", "--chat-profile", profile_path]
    )
    scenario = _build_scenario(args)
    assert scenario.profile is not None

    metadata = _run_metadata(
        args,
        dataset="longmemeval",
        backend="memmachine",
        build={},
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
    )
    assert [group["users"] for group in metadata["chat_profile"]["groups"]] == [
        8,
        2,
        0,
    ]

    invalid = parser.parse_args(
        [*common, "--scenario", "search-load", "--chat-profile", profile_path]
    )
    with pytest.raises(ValueError, match="only valid"):
        _build_scenario(invalid)


def test_raw_report_includes_group_only_when_present(tmp_path):
    path = tmp_path / "raw.ndjson"
    write_raw_ndjson(
        [
            OpResult(
                OpType.SEARCH,
                "u0",
                1.0,
                2.0,
                "ok",
                group="power",
                session_id=2,
            ),
            OpResult(OpType.ADD, "u1", 2.0, 3.0, "ok"),
        ],
        path,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["group"] == "power"
    assert rows[0]["session_id"] == 2
    assert "group" not in rows[1]
    assert "session_id" not in rows[1]
