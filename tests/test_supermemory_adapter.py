"""Supermemory REST contract tests; the fake server is not an E2E substitute."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from ltm100.adapters.backends.supermemory import SupermemoryClient
from ltm100.adapters.transports.rest import RestError
from ltm100.common import MemoryItem, QueryItem
from ltm100.config import AdapterConfig, build_backend, load_config

REQUESTS = web.AppKey("requests", list)
STORE = web.AppKey("store", dict)
FAILURES = web.AppKey("failures", dict)
DELAY = web.AppKey("delay", dict)


def make_app() -> web.Application:
    app = web.Application()
    app[REQUESTS] = []
    app[STORE] = {}
    app[FAILURES] = {}
    app[DELAY] = {"seconds": 0.0}

    async def add(request):
        body = await request.json()
        app[REQUESTS].append(("add", body, dict(request.headers)))
        if app[DELAY]["seconds"]:
            await asyncio.sleep(app[DELAY]["seconds"])
        if "add" in app[FAILURES]:
            return web.json_response({"error": "injected"}, status=app[FAILURES]["add"])
        rows = app[STORE].setdefault(body["containerTag"], [])
        created = []
        for item in body["memories"]:
            row = {
                "id": f"{body['containerTag']}-{len(rows)}",
                "memory": item["content"],
                "metadata": item["metadata"],
                "similarity": 0.9,
            }
            rows.append(row)
            created.append(row)
        return web.json_response({"documentId": None, "memories": created}, status=201)

    async def search(request):
        body = await request.json()
        app[REQUESTS].append(("search", body, dict(request.headers)))
        if "search" in app[FAILURES]:
            return web.json_response(
                {"error": "injected"}, status=app[FAILURES]["search"]
            )
        rows = app[STORE].get(body["containerTag"], [])
        for condition in body.get("filters", {}).get("AND", []):
            rows = [
                r
                for r in rows
                if r["metadata"].get(condition["key"]) == condition["value"]
            ]
        rows = rows[: body["limit"]]
        return web.json_response({"results": rows, "total": len(rows), "timing": 1})

    async def delete(request):
        tag = request.match_info["tag"]
        app[REQUESTS].append(("delete", tag, dict(request.headers)))
        if tag not in app[STORE]:
            return web.json_response({"error": "not found"}, status=404)
        del app[STORE][tag]
        return web.json_response({"success": True})

    app.router.add_post("/v4/memories", add)
    app.router.add_post("/v4/search", search)
    app.router.add_delete("/v3/container-tags/{tag}", delete)
    return app


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "local-contract-test")


@pytest.fixture
async def supermemory_server():
    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        await runner.cleanup()


def test_registry_and_example_resolve():
    client = build_backend(AdapterConfig("supermemory", {}))
    assert isinstance(client, SupermemoryClient)
    cfg = load_config("examples/supermemory.yaml")
    assert isinstance(build_backend(cfg.backend), SupermemoryClient)


def test_key_is_required_and_can_use_custom_environment(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY")
    with pytest.raises(ValueError, match="SUPERMEMORY_API_KEY"):
        SupermemoryClient()
    monkeypatch.setenv("CUSTOM_SM_KEY", "secret-not-in-config")
    client = SupermemoryClient(api_key_env="CUSTOM_SM_KEY")
    assert client._transport.headers == {"Authorization": "Bearer secret-not-in-config"}


@pytest.mark.parametrize(
    "options",
    [
        {"user_prefix": ""},
        {"user_prefix": "x" * 36},
        {"user_prefix": "bad/slash"},
        {"user_prefix": "사용자"},
        {"api_key_env": ""},
        {"api_key_env": None},
        {"add_batch_size": 0},
        {"add_batch_size": 101},
        {"add_batch_size": True},
        {"add_batch_size": 1.5},
        {"retries": -1},
        {"retries": True},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"timeout": True},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"rerank": "false"},
    ],
)
def test_invalid_options_fail_early(options):
    with pytest.raises((ValueError, TypeError), match="Supermemory"):
        SupermemoryClient(**options)


def test_tags_are_deterministic_bounded_and_do_not_alias():
    client = SupermemoryClient(user_prefix="p" * 35)
    users = [
        "u0",
        "u1",
        "user/a",
        "user_a",
        "사용자",
        "x" * 500,
        "sweep-point-1-user0",
        "sweep-point-2-user0",
    ]
    tags = [client._container_tag(user) for user in users]
    assert len(set(tags)) == len(users)
    assert all(
        len(tag) == 100 and re.fullmatch(r"[a-zA-Z0-9_:-]+", tag) for tag in tags
    )
    assert tags == [
        SupermemoryClient(user_prefix="p" * 35)._container_tag(u) for u in users
    ]
    assert SupermemoryClient(user_prefix="other")._container_tag("u0") != tags[0]


async def test_setup_empty_add_and_no_delete_do_not_write(supermemory_server):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        await client.setup(["u0"])
        assert await client.add("u0", []) == []
        await client.teardown(["u0"], delete=False)
    assert app[REQUESTS] == []
    assert client._transport._session is None


async def test_add_preserves_text_metadata_and_speaker_provenance(supermemory_server):
    url, app = supermemory_server
    original = {"category": "a", "number": 7, "flag": True}
    items = [
        MemoryItem(
            "hello",
            role="user",
            producer="human",
            timestamp="2026-10-03",
            metadata=original,
        ),
        MemoryItem("reply", role="assistant", producer="agent"),
    ]
    async with SupermemoryClient(url, add_batch_size=2) as client:
        ids = await client.add("u0", items)
    assert len(ids) == 2 and len(set(ids)) == 2
    _, body, headers = app[REQUESTS][0]
    assert headers["Authorization"] == "Bearer local-contract-test"
    assert [m["content"] for m in body["memories"]] == ["hello", "reply"]
    assert body["memories"][0]["metadata"] == {
        **original,
        "ltm100_role": "user",
        "ltm100_producer": "human",
        "ltm100_timestamp": "2026-10-03",
    }
    assert body["memories"][1]["metadata"] == {
        "ltm100_role": "assistant",
        "ltm100_producer": "agent",
    }
    assert all(m["isStatic"] is False for m in body["memories"])
    assert original == {"category": "a", "number": 7, "flag": True}


async def test_batch_limit_order_and_returned_ids(supermemory_server):
    url, app = supermemory_server
    async with SupermemoryClient(url, add_batch_size=100) as client:
        ids = await client.add("u0", [MemoryItem(str(i)) for i in range(203)])
    batches = [body["memories"] for kind, body, _ in app[REQUESTS] if kind == "add"]
    assert [len(b) for b in batches] == [100, 100, 3]
    assert [m["content"] for b in batches for m in b] == [str(i) for i in range(203)]
    assert len(ids) == 203 and len(set(ids)) == 203


@pytest.mark.parametrize("key", ["ltm100_role", "ltm100_producer", "ltm100_timestamp"])
async def test_reserved_metadata_rejected_before_any_write(supermemory_server, key):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        with pytest.raises(ValueError, match="reserved"):
            await client.add(
                "u0", [MemoryItem("good"), MemoryItem("bad", metadata={key: "x"})]
            )
    assert not app[REQUESTS]


async def test_search_scope_top_k_options_and_metadata_filter(supermemory_server):
    url, app = supermemory_server
    async with SupermemoryClient(url, threshold=0.25, rerank=True) as client:
        await client.add("u0", [MemoryItem("zero", metadata={"category": "a=b"})] * 3)
        await client.add("u1", [MemoryItem("one")])
        results = await client.search(
            "u0", QueryItem("q", top_k=2, filter="metadata.category=a=b")
        )
        assert len(results) == 2 and all(r.content == "zero" for r in results)
        assert all(
            r.uid and r.score == 0.9 and r.metadata == {"category": "a=b"}
            for r in results
        )
        assert await client.search("empty", QueryItem("q", top_k=1)) == []
    body = app[REQUESTS][4][1]
    assert body == {
        "q": "q",
        "containerTag": client._container_tag("u0"),
        "limit": 2,
        "searchMode": "memories",
        "threshold": 0.25,
        "rerank": True,
        "aggregate": False,
        "rewriteQuery": False,
        "filters": {"AND": [{"key": "category", "value": "a=b"}]},
    }


@pytest.mark.parametrize("value", ["broken", "=x", "metadata.=x", "category= "])
async def test_invalid_filter_is_not_silently_ignored(supermemory_server, value):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        with pytest.raises(ValueError, match="filter"):
            await client.search("u0", QueryItem("q", filter=value))
    assert not app[REQUESTS]


@pytest.mark.parametrize("top_k", [0, 101, -1, True, 2.5])
async def test_top_k_is_not_clamped(supermemory_server, top_k):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        with pytest.raises(ValueError, match="top_k"):
            await client.search("u0", QueryItem("q", top_k=top_k))
    assert not app[REQUESTS]


async def test_expand_is_disabled_with_one_warning(supermemory_server, caplog):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        for _ in range(3):
            assert await client.search("u0", QueryItem("q", expand_context=3)) == []
        assert client.supports_server_metrics is False
    assert (
        sum("context expansion is disabled" in r.message for r in caplog.records) == 1
    )
    assert all("include" not in body for _, body, _ in app[REQUESTS])


async def test_delete_is_scoped_idempotent_and_deduplicated(supermemory_server):
    url, app = supermemory_server
    async with SupermemoryClient(url) as client:
        await client.add("u0", [MemoryItem("zero")])
        await client.add("u1", [MemoryItem("one")])
        await client.teardown(["u0", "u0", "empty"], delete=True)
        assert await client.search("u0", QueryItem("q")) == []
        assert [r.content for r in await client.search("u1", QueryItem("q"))] == ["one"]
    assert len([r for r in app[REQUESTS] if r[0] == "delete"]) == 2


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_cleanup_does_not_hide_auth_rate_limit_or_server_errors(status):
    client = SupermemoryClient()
    path = f"/v3/container-tags/{client._container_tag('u0')}"
    client._transport.request = AsyncMock(
        side_effect=RestError(f"DELETE {path} -> {status}: failure")
    )
    with pytest.raises(RestError):
        await client.teardown(["u0"], delete=True)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"memories": None},
        {"memories": [None]},
        {"memories": []},
        {"memories": [{}]},
        {"memories": [{"id": ""}]},
        {"memories": [{"id": 123}]},
        {"memories": [{"id": "a"}, {"id": "b"}]},
    ],
)
async def test_invalid_add_response_is_an_error(response):
    client = SupermemoryClient()
    client._transport.request = AsyncMock(return_value=response)
    with pytest.raises(RuntimeError, match="Supermemory"):
        await client.add("u0", [MemoryItem("a")])


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"results": None},
        {"results": [None]},
        {"results": [{"id": "a"}]},
        {"results": [{"memory": "a"}]},
        {"results": [{"id": "a", "memory": "a", "similarity": float("nan")}]},
        {"results": [{"id": "a", "memory": "a", "similarity": "0.9"}]},
        {"results": [{"id": "a", "memory": "a", "metadata": "x"}]},
    ],
)
async def test_invalid_search_response_is_not_successfully_empty(response):
    client = SupermemoryClient()
    client._transport.request = AsyncMock(return_value=response)
    with pytest.raises(RuntimeError, match="Supermemory"):
        await client.search("u0", QueryItem("q"))


@pytest.mark.parametrize(
    "error", [asyncio.TimeoutError(), RestError("POST /v4/memories -> 429: limit")]
)
async def test_request_errors_propagate(error):
    client = SupermemoryClient()
    client._transport.request = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await client.add("u0", [MemoryItem("a")])


@pytest.mark.parametrize("operation", ["add", "search"])
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_errors_are_not_retried(supermemory_server, operation, status):
    url, app = supermemory_server
    app[FAILURES][operation] = status
    async with SupermemoryClient(url, retries=2) as client:
        with pytest.raises(RestError, match=f"-> {status}:"):
            if operation == "add":
                await client.add("u0", [MemoryItem("a")])
            else:
                await client.search("u0", QueryItem("q"))
    assert len(app[REQUESTS]) == 1


async def test_real_timeout_is_not_a_success(supermemory_server):
    url, app = supermemory_server
    app[DELAY]["seconds"] = 0.05
    async with SupermemoryClient(url, timeout=0.005) as client:
        with pytest.raises(asyncio.TimeoutError):
            await client.add("u0", [MemoryItem("a")])
    assert len(app[REQUESTS]) == 1


async def test_malformed_response_is_recorded_as_runner_error():
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter
    from ltm100.core.config import RunConfig
    from ltm100.core.runner import LoadRunner
    from ltm100.core.scenarios import AddLoad

    client = SupermemoryClient()
    client._transport.request = AsyncMock(return_value={})
    runner = LoadRunner(
        client=client,
        dataset=SyntheticAdapter(memories_per_user=2),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=2),
    )
    results = await runner.run()
    assert len(results) == 2 and all(r.status == "error" for r in results)
    summary = runner.recorder.summary()
    assert summary["successful"] == 0 and summary["errors"] == 2


async def test_http_preingest_failure_aborts_runner(supermemory_server):
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter
    from ltm100.core.config import RunConfig
    from ltm100.core.runner import LoadRunner
    from ltm100.core.scenarios import SearchLoad

    url, app = supermemory_server
    app[FAILURES]["add"] = 500
    async with SupermemoryClient(url) as client:
        runner = LoadRunner(
            client=client,
            dataset=SyntheticAdapter(memories_per_user=2),
            scenario=SearchLoad(),
            config=RunConfig(users=1, ops=2, preingest=True),
        )
        with pytest.raises(RuntimeError, match="[Pp]re.?ingest"):
            await runner.run()
    assert all(kind == "add" for kind, _, _ in app[REQUESTS])


async def test_invalid_empty_metadata_and_excess_results_are_errors():
    client = SupermemoryClient()
    for response in (
        {"results": [{"id": "a", "memory": "a", "metadata": []}]},
        {"results": [{"id": "a", "memory": "a"}] * 2},
    ):
        client._transport.request = AsyncMock(return_value=response)
        with pytest.raises(RuntimeError):
            await client.search("u0", QueryItem("q", top_k=1))
