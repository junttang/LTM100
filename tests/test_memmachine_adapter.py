"""Tests for the MemMachine REST adapter against a fake aiohttp server.

We stand up a tiny aiohttp app implementing the subset of /api/v2 endpoints the
adapter uses, then assert the adapter maps virtual users to org_id/project_id,
batches adds, parses search episodes, and tolerates 409 on setup.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from ltm100.adapters.backends.memmachine import MemMachineClient, _to_message
from ltm100.common import MemoryItem, QueryItem


def _make_app() -> web.Application:
    """A minimal fake MemMachine /api/v2 server."""
    created: set[tuple[str, str]] = set()
    store: dict[tuple[str, str], list[dict]] = {}

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def create_project(request: web.Request) -> web.Response:
        body = await request.json()
        key = (body["org_id"], body["project_id"])
        if key in created:
            return web.json_response(
                {"detail": "Project already exists"}, status=409
            )
        created.add(key)
        store.setdefault(key, [])
        return web.json_response(
            {"org_id": key[0], "project_id": key[1], "description": body.get("description", "")},
            status=201,
        )

    async def add_memories(request: web.Request) -> web.Response:
        body = await request.json()
        key = (body["org_id"], body["project_id"])
        msgs = body.get("messages", [])
        uids = []
        for i, m in enumerate(msgs):
            uid = f"{key[0]}-{key[1]}-{len(store.get(key, [])) + i}"
            store.setdefault(key, []).append({"uid": uid, "content": m["content"]})
            uids.append(uid)
        return web.json_response({"results": [{"uid": u} for u in uids]})

    async def search_memories(request: web.Request) -> web.Response:
        body = await request.json()
        key = (body["org_id"], body["project_id"])
        top_k = body.get("top_k", 10)
        episodes = store.get(key, [])[:top_k]
        return web.json_response(
            {
                "status": 0,
                "content": {
                    "episodic_memory": {
                        "long_term_memory": {"episodes": episodes},
                        "short_term_memory": {"episodes": [], "episode_summary": []},
                    },
                    "semantic_memory": None,
                },
            }
        )

    async def delete_project(request: web.Request) -> web.Response:
        body = await request.json()
        key = (body["org_id"], body["project_id"])
        created.discard(key)
        store.pop(key, None)
        return web.Response(status=204)

    app = web.Application()
    app.router.add_get("/api/v2/health", health)
    app.router.add_post("/api/v2/projects", create_project)
    app.router.add_post("/api/v2/memories", add_memories)
    app.router.add_post("/api/v2/memories/search", search_memories)
    app.router.add_post("/api/v2/projects/delete", delete_project)
    return app


@pytest.fixture
async def server_url():
    runner = web.AppRunner(_make_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    yield url
    await runner.cleanup()


@pytest.mark.asyncio
async def test_setup_creates_per_user_projects(server_url):
    async with MemMachineClient(server_url) as client:
        await client.setup(["u0", "u1"])
        assert client._tenant("u0") == ("ltm100", "user_u0")
        assert client._tenant("u1") == ("ltm100", "user_u1")


@pytest.mark.asyncio
async def test_setup_idempotent_on_existing_project(server_url):
    async with MemMachineClient(server_url) as client:
        await client.setup(["u0"])
        # Second setup against same tenant must not raise.
        await client.setup(["u0"])


@pytest.mark.asyncio
async def test_add_returns_uids_per_item(server_url):
    async with MemMachineClient(server_url, add_batch_size=2) as client:
        await client.setup(["u0"])
        items = [MemoryItem(content=f"mem {i}", producer="u0") for i in range(5)]
        uids = await client.add("u0", items)
        assert len(uids) == 5
        assert all(isinstance(u, str) and u for u in uids)


def test_add_message_preserves_role():
    assert _to_message(
        MemoryItem(content="answer", producer="u0", role="assistant")
    ) == {
        "content": "answer",
        "producer": "u0",
        "role": "assistant",
    }


@pytest.mark.asyncio
async def test_search_returns_only_that_users_episodes(server_url):
    async with MemMachineClient(server_url) as client:
        await client.setup(["u0", "u1"])
        await client.add("u0", [MemoryItem(content="u0-only", producer="u0")])
        await client.add("u1", [MemoryItem(content="u1-only", producer="u1")])
        res0 = await client.search("u0", QueryItem(query="anything"))
        res1 = await client.search("u1", QueryItem(query="anything"))
        assert [r.content for r in res0] == ["u0-only"]
        assert [r.content for r in res1] == ["u1-only"]


@pytest.mark.asyncio
async def test_search_respects_top_k(server_url):
    async with MemMachineClient(server_url) as client:
        await client.setup(["u0"])
        await client.add("u0", [MemoryItem(content=f"m{i}", producer="u0") for i in range(10)])
        res = await client.search("u0", QueryItem(query="q", top_k=3))
        assert len(res) == 3


@pytest.mark.asyncio
async def test_teardown_delete_clears_user(server_url):
    async with MemMachineClient(server_url) as client:
        await client.setup(["u0"])
        await client.add("u0", [MemoryItem(content="x", producer="u0")])
        await client.teardown(["u0"], delete=True)
        # After delete, search returns nothing.
        res = await client.search("u0", QueryItem(query="q"))
        assert res == []
