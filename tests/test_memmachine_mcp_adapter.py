"""Tests for the MemMachine MCP adapter.

The MCP transport is faked (no live fastmcp server needed): we record the
tool name + arguments the adapter sends and return canned responses. This
verifies the adapter maps users to org_id/proj_id/user_id tool arguments,
parses the search_memory SearchResult into ResultItems, and counts sent
items for add (the MCP tool returns no ids). Lifecycle (setup/teardown) is
exercised against a tiny fake REST server.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient
from ltm100.common import MemoryItem, QueryItem


class FakeMcpTransport:
    """Records tool calls; returns canned data shaped like fastmcp results."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self) -> FakeMcpTransport:  # noqa: PYI034
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict) -> object:
        self.calls.append((name, dict(arguments)))
        if name == "add_memory":
            return _McpResponse(status=200, message="Success")
        if name == "search_memory":
            return _FakeSearchResult()
        raise AssertionError(f"unexpected tool {name!r}")


class _McpResponse:
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        self.content = None


class _Ep:
    def __init__(self, content: str, score: float, uid: str) -> None:
        self.content = content
        self.score = score
        self.uid = uid
        self.metadata = {}


class _Ltm:
    def __init__(self, episodes) -> None:
        self.episodes = episodes


class _Em:
    """Mimics EpisodicSearchResult: has long_term_memory/short_term_memory."""

    def __init__(self, episodes) -> None:
        self.long_term_memory = _Ltm(episodes)
        self.short_term_memory = _Ltm([])


class _FakeSearchContent:
    """Mimics SearchResultContent: has episodic_memory."""

    def __init__(self, episodes) -> None:
        self.episodic_memory = _Em(episodes)
        self.semantic_memory = None


class _FakeSearchResult:
    """Mimics SearchResult: status defaults to 0 (success), has content."""

    def __init__(self) -> None:
        self.status = 0
        self.message = None
        self.content = _FakeSearchContent([_Ep("recall content", 0.9, "uid-1")])


def _make_rest_app() -> web.Application:
    created: set[tuple[str, str]] = set()

    async def create_project(request: web.Request) -> web.Response:
        body = await request.json()
        key = (body["org_id"], body["project_id"])
        if key in created:
            return web.json_response({"detail": "Project already exists"}, status=409)
        created.add(key)
        return web.json_response({"org_id": key[0], "project_id": key[1]}, status=201)

    async def delete_project(request: web.Request) -> web.Response:
        body = await request.json()
        created.discard((body["org_id"], body["project_id"]))
        return web.json_response({}, status=204)

    app = web.Application()
    app.router.add_post("/api/v2/projects", create_project)
    app.router.add_post("/api/v2/projects/delete", delete_project)
    return app


@pytest.fixture
async def server_url():
    runner = web.AppRunner(_make_rest_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    yield url
    await runner.cleanup()


@pytest.mark.asyncio
async def test_mcp_add_sends_tenant_args(server_url):
    client = MemMachineMcpClient(server_url, org_prefix="ltm100")
    client._mcp = FakeMcpTransport()  # type: ignore[assignment]
    async with client:
        uids = await client.add(
            "alice",
            [MemoryItem(content="hello", producer="alice")],
        )
    assert len(uids) == 1  # one placeholder id per sent item
    name, args = client._mcp.calls[0]
    assert name == "add_memory"
    assert args["content"] == "hello"
    assert args["org_id"] == "ltm100"
    assert args["proj_id"] == "user_alice"
    assert args["user_id"] == "alice"


@pytest.mark.asyncio
async def test_mcp_add_accepts_role_but_tool_contract_omits_it(server_url):
    client = MemMachineMcpClient(server_url, org_prefix="ltm100")
    client._mcp = FakeMcpTransport()  # type: ignore[assignment]
    async with client:
        await client.add(
            "alice",
            [MemoryItem(content="hello", producer="alice", role="assistant")],
        )

    _, args = client._mcp.calls[0]
    assert "role" not in args


@pytest.mark.asyncio
async def test_mcp_add_counts_sent_items(server_url):
    client = MemMachineMcpClient(server_url, org_prefix="ltm100")
    client._mcp = FakeMcpTransport()  # type: ignore[assignment]
    async with client:
        uids = await client.add(
            "bob",
            [MemoryItem(content=f"m{i}", producer="bob") for i in range(3)],
        )
    # One add_memory call per item (no batch form in the MCP tool).
    assert len(client._mcp.calls) == 3
    assert len(uids) == 3  # n_items = len(items) sent


@pytest.mark.asyncio
async def test_mcp_search_parses_episodes(server_url):
    client = MemMachineMcpClient(server_url, org_prefix="ltm100")
    client._mcp = FakeMcpTransport()  # type: ignore[assignment]
    async with client:
        results = await client.search("alice", QueryItem(query="hello", top_k=5))
    assert len(results) == 1
    assert results[0].content == "recall content"
    assert results[0].score == 0.9
    assert results[0].uid == "uid-1"
    name, args = client._mcp.calls[0]
    assert name == "search_memory"
    assert args["query"] == "hello"
    assert args["top_k"] == 5
    assert args["proj_id"] == "user_alice"


@pytest.mark.asyncio
async def test_mcp_setup_teardown_uses_rest(server_url):
    client = MemMachineMcpClient(server_url, org_prefix="ltm100")
    client._mcp = FakeMcpTransport()  # type: ignore[assignment]
    async with client:
        # setup creates projects via REST (not MCP), so no add_memory calls.
        await client.setup(["alice", "bob"])
        assert client._mcp.calls == []
        await client.teardown(["alice", "bob"], delete=True)
        assert client._mcp.calls == []
