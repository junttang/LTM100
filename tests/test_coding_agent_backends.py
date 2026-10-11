"""Coding-agent policies through real adapters and local HTTP contract servers.

These contract servers verify payloads; live MemMachine E2E is separate.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from test_coding_agent import AgentSource, task
from test_mem0_adapter import REQUESTS as MEM0_REQUESTS
from test_mem0_adapter import _make_app as mem0_app
from test_memmachine_adapter import _make_app as memmachine_app
from test_supermemory_adapter import REQUESTS as SUPER_REQUESTS
from test_supermemory_adapter import make_app as supermemory_app

from ltm100.adapters.backends.mem0 import Mem0Client
from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.backends.supermemory import SupermemoryClient
from ltm100.core.agent_profile import AgentProfile
from ltm100.core.coding_agent import CodingAgent
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner


@pytest.mark.parametrize("backend", ["memmachine", "mem0", "supermemory"])
@pytest.mark.parametrize(
    "integration,scope",
    [
        ("hook", "user-facing"),
        ("hook", "all-steps"),
        ("mcp", "user-facing"),
        ("hybrid", "user-facing"),
        ("hybrid", "all-steps"),
    ],
)
async def test_policy_adapter_payloads(backend, integration, scope, monkeypatch):
    app = {
        "memmachine": memmachine_app,
        "mem0": mem0_app,
        "supermemory": supermemory_app,
    }[backend]()
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "agent-local-contract")
    client = {
        "memmachine": MemMachineClient,
        "mem0": Mem0Client,
        "supermemory": SupermemoryClient,
    }[backend](url)
    profile = AgentProfile(
        integration=integration,
        hook_scope=scope,
        search_probability=1,
        add_probability=1,
        task_repeat=False,
    )
    run = LoadRunner(
        client=client,
        dataset=AgentSource([task()]),
        scenario=CodingAgent(profile, top_k=50),
        config=RunConfig(users=1, duration=1),
    )
    try:
        async with client:
            results = await run.run()
            assert results and all(result.status == "ok" for result in results)
            expected_roles = (
                {"user", "assistant"}
                if scope == "user-facing" and integration == "hook"
                else {"tool"}
            )
            if backend == "mem0":
                adds = [body for kind, body, _ in app[MEM0_REQUESTS] if kind == "add"]
                assert expected_roles <= {
                    m["role"] for body in adds for m in body["messages"]
                }
                assert all(body["infer"] is False for body in adds)
                assert all(body.get("metadata", {}) == {} for body in adds)
                searches = [
                    body for kind, body, _ in app[MEM0_REQUESTS] if kind == "search"
                ]
                assert all(body["top_k"] == 50 for body in searches)
            elif backend == "supermemory":
                adds = [body for kind, body, _ in app[SUPER_REQUESTS] if kind == "add"]
                assert expected_roles <= {
                    item["metadata"]["ltm100_role"]
                    for body in adds
                    for item in body["memories"]
                }
                assert all(
                    item["metadata"]["ltm100_producer"] == "u0"
                    for body in adds
                    for item in body["memories"]
                )
                searches = [
                    body for kind, body, _ in app[SUPER_REQUESTS] if kind == "search"
                ]
                assert all(body["limit"] == 50 for body in searches)
            await client.teardown(["u0"], delete=True)
    finally:
        await server.cleanup()
