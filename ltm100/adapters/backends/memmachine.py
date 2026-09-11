"""MemMachine backend adapter (REST /api/v2).

MemMachine's multi-tenancy is `session_key = f"{org_id}/{project_id}"`; every
add/search is scoped by it. We map each LTM100 virtual user to one project:

    UserId  ->  org_id, project_id

All requests carry that pair so a user only ever sees their own memories. The
adapter is async (aiohttp via RestTransport) and lives entirely behind the
LTMClient Protocol, so the load core never touches MemMachine specifics.

Endpoints used:
  POST /api/v2/projects           create a project (per-user tenant)
  POST /api/v2/projects/delete    delete a project
  POST /api/v2/memories           add memories (episodic)
  POST /api/v2/memories/search    search memories (episodic)
  GET  /api/v2/health             readiness check
"""

from __future__ import annotations

import logging
from typing import Any

from ltm100.adapters.transports.rest import RestError, RestTransport
from ltm100.common import MemoryItem, QueryItem, ResultItem, UserId

logger = logging.getLogger(__name__)

# We only ingest/search episodic memory in this benchmark.
_EPISODIC_TYPES = ["episodic"]


class MemMachineClient:
    """LTMClient adapter for MemMachine over REST."""

    name = "memmachine"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        org_prefix: str = "ltm100",
        timeout: float = 60.0,
        add_batch_size: int = 50,
        retries: int = 0,
    ) -> None:
        self.org_prefix = org_prefix
        self.add_batch_size = add_batch_size
        self._transport = RestTransport(base_url, timeout=timeout, retries=retries)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "MemMachineClient":
        await self._transport.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._transport.close()

    def _tenant(self, user: UserId) -> tuple[str, str]:
        """Map a virtual user to (org_id, project_id).

        We pin every virtual user under a single org and use the user id as
        the project id. A run's users thus share an org but have distinct
        projects, giving per-user memory isolation."""
        return self.org_prefix, f"user_{user}"

    async def health(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/api/v2/health")

    # -- LTMClient --------------------------------------------------------

    async def setup(self, users: list[UserId]) -> None:
        # Create one project per user; 409 (already exists) is acceptable so
        # reruns against the same tenant don't fail.
        for user in users:
            org_id, project_id = self._tenant(user)
            try:
                await self._transport.request(
                    "POST",
                    "/api/v2/projects",
                    json={
                        "org_id": org_id,
                        "project_id": project_id,
                        "description": f"ltm100 virtual user {user}",
                    },
                )
            except RestError as e:
                if "409" in str(e) or "already exists" in str(e).lower():
                    logger.debug("project %s/%s already exists", org_id, project_id)
                else:
                    raise

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        org_id, project_id = self._tenant(user)
        uids: list[str] = []
        for start in range(0, len(items), self.add_batch_size):
            batch = items[start : start + self.add_batch_size]
            payload = {
                "org_id": org_id,
                "project_id": project_id,
                "types": _EPISODIC_TYPES,
                "messages": [_to_message(it) for it in batch],
            }
            resp = await self._transport.request("POST", "/api/v2/memories", json=payload)
            for r in resp.get("results", []):
                uids.append(r.get("uid", ""))
        return uids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        org_id, project_id = self._tenant(user)
        payload: dict[str, Any] = {
            "org_id": org_id,
            "project_id": project_id,
            "query": query.query,
            "top_k": query.top_k,
            "types": _EPISODIC_TYPES,
        }
        # Omitted rather than sent as 0/"" so a default run's payload is
        # unchanged and the server applies its own defaults.
        if query.expand_context:
            payload["expand_context"] = query.expand_context
        if query.filter:
            payload["filter"] = query.filter
        resp = await self._transport.request("POST", "/api/v2/memories/search", json=payload)
        return _parse_episodes(resp)

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        if not delete:
            return
        for user in users:
            org_id, project_id = self._tenant(user)
            try:
                await self._transport.request(
                    "POST",
                    "/api/v2/projects/delete",
                    json={"org_id": org_id, "project_id": project_id},
                )
            except RestError as e:
                logger.debug("delete project %s/%s failed: %s", org_id, project_id, e)


# -- helpers -----------------------------------------------------------------


def _to_message(item: MemoryItem) -> dict[str, Any]:
    msg: dict[str, Any] = {"content": item.content}
    if item.producer is not None:
        msg["producer"] = item.producer
    if item.role:
        msg["role"] = item.role
    if item.timestamp:
        msg["timestamp"] = item.timestamp
    if item.metadata:
        # MemMachine metadata values must be strings.
        msg["metadata"] = {k: str(v) for k, v in item.metadata.items()}
    return msg


def _parse_episodes(resp: dict[str, Any]) -> list[ResultItem]:
    """Extract long-term episodic episodes from a /memories/search response."""
    content = resp.get("content", {}) or {}
    em = content.get("episodic_memory")
    if not em:
        return []
    ltm = em.get("long_term_memory", {}) or {}
    episodes = ltm.get("episodes", []) or []
    results: list[ResultItem] = []
    for ep in episodes:
        results.append(
            ResultItem(
                content=ep.get("content", ""),
                score=ep.get("score"),
                uid=ep.get("uid") or ep.get("id"),
                metadata=ep.get("metadata", {}) or {},
            )
        )
    return results


__all__ = ["MemMachineClient"]
