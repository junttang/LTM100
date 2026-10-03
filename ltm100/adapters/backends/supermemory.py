"""Supermemory REST adapter using synchronous direct-memory creation.

Document ingestion/extraction is intentionally not used: acknowledging a queued
document would not measure completion of a memory add. Each virtual user has a
deterministic container tag shared by add, search, and cleanup.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any

from ltm100.adapters.transports.rest import RestError, RestTransport
from ltm100.common import MemoryItem, QueryItem, ResultItem, UserId

_LOG = logging.getLogger(__name__)
_PROVENANCE = ("ltm100_role", "ltm100_producer", "ltm100_timestamp")


class SupermemoryClient:
    """LTMClient adapter for Supermemory's hosted or local REST endpoint."""

    name = "supermemory"
    supports_server_metrics = False

    def __init__(
        self,
        base_url: str = "http://localhost:6767",
        *,
        user_prefix: str = "ltm100",
        api_key_env: str = "SUPERMEMORY_API_KEY",
        timeout: float = 60.0,
        retries: int = 0,
        add_batch_size: int = 1,
        threshold: float = 0.0,
        rerank: bool = False,
    ) -> None:
        if not isinstance(user_prefix, str) or not re.fullmatch(
            r"[a-zA-Z0-9_:-]{1,35}", user_prefix
        ):
            raise ValueError("Supermemory user_prefix must be 1-35 tag-safe characters")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError(
                "Supermemory api_key_env must name an environment variable"
            )
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Supermemory requires an API key in {api_key_env}")
        _integer_range(add_batch_size, "add_batch_size", 1, 100)
        _integer_range(retries, "retries", 0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Supermemory timeout must be finite and positive")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Supermemory threshold must be finite and in [0, 1]")
        if not isinstance(rerank, bool):
            raise TypeError("Supermemory rerank must be a boolean")
        self.user_prefix = user_prefix
        self.add_batch_size = add_batch_size
        self.threshold = threshold
        self.rerank = rerank
        self._warned_expand = False
        self._transport = RestTransport(
            base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            retries=retries,
        )

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> SupermemoryClient:  # noqa: PYI034
        await self._transport.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._transport.close()

    def _container_tag(self, user: UserId) -> str:
        # Hash the complete ID, including sweep namespaces. Sanitizing/truncating
        # arbitrary IDs could merge tenants; a full digest also fits the API limit.
        digest = hashlib.sha256(user.encode("utf-8")).hexdigest()
        return f"{self.user_prefix}_{digest}"

    # -- LTMClient ---------------------------------------------------------

    async def setup(self, users: list[UserId]) -> None:
        # Containers are created lazily. Do not reset existing server state here.
        return None

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        # Validate every item before writing any batch, preserving input metadata
        # and text without converting speaker identity into content prefixes.
        memories = []
        for item in items:
            if any(key in item.metadata for key in _PROVENANCE):
                raise ValueError("Supermemory metadata uses a reserved ltm100_* key")
            metadata = dict(item.metadata)
            for key, value in zip(
                _PROVENANCE, (item.role, item.producer, item.timestamp)
            ):
                if value is not None:
                    metadata[key] = value
            memories.append(
                {"content": item.content, "isStatic": False, "metadata": metadata}
            )
        tag = self._container_tag(user)
        ids: list[str] = []
        for offset in range(0, len(memories), self.add_batch_size):
            batch = memories[offset : offset + self.add_batch_size]
            response = await self._transport.request(
                "POST", "/v4/memories", json={"containerTag": tag, "memories": batch}
            )
            rows = _rows(response, "memories")
            if len(rows) != len(batch):
                raise RuntimeError("Supermemory add did not return one ID per input")
            ids.extend(_uid(row) for row in rows)
        return ids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        # Never silently clamp a workload's recall depth to a different value.
        _integer_range(query.top_k, "top_k", 1, 100)
        payload: dict[str, Any] = {
            "q": query.query,
            "containerTag": self._container_tag(user),
            "limit": query.top_k,
            "searchMode": "memories",
            "threshold": self.threshold,
            "rerank": self.rerank,
            "aggregate": False,
            "rewriteQuery": False,
        }
        if query.filter:
            payload["filters"] = _parse_filter(query.filter)
        if query.expand_context and not self._warned_expand:
            _LOG.warning(
                "Supermemory does not support expand_context; context expansion "
                "is disabled for this backend (requested=%s)",
                query.expand_context,
            )
            self._warned_expand = True
        response = await self._transport.request("POST", "/v4/search", json=payload)
        results = []
        rows = _rows(response, "results")
        if len(rows) > query.top_k:
            raise RuntimeError("Supermemory returned more results than requested top_k")
        for row in rows:
            content = row.get("memory")
            metadata = row.get("metadata")
            if metadata is None:
                metadata = {}
            score = row.get("similarity")
            if not isinstance(content, str) or not isinstance(metadata, dict):
                raise RuntimeError(  # noqa: TRY004 - malformed remote response
                    "Supermemory returned an invalid memory result"
                )
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RuntimeError("Supermemory returned an invalid similarity score")
            results.append(
                ResultItem(
                    content=content, score=score, uid=_uid(row), metadata=metadata
                )
            )
        return results

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        if not delete:
            return
        for user in dict.fromkeys(users):
            path = f"/v3/container-tags/{self._container_tag(user)}"
            try:
                await self._transport.request("DELETE", path)
            except RestError as exc:
                # RestTransport exposes HTTP status in its error prefix. Ignore
                # only this exact DELETE's 404 (empty/already-deleted container).
                if not str(exc).startswith(f"DELETE {path} -> 404:"):
                    raise


# -- helpers ---------------------------------------------------------------


def _integer_range(
    value: int, name: str, minimum: int, maximum: int | None = None
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f"{minimum}-{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"Supermemory {name} must be an integer in {bound}")


def _rows(response: Any, key: str) -> list[dict[str, Any]]:
    rows = response.get(key) if isinstance(response, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Supermemory response requires a '{key}' array of objects")
    return rows


def _uid(row: dict[str, Any]) -> str:
    uid = row.get("id")
    if not isinstance(uid, str) or not uid:
        raise RuntimeError("Supermemory returned a missing or invalid memory ID")
    return uid


def _parse_filter(value: str) -> dict[str, Any]:
    """Translate the existing exact-match syntax without touching tenant scope."""
    if "=" not in value:
        raise ValueError("Supermemory filter must use metadata.key=value syntax")
    key, expected = value.split("=", 1)
    key = key.strip().removeprefix("metadata.")
    expected = expected.strip()
    if not key or not expected:
        raise ValueError("Supermemory filter must have a non-empty key and value")
    return {"AND": [{"key": key, "value": expected}]}


__all__ = ["SupermemoryClient"]
