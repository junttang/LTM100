"""REST transport over aiohttp for backends exposing an HTTP JSON API.

A thin JSON POST/GET client. Backends that ship only an SDK (no server)
implement LTMClient directly and ignore this layer.
"""

from __future__ import annotations

from typing import Any

import aiohttp


class RestTransport:
    """Stateless JSON-over-HTTP transport with a shared session.

    Methods take an absolute path and return the parsed JSON body. The
    transport raises on HTTP error status, letting the caller decide how to
    surface failures (the runner records them as op errors).
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        # Strip trailing slash so paths join cleanly.
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "RestTransport":
        await self.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert self._session is not None, "transport not opened"
        async with self._session.request(
            method, path, json=json, params=params
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RestError(
                    f"{method} {path} -> {resp.status}: {text[:500]}"
                )
            if not text:
                return {}
            import json as _json

            return _json.loads(text)


class RestError(RuntimeError):
    """Raised when the server returns an HTTP error status."""


__all__ = ["RestTransport", "RestError"]
