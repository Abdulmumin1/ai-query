"""HTTP transport using aiohttp for standard Python environments."""

import atexit
import asyncio
from typing import Any, AsyncIterator

import aiohttp

from .base import HTTPStatusError, HTTPTransport
from .tls import certifi_ssl_context


class AioHTTPTransport(HTTPTransport):
    """HTTP transport using aiohttp for standard Python environments.

    This is the default transport for local development, FastAPI servers,
    and any standard Python environment with socket support.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        atexit.register(self._close_at_exit)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=certifi_ssl_context())
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make a POST request and return JSON response."""
        session = await self._get_session()
        async with session.post(url, json=json, headers=headers) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise HTTPStatusError(
                    resp.status,
                    error_text,
                    headers=dict(resp.headers),
                    url=url,
                )
            return await resp.json()

    async def stream(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Make a POST request and stream response bytes."""
        session = await self._get_session()
        async with session.post(url, json=json, headers=headers) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise HTTPStatusError(
                    resp.status,
                    error_text,
                    headers=dict(resp.headers),
                    url=url,
                )
            async for chunk in resp.content:
                yield chunk

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, str]:
        """Make a GET request and return body bytes with content type."""
        session = await self._get_session()
        async with session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise HTTPStatusError(
                    resp.status,
                    error_text,
                    headers=dict(resp.headers),
                    url=url,
                )
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            return await resp.read(), content_type

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _close_at_exit(self) -> None:
        """Best-effort cleanup for process exit.

        This should not affect normal long-lived agent usage because it only runs
        during interpreter shutdown.
        """
        if self._session is None or self._session.closed:
            return
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # Best-effort shutdown path; ignore if the interpreter/event loop
            # state no longer allows awaiting cleanup.
            pass
