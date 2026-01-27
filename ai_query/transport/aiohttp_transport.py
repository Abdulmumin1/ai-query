"""HTTP transport using aiohttp for standard Python environments."""

from typing import Any, AsyncIterator

import aiohttp

from .base import HTTPTransport


class AioHTTPTransport(HTTPTransport):
    """HTTP transport using aiohttp for standard Python environments.

    This is the default transport for local development, FastAPI servers,
    and any standard Python environment with socket support.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
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
                raise Exception(f"HTTP {resp.status}: {error_text}")
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
                raise Exception(f"HTTP {resp.status}: {error_text}")
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
                raise Exception(f"HTTP {resp.status}: {error_text}")
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            return await resp.read(), content_type

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
