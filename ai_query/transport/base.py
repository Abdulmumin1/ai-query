"""Abstract HTTP transport interface for ai-query."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class HTTPTransport(ABC):
    """Abstract HTTP transport for making requests to AI provider APIs.

    This abstraction allows ai-query to work across different runtime environments:
    - Standard Python (using aiohttp)
    - Cloudflare Workers (using JS fetch API)
    - Future: Deno, Bun, browser, etc.
    """

    @abstractmethod
    async def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make a POST request and return JSON response.

        Args:
            url: The URL to POST to.
            json: JSON body to send.
            headers: Optional headers to include.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            Exception: If the request fails or returns non-2xx status.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Make a POST request and stream response bytes.

        Args:
            url: The URL to POST to.
            json: JSON body to send.
            headers: Optional headers to include.

        Yields:
            Raw bytes from the response stream.

        Raises:
            Exception: If the request fails or returns non-2xx status.
        """
        ...

    @abstractmethod
    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, str]:
        """Make a GET request and return body bytes with content type.

        Args:
            url: The URL to GET.
            headers: Optional headers to include.

        Returns:
            Tuple of (body_bytes, content_type).

        Raises:
            Exception: If the request fails or returns non-2xx status.
        """
        ...

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    async def __aenter__(self) -> "HTTPTransport":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
