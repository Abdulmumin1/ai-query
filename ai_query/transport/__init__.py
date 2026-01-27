"""HTTP Transport abstraction for ai-query.

This module provides a transport layer that abstracts HTTP client implementations,
enabling ai-query to work across different runtime environments:

- Standard Python (aiohttp) - for local dev, FastAPI, etc.
- Cloudflare Workers (JS fetch) - no socket support needed

Usage:
    from ai_query.transport import get_default_transport, HTTPTransport

    transport = get_default_transport()
    response = await transport.post(url, json_body)
"""

from .base import HTTPTransport

__all__ = ["HTTPTransport", "get_default_transport"]


def get_default_transport() -> HTTPTransport:
    """Get the appropriate transport for the current environment.

    Returns:
        HTTPTransport: AioHTTPTransport for standard Python,
                       WorkerFetchTransport for Cloudflare Workers.
    """
    import os

    if os.environ.get("WORKER_RUNTIME") == "cloudflare":
        from .worker import WorkerFetchTransport

        return WorkerFetchTransport()

    from .aiohttp_transport import AioHTTPTransport

    return AioHTTPTransport()
