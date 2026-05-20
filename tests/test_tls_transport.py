"""Tests for TLS verification wiring in HTTP transports."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_certifi_ca_bundle_uses_certifi():
    from ai_query.transport.tls import certifi_ca_bundle

    assert isinstance(certifi_ca_bundle(), str)


@pytest.mark.asyncio
async def test_aiohttp_transport_uses_certifi_ssl_context():
    from ai_query.transport.aiohttp_transport import AioHTTPTransport

    session = MagicMock()
    session.closed = False

    with (
        patch("ai_query.transport.aiohttp_transport.certifi_ssl_context") as context,
        patch("ai_query.transport.aiohttp_transport.aiohttp.TCPConnector") as connector,
        patch("ai_query.transport.aiohttp_transport.aiohttp.ClientSession") as client,
    ):
        context.return_value = "ssl-context"
        connector.return_value = "connector"
        client.return_value = session

        transport = AioHTTPTransport()
        try:
            assert await transport._get_session() is session
        finally:
            transport._session = None

    connector.assert_called_once_with(ssl="ssl-context")
    client.assert_called_once_with(connector="connector")


@pytest.mark.asyncio
async def test_agent_http_transport_uses_certifi_ca_bundle():
    from ai_query.agents.transport.http import HTTPTransport

    client = AsyncMock()

    with (
        patch("ai_query.agents.transport.http.certifi_ca_bundle") as ca_bundle,
        patch("ai_query.agents.transport.http.httpx.AsyncClient") as async_client,
    ):
        ca_bundle.return_value = "/tmp/cacert.pem"
        async_client.return_value = client

        transport = HTTPTransport(headers={"Authorization": "Bearer token"})
        assert await transport._get_client() is client

    async_client.assert_called_once_with(
        headers={"Authorization": "Bearer token"},
        verify="/tmp/cacert.pem",
    )
