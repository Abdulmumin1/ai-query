"""Tests for transport implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


class TestAioHTTPTransport:
    def test_close_at_exit_runs_async_close(self):
        """AioHTTPTransport should perform best-effort session cleanup at exit."""
        from ai_query.transport.aiohttp_transport import AioHTTPTransport

        transport = AioHTTPTransport()
        session = AsyncMock()
        session.closed = False
        transport._session = session

        def consume(coro):
            coro.close()

        with patch("asyncio.run", side_effect=consume) as mock_run:
            transport._close_at_exit()

        mock_run.assert_called_once()

    def test_close_at_exit_ignores_closed_session(self):
        """AioHTTPTransport should no-op when session is already closed."""
        from ai_query.transport.aiohttp_transport import AioHTTPTransport

        transport = AioHTTPTransport()
        session = AsyncMock()
        session.closed = True
        transport._session = session

        with patch("asyncio.run") as mock_run:
            transport._close_at_exit()

        mock_run.assert_not_called()
