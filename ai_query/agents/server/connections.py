"""Connection adapters for AgentServer."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from ai_query.agents.websocket import Connection


class AioHttpConnection(Connection):
    """Wraps aiohttp WebSocket in our Connection interface."""
    state: dict[str, Any]

    def __init__(self, ws: web.WebSocketResponse, request: web.Request):
        self._ws = ws
        self._request = request
        self.username: str | None = None
        self.agent_id: str | None = None
        self.state = {}

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws.send_bytes(message)
        else:
            await self._ws.send_str(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, message=reason.encode())

    async def send_event(self, event: str, data: dict[str, Any]) -> None:
        """Send a structured event as JSON."""
        payload = {"type": event, **data}
        await self.send(json.dumps(payload))


class AioHttpSSEConnection(Connection):
    """Wraps aiohttp StreamResponse for SSE in our Connection interface."""
    state: dict[str, Any]

    def __init__(self, response: web.StreamResponse, request: web.Request):
        self._response = response
        self._request = request
        self.state = {}

    async def send(self, message: str | bytes) -> None:
        """Send message as SSE data event."""
        # This handles raw send calls (e.g. broadcast)
        # We assume it's a JSON string, wrap it in 'message' event or generic data
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        # If message looks like JSON with 'type', try to use that?
        # Simpler: just send as data
        await self._response.write(f"data: {message}\n\n".encode("utf-8"))

    async def send_event(self, event: str, data: dict[str, Any]) -> None:
        """Send structured event as SSE."""
        # Escape newlines for SSE data
        json_data = json.dumps(data)
        await self._response.write(f"event: {event}\ndata: {json_data}\n\n".encode("utf-8"))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        # SSE connections are closed by client disconnecting or server finishing
        # We can't explicitly close from here except by finishing response
        await self._response.write_eof()
