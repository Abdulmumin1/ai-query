"""Agent output abstraction for protocol-agnostic feedback."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from ai_query.agents.websocket import Connection

if TYPE_CHECKING:
    from ai_query.agents.base import Agent
    from ai_query.types import AgentEvent


@runtime_checkable
class AgentOutput(Protocol):
    """Abstract interface for sending feedback to the user.

    Implement this protocol to handle agent output for different transports
    (WebSocket, SSE, HTTP, etc.) without changing agent logic.
    """

    async def send_message(self, content: str) -> None:
        """Send a standard message (final or partial)."""
        ...

    async def send_status(self, status: str, details: dict[str, Any] | None = None) -> None:
        """Send a status update (e.g., 'Thinking...', 'Running tool...')."""
        ...

    async def send_error(self, error: str) -> None:
        """Send an error notification."""
        ...


class NullOutput(AgentOutput):
    """No-op output for when no transport is available."""

    async def send_message(self, content: str) -> None:
        pass

    async def send_status(self, status: str, details: dict[str, Any] | None = None) -> None:
        pass

    async def send_error(self, error: str) -> None:
        pass


class WebSocketOutput(AgentOutput):
    """Output adapter for WebSocket connections.

    Sends messages as JSON objects:
    - {"type": "message", "content": "..."}
    - {"type": "status", "status": "...", "details": {...}}
    - {"type": "error", "error": "..."}
    """

    def __init__(self, conn: Connection):
        self.conn = conn

    async def send_message(self, content: str) -> None:
        await self.conn.send(json.dumps({"type": "message", "content": content}))

    async def send_status(self, status: str, details: dict[str, Any] | None = None) -> None:
        payload = {"type": "status", "status": status}
        if details:
            payload["details"] = details
        await self.conn.send(json.dumps(payload))

    async def send_error(self, error: str) -> None:
        await self.conn.send(json.dumps({"type": "error", "error": error}))


class SSEOutput(AgentOutput):
    """Output adapter for Server-Sent Events (SSE).

    Sends messages as SSE events:
    - event: message
    - event: status
    - event: error
    """

    def __init__(self, writer: Any):
        """Initialize with an object that has a write() method (e.g., aiohttp StreamResponse)."""
        self.writer = writer

    async def send_message(self, content: str) -> None:
        # Escape newlines for SSE data
        safe_content = content.replace("\n", "\\n")
        await self.writer.write(f"event: message\ndata: {safe_content}\n\n".encode("utf-8"))

    async def send_status(self, status: str, details: dict[str, Any] | None = None) -> None:
        data = {"status": status}
        if details:
            data["details"] = details
        await self.writer.write(f"event: status\ndata: {json.dumps(data)}\n\n".encode("utf-8"))

    async def send_error(self, error: str) -> None:
        await self.writer.write(f"event: error\ndata: {error}\n\n".encode("utf-8"))


class PersistingOutput(AgentOutput):
    """Wrapper that persists events to the agent's log before sending.

    Automatically used by ChatAgent when enable_event_log is True.
    """

    def __init__(self, wrapped: AgentOutput, agent: "Agent"):
        self.wrapped = wrapped
        self.agent = agent
        # Simple monotonic counter for this session - ideal would be database sequence
        # For MVP, we'll use timestamp * 1000 to be roughly unique and ordered
        self._seq = int(time.time() * 1000)

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    async def _save(self, type: str, data: dict[str, Any]) -> None:
        # Import here to avoid circular imports if any
        from ai_query.types import AgentEvent

        event = AgentEvent(
            id=self._next_id(),
            type=type,
            data=data,
            created_at=time.time()
        )
        await self.agent._save_event(event)

    async def send_message(self, content: str) -> None:
        await self._save("message", {"content": content})
        await self.wrapped.send_message(content)

    async def send_status(self, status: str, details: dict[str, Any] | None = None) -> None:
        data = {"status": status}
        if details:
            data["details"] = details
        await self._save("status", data)
        await self.wrapped.send_status(status, details)

    async def send_error(self, error: str) -> None:
        await self._save("error", {"error": error})
        await self.wrapped.send_error(error)
