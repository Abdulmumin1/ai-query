"""Agent output abstraction for protocol-agnostic feedback."""

from __future__ import annotations

import asyncio
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

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        """Send a standard message (final or partial)."""
        ...

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        """Send a status update (e.g., 'Thinking...', 'Running tool...')."""
        ...

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        """Send an error notification."""
        ...

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        """Send a custom event."""
        ...


class NullOutput(AgentOutput):
    """No-op output for when no transport is available."""

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        pass

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        pass

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        pass

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        pass


class WebSocketOutput(AgentOutput):
    """Output adapter for WebSocket connections.

    Sends messages as JSON objects:
    - {"type": "message", "content": "...", "id": 123}
    - {"type": "status", "status": "...", "details": {...}, "id": 123}
    - {"type": "error", "error": "...", "id": 123}
    """

    def __init__(self, conn: Connection):
        self.conn = conn

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "message", "content": content}
        if event_id is not None:
            payload["id"] = event_id
        await self.conn.send(json.dumps(payload))

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "status", "status": status}
        if details:
            payload["details"] = details
        if event_id is not None:
            payload["id"] = event_id
        await self.conn.send(json.dumps(payload))

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "error", "error": error}
        if event_id is not None:
            payload["id"] = event_id
        await self.conn.send(json.dumps(payload))

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": event, **data}
        if event_id is not None:
            payload["id"] = event_id
        await self.conn.send(json.dumps(payload))


class SSEOutput(AgentOutput):
    """Output adapter for Server-Sent Events (SSE).

    Sends messages as SSE events with optional id field:
    - id: 123
      event: message
      data: {"content": "..."}
    """

    def __init__(self, writer: Any):
        """Initialize with an object that has a write() method (e.g., aiohttp StreamResponse)."""
        self.writer = writer

    def _format_sse(self, event: str, data: str, event_id: int | None = None) -> bytes:
        """Format an SSE message with optional id field."""
        lines = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event}")
        lines.append(f"data: {data}")
        lines.append("")
        lines.append("")
        return "\n".join(lines).encode("utf-8")

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        # Escape newlines for SSE data
        safe_content = content.replace("\n", "\\n")
        await self.writer.write(self._format_sse("message", safe_content, event_id))

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        data = {"status": status}
        if details:
            data["details"] = details
        await self.writer.write(self._format_sse("status", json.dumps(data), event_id))

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        await self.writer.write(self._format_sse("error", error, event_id))

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        await self.writer.write(self._format_sse(event, json.dumps(data), event_id))


class BroadcastOutput(AgentOutput):
    """Output adapter that broadcasts to all active agent connections.

    Sends to all WebSocket connections and SSE streams managed by the agent.
    This is the default output for ChatAgent.

    Automatically generates sequential event IDs for SSE reconnection support,
    even when event persistence is not enabled.
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent
        # Generate sequential IDs for SSE reconnection support
        self._seq = int(time.time() * 1000)

    def _next_id(self) -> int:
        """Generate the next sequential event ID."""
        self._seq += 1
        return self._seq

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        # Generate ID if not provided
        if event_id is None:
            event_id = self._next_id()

        # Broadcast to WebSockets
        payload: dict[str, Any] = {"type": "message", "content": content, "id": event_id}
        msg_payload = json.dumps(payload)
        for conn in list(self.agent._connections):
            try:
                await conn.send(msg_payload)
            except Exception:
                pass

        # Broadcast to SSE
        if hasattr(self.agent, "stream_to_sse_with_id"):
            await self.agent.stream_to_sse_with_id("message", content, event_id)
        elif hasattr(self.agent, "stream_to_sse"):
            await self.agent.stream_to_sse("message", content)

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        # Generate ID if not provided
        if event_id is None:
            event_id = self._next_id()

        # WebSockets
        payload: dict[str, Any] = {"type": "status", "status": status, "id": event_id}
        if details:
            payload["details"] = details
        json_payload = json.dumps(payload)

        for conn in list(self.agent._connections):
            try:
                await conn.send(json_payload)
            except Exception:
                pass

        # SSE
        if hasattr(self.agent, "stream_to_sse_with_id"):
            await self.agent.stream_to_sse_with_id("status", json.dumps(payload), event_id)
        elif hasattr(self.agent, "stream_to_sse"):
            await self.agent.stream_to_sse("status", json.dumps(payload))

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        # Generate ID if not provided
        if event_id is None:
            event_id = self._next_id()

        # WebSockets
        payload: dict[str, Any] = {"type": "error", "error": error, "id": event_id}
        json_payload = json.dumps(payload)
        for conn in list(self.agent._connections):
            try:
                await conn.send(json_payload)
            except Exception:
                pass

        # SSE
        if hasattr(self.agent, "stream_to_sse_with_id"):
            await self.agent.stream_to_sse_with_id("error", error, event_id)
        elif hasattr(self.agent, "stream_to_sse"):
            await self.agent.stream_to_sse("error", error)

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        # Generate ID if not provided
        if event_id is None:
            event_id = self._next_id()

        # WebSockets
        payload: dict[str, Any] = {"type": event, **data, "id": event_id}
        json_payload = json.dumps(payload)
        for conn in list(self.agent._connections):
            try:
                await conn.send(json_payload)
            except Exception:
                pass

        # SSE
        if hasattr(self.agent, "stream_to_sse_with_id"):
            await self.agent.stream_to_sse_with_id(event, json.dumps(data), event_id)
        elif hasattr(self.agent, "stream_to_sse"):
            await self.agent.stream_to_sse(event, json.dumps(data))


class QueueOutput(AgentOutput):
    """Output adapter that puts events into an asyncio.Queue.

    Useful for creating custom streaming generators (e.g. for HTTP streaming).
    Events are put as dicts:
    - {"type": "message", "content": "...", "id": 123}
    - {"type": "status", "status": "...", "details": {...}, "id": 123}
    - {"type": "error", "error": "...", "id": 123}
    - {"type": "custom_event", ..., "id": 123}
    """

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "message", "content": content}
        if event_id is not None:
            payload["id"] = event_id
        await self.queue.put(payload)

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "status", "status": status}
        if details:
            payload["details"] = details
        if event_id is not None:
            payload["id"] = event_id
        await self.queue.put(payload)

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": "error", "error": error}
        if event_id is not None:
            payload["id"] = event_id
        await self.queue.put(payload)

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        payload: dict[str, Any] = {"type": event, **data}
        if event_id is not None:
            payload["id"] = event_id
        await self.queue.put(payload)


class PersistingOutput(AgentOutput):
    """Wrapper that persists events to the agent's log before sending.

    Automatically used by ChatAgent when enable_event_log is True.
    Generates event IDs and includes them in both the persisted log and sent events.
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

    async def _save(self, type: str, data: dict[str, Any]) -> int:
        """Save event and return the generated ID."""
        from ai_query.types import AgentEvent

        event_id = self._next_id()
        event = AgentEvent(
            id=event_id,
            type=type,
            data=data,
            created_at=time.time()
        )
        await self.agent._save_event(event)
        return event_id

    async def send_message(self, content: str, *, event_id: int | None = None) -> None:
        generated_id = await self._save("message", {"content": content})
        await self.wrapped.send_message(content, event_id=generated_id)

    async def send_status(self, status: str, details: dict[str, Any] | None = None, *, event_id: int | None = None) -> None:
        data = {"status": status}
        if details:
            data["details"] = details
        generated_id = await self._save("status", data)
        await self.wrapped.send_status(status, details, event_id=generated_id)

    async def send_error(self, error: str, *, event_id: int | None = None) -> None:
        generated_id = await self._save("error", {"error": error})
        await self.wrapped.send_error(error, event_id=generated_id)

    async def send_event(self, event: str, data: dict[str, Any], *, event_id: int | None = None) -> None:
        generated_id = await self._save(event, data)
        await self.wrapped.send_event(event, data, event_id=generated_id)
