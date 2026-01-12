"""Message types for agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_query.agents.websocket import Connection


@dataclass
class IncomingMessage:
    """Unified message from any source.
    
    Represents a message received by an agent, regardless of whether it came
    from a WebSocket client, another agent via invoke(), or an HTTP request.
    
    Attributes:
        content: The message content (string, bytes, or dict).
        source: Where the message came from.
        source_id: Identifier for the source (connection ID, agent ID, or request ID).
        metadata: Additional context about the message.
    
    Example:
        async def on_receive(self, message: IncomingMessage) -> Any:
            if message.source == "client":
                # Handle WebSocket client message
                response = await self.chat(message.content)
                await message.reply(response)
            elif message.source == "agent":
                # Handle invoke from another agent
                return await self.process_task(message.content)
    """
    
    content: str | bytes | dict
    source: Literal["client", "agent", "http"]
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Internal fields for reply functionality
    _reply_fn: Callable[[Any], Any] | None = field(default=None, repr=False)
    _connection: "Connection | None" = field(default=None, repr=False)
    
    async def reply(self, response: Any) -> None:
        """Send a response back to the message source.
        
        Args:
            response: The response to send.
        
        Raises:
            RuntimeError: If no reply function is available.
        """
        if self._reply_fn is not None:
            result = self._reply_fn(response)
            if hasattr(result, "__await__"):
                await result
        elif self._connection is not None:
            if isinstance(response, (dict, list)):
                import json
                response = json.dumps(response)
            await self._connection.send(response)
        else:
            raise RuntimeError("Cannot reply: no reply function or connection available")
