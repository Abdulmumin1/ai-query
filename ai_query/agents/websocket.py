"""WebSocket types for agent connections."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Connection(Protocol):
    """WebSocket connection interface.
    
    This protocol defines the interface that any WebSocket connection
    must implement to work with agents.
    """
    
    async def send(self, message: str | bytes) -> None:
        """Send a message to the client."""
        ...
    
    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""
        ...


@dataclass
class ConnectionContext:
    """Context for a WebSocket connection.
    
    Contains metadata about the connection, including the original
    HTTP request that initiated the WebSocket upgrade.
    """
    request: Any = None  # Original HTTP request
    metadata: dict[str, Any] = field(default_factory=dict)
