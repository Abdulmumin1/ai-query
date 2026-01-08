"""MCP type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_query.types import ToolSet


# Transport types
TransportType = Literal["stdio", "sse", "streamable_http"]


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    _call_fn: Callable[..., Awaitable[Any]]

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool via the MCP server."""
        return await self._call_fn(self.name, kwargs)


@dataclass
class MCPServer:
    """Represents a connected MCP server with its tools.

    This class manages the connection to an MCP server and provides
    access to its tools in a format compatible with ai-query.

    Attributes:
        tools: Dictionary of Tool objects that can be passed to generate_text/stream_text.
        raw_tools: List of raw MCP tool definitions from the server.
        transport: The transport type used for this connection.
    """

    tools: "ToolSet" = field(default_factory=dict)
    raw_tools: list[dict[str, Any]] = field(default_factory=list)
    transport: TransportType = "stdio"

    # Internal state
    _session: Any = None
    _exit_stack: Any = None
    _client: Any = None  # Reference to MCPClient for close()

    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if self._client:
            await self._client.disconnect()
