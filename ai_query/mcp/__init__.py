"""MCP (Model Context Protocol) support for ai-query.

This module provides integration with MCP servers, allowing tools exposed
by MCP servers to be used seamlessly with generate_text and stream_text.

Supports all MCP transports:
- stdio: Local process-based servers (python, node, npx)
- sse: Server-Sent Events for remote servers (legacy)
- streamable_http: Streamable HTTP transport for remote servers (recommended)

Example:
    >>> from ai_query import generate_text, openai
    >>> from ai_query.mcp import mcp, mcp_sse, mcp_http
    >>>
    >>> # Connect to a local stdio server
    >>> async with mcp("python", "path/to/server.py") as server:
    ...     result = await generate_text(
    ...         model=openai("gpt-4o"),
    ...         prompt="What's the weather in Paris?",
    ...         tools=server.tools,
    ...     )
    >>>
    >>> # Connect to a remote SSE server
    >>> async with mcp_sse("http://localhost:8000/sse") as server:
    ...     result = await generate_text(
    ...         model=openai("gpt-4o"),
    ...         prompt="Hello!",
    ...         tools=server.tools,
    ...     )
    >>>
    >>> # Connect to a remote Streamable HTTP server
    >>> async with mcp_http("http://localhost:8000/mcp") as server:
    ...     result = await generate_text(
    ...         model=openai("gpt-4o"),
    ...         prompt="Hello!",
    ...         tools=server.tools,
    ...     )
"""

from ai_query.mcp.types import MCPServer, MCPTool, TransportType
from ai_query.mcp.client import MCPClient
from ai_query.mcp.transports import (
    mcp,
    mcp_sse,
    mcp_http,
    connect_mcp,
    connect_mcp_sse,
    connect_mcp_http,
)
from ai_query.mcp.utils import merge_tools

__all__ = [
    # Types
    "MCPServer",
    "MCPTool",
    "TransportType",
    # Client
    "MCPClient",
    # Context managers (recommended)
    "mcp",
    "mcp_sse",
    "mcp_http",
    # Non-context manager functions
    "connect_mcp",
    "connect_mcp_sse",
    "connect_mcp_http",
    # Utilities
    "merge_tools",
]
