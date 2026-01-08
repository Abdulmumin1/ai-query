"""MCP transport context managers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from ai_query.mcp.types import MCPServer
from ai_query.mcp.client import MCPClient


@asynccontextmanager
async def mcp(
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
) -> AsyncIterator[MCPServer]:
    """Connect to a local MCP server using stdio transport.

    This is the recommended way to use local MCP servers with ai-query.
    The connection is automatically cleaned up when the context exits.

    Args:
        command: The command to run (e.g., "python", "node", "npx").
        *args: Arguments for the command (e.g., "path/to/server.py").
        env: Environment variables for the subprocess.

    Yields:
        MCPServer with tools loaded and ready to use.

    Example:
        >>> async with mcp("python", "weather_server.py") as server:
        ...     result = await generate_text(
        ...         model=openai("gpt-4o"),
        ...         prompt="What's the weather?",
        ...         tools=server.tools,
        ...     )
        >>>
        >>> # With npx for npm packages
        >>> async with mcp("npx", "-y", "@modelcontextprotocol/server-fetch") as server:
        ...     result = await generate_text(
        ...         model=openai("gpt-4o"),
        ...         prompt="Fetch example.com",
        ...         tools=server.tools,
        ...     )
    """
    client = MCPClient()
    try:
        server = await client.connect_stdio(command, list(args), env)
        yield server
    finally:
        await client.disconnect()


@asynccontextmanager
async def mcp_sse(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPServer]:
    """Connect to a remote MCP server using SSE (Server-Sent Events) transport.

    Use this for remote MCP servers that use the legacy SSE transport.

    Args:
        url: The SSE endpoint URL (e.g., "http://localhost:8000/sse").
        headers: Optional HTTP headers for authentication or other purposes.

    Yields:
        MCPServer with tools loaded and ready to use.

    Example:
        >>> async with mcp_sse("http://localhost:8000/sse") as server:
        ...     result = await generate_text(
        ...         model=openai("gpt-4o"),
        ...         prompt="Hello!",
        ...         tools=server.tools,
        ...     )
        >>>
        >>> # With authentication
        >>> async with mcp_sse(
        ...     "https://api.example.com/mcp/sse",
        ...     headers={"Authorization": "Bearer token123"}
        ... ) as server:
        ...     result = await generate_text(...)
    """
    client = MCPClient()
    try:
        server = await client.connect_sse(url, headers)
        yield server
    finally:
        await client.disconnect()


@asynccontextmanager
async def mcp_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPServer]:
    """Connect to a remote MCP server using Streamable HTTP transport.

    This is the recommended transport for remote MCP servers (MCP spec 2025-11-25).

    Args:
        url: The HTTP endpoint URL (e.g., "http://localhost:8000/mcp").
        headers: Optional HTTP headers for authentication or other purposes.

    Yields:
        MCPServer with tools loaded and ready to use.

    Example:
        >>> async with mcp_http("http://localhost:8000/mcp") as server:
        ...     result = await generate_text(
        ...         model=openai("gpt-4o"),
        ...         prompt="Hello!",
        ...         tools=server.tools,
        ...     )
        >>>
        >>> # With authentication
        >>> async with mcp_http(
        ...     "https://api.example.com/mcp",
        ...     headers={"Authorization": "Bearer token123"}
        ... ) as server:
        ...     result = await generate_text(...)
    """
    client = MCPClient()
    try:
        server = await client.connect_http(url, headers)
        yield server
    finally:
        await client.disconnect()


async def connect_mcp(
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
) -> MCPServer:
    """Connect to a local MCP server without using a context manager.

    Use this when you need to manage the connection lifecycle manually.
    Remember to call server.close() when done.

    Args:
        command: The command to run (e.g., "python", "node", "npx").
        *args: Arguments for the command.
        env: Environment variables for the subprocess.

    Returns:
        MCPServer with tools loaded. Call server.close() when done.

    Example:
        >>> server = await connect_mcp("python", "weather_server.py")
        >>> try:
        ...     result = await generate_text(
        ...         model=openai("gpt-4o"),
        ...         prompt="What's the weather?",
        ...         tools=server.tools,
        ...     )
        ... finally:
        ...     await server.close()
    """
    client = MCPClient()
    return await client.connect_stdio(command, list(args), env)


async def connect_mcp_sse(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> MCPServer:
    """Connect to a remote SSE MCP server without using a context manager.

    Args:
        url: The SSE endpoint URL.
        headers: Optional HTTP headers.

    Returns:
        MCPServer with tools loaded. Call server.close() when done.
    """
    client = MCPClient()
    return await client.connect_sse(url, headers)


async def connect_mcp_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> MCPServer:
    """Connect to a remote Streamable HTTP MCP server without using a context manager.

    Args:
        url: The HTTP endpoint URL.
        headers: Optional HTTP headers.

    Returns:
        MCPServer with tools loaded. Call server.close() when done.
    """
    client = MCPClient()
    return await client.connect_http(url, headers)
