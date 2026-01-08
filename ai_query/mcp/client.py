"""MCP Client for connecting to MCP servers."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from ai_query.types import Tool, ToolSet
from ai_query.mcp.types import MCPServer, TransportType


class MCPClient:
    """Client for connecting to MCP servers.

    This client handles the connection lifecycle and tool discovery
    for MCP servers using stdio, SSE, or Streamable HTTP transports.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._exit_stack: Any = None

    def _check_mcp_installed(self) -> None:
        """Check if the mcp package is installed."""
        try:
            import mcp
        except ImportError as e:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install mcp"
            ) from e

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPServer:
        """Connect to an MCP server using stdio transport.

        Args:
            command: The command to run (e.g., "python", "node", "npx").
            args: Arguments for the command (e.g., ["path/to/server.py"]).
            env: Environment variables for the subprocess.

        Returns:
            MCPServer with tools loaded.
        """
        self._check_mcp_installed()

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._exit_stack = AsyncExitStack()

        # Configure server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        # Set up transport and session
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        return await self._initialize_server("stdio")

    async def connect_sse(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> MCPServer:
        """Connect to an MCP server using SSE (Server-Sent Events) transport.

        Args:
            url: The SSE endpoint URL (e.g., "http://localhost:8000/sse").
            headers: Optional HTTP headers for the connection.

        Returns:
            MCPServer with tools loaded.
        """
        self._check_mcp_installed()

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._exit_stack = AsyncExitStack()

        # Set up SSE transport and session
        sse_transport = await self._exit_stack.enter_async_context(
            sse_client(url, headers=headers)
        )
        read_stream, write_stream = sse_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        return await self._initialize_server("sse")

    async def connect_http(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> MCPServer:
        """Connect to an MCP server using Streamable HTTP transport.

        This is the recommended transport for remote MCP servers (MCP spec 2025-11-25).

        Args:
            url: The HTTP endpoint URL (e.g., "http://localhost:8000/mcp").
            headers: Optional HTTP headers for the connection.

        Returns:
            MCPServer with tools loaded.
        """
        self._check_mcp_installed()

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx

        self._exit_stack = AsyncExitStack()

        # Create httpx client with headers if provided
        http_client = httpx.AsyncClient(headers=headers) if headers else None

        # Set up Streamable HTTP transport and session
        # streamable_http_client returns (read_stream, write_stream, get_session_id)
        http_transport = await self._exit_stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        read_stream, write_stream, _get_session_id = http_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        return await self._initialize_server("streamable_http")

    async def _initialize_server(self, transport: TransportType) -> MCPServer:
        """Initialize the session and load tools."""
        # Initialize the session
        await self._session.initialize()

        # List available tools
        response = await self._session.list_tools()
        raw_tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Convert MCP tools to ai-query Tool format
        tools: ToolSet = {}
        for mcp_tool in response.tools:
            tools[mcp_tool.name] = self._convert_tool(mcp_tool)

        server = MCPServer(
            tools=tools,
            raw_tools=raw_tools,
            transport=transport,
            _session=self._session,
            _exit_stack=self._exit_stack,
            _client=self,
        )

        return server

    def _convert_tool(self, mcp_tool: Any) -> Tool:
        """Convert an MCP tool to an ai-query Tool."""

        async def execute_fn(**kwargs: Any) -> Any:
            """Execute the tool via MCP."""
            result = await self._session.call_tool(mcp_tool.name, kwargs)
            # MCP returns content as a list of content blocks
            if hasattr(result, "content") and result.content:
                # Concatenate all text content
                texts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif hasattr(block, "data"):
                        # Binary data - return as-is or encode
                        texts.append(f"[Binary data: {len(block.data)} bytes]")
                return "\n".join(texts) if texts else str(result)
            return str(result)

        return Tool(
            description=mcp_tool.description or f"Execute {mcp_tool.name}",
            parameters=mcp_tool.inputSchema or {"type": "object", "properties": {}},
            execute=execute_fn,
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
