"""Built-in aiohttp WebSocket server for agents."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiohttp import web, WSMsgType

from ai_query.agents.websocket import Connection, ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.base import Agent


class AioHttpConnection(Connection):
    """Wraps aiohttp WebSocket in our Connection interface."""
    
    def __init__(self, ws: web.WebSocketResponse, request: web.Request):
        self._ws = ws
        self._request = request
        self.username: str | None = None  # For user convenience
    
    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws.send_bytes(message)
        else:
            await self._ws.send_str(message)
    
    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, message=reason.encode())


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections for an agent."""
    agent: Agent = request.app["agent"]
    
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Create connection and context
    connection = AioHttpConnection(ws, request)
    ctx = ConnectionContext(
        request=request,
        metadata=dict(request.query),  # Query params as metadata
    )
    
    # Connect
    await agent.on_connect(connection, ctx)
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await agent.on_message(connection, msg.data)
            elif msg.type == WSMsgType.BINARY:
                await agent.on_message(connection, msg.data)
            elif msg.type == WSMsgType.ERROR:
                await agent.on_error(connection, ws.exception())
    except Exception as e:
        await agent.on_error(connection, e)
    finally:
        await agent.on_close(connection, 1000, "Client disconnected")
    
    return ws


def run_agent_server(
    agent: "Agent",
    host: str = "localhost",
    port: int = 8080,
    path: str = "/ws",
) -> None:
    """Run the agent as a WebSocket server (blocking).
    
    Args:
        agent: The agent to serve.
        host: Host to bind to (default: localhost).
        port: Port to bind to (default: 8080).
        path: WebSocket endpoint path (default: /ws).
    """
    asyncio.run(run_agent_server_async(agent, host, port, path))


async def sse_handler(request: web.Request) -> web.StreamResponse:
    """Handle SSE connections for AI streaming."""
    agent: "Agent" = request.app["agent"]
    
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )
    await response.prepare(request)
    
    # Register this SSE connection
    agent._sse_connections.add(response)
    
    try:
        # Keep alive until client disconnects
        while True:
            await asyncio.sleep(30)
            await response.write(b": keepalive\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        agent._sse_connections.discard(response)
    
    return response


async def run_agent_server_async(
    agent: "Agent",
    host: str = "localhost",
    port: int = 8080,
    path: str = "/ws",
) -> None:
    """Run the agent as a WebSocket server (async).
    
    Args:
        agent: The agent to serve.
        host: Host to bind to (default: localhost).
        port: Port to bind to (default: 8080).
        path: WebSocket endpoint path (default: /ws).
    """
    # Start the agent
    await agent.start()
    
    # Create aiohttp app
    app = web.Application()
    app["agent"] = agent
    app.router.add_get(path, websocket_handler)      # WebSocket for chat
    app.router.add_get("/events", sse_handler)       # SSE for AI streaming
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    print(f"Agent server running at ws://{host}:{port}{path}")
    print(f"SSE endpoint at http://{host}:{port}/events")
    await site.start()
    
    # Run forever
    await asyncio.Event().wait()
