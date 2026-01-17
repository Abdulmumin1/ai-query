"""Built-in aiohttp server for agents with WebSocket, SSE, and REST endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from aiohttp import web, WSMsgType

from ai_query.agents.websocket import Connection, ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.v2 import Agent


class AioHttpConnection(Connection):
    """Wraps aiohttp WebSocket in our Connection interface."""

    def __init__(self, ws: web.WebSocketResponse, request: web.Request):
        self._ws = ws
        self._request = request

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws.send_bytes(message)
        else:
            await self._ws.send_str(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, message=reason.encode())

    async def send_event(self, event: str, data: dict) -> None:
        """Send a structured event as JSON."""
        await self.send(json.dumps({"type": event, **data}))


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections for an agent."""
    agent: Agent = request.app["agent"]

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connection = AioHttpConnection(ws, request)
    ctx = ConnectionContext(
        request=request,
        metadata=dict(request.query),
    )

    # Handle replay on reconnect
    last_event_id = request.query.get("last_event_id")
    if last_event_id:
        try:
            after_id = int(last_event_id)
            async for event in agent.replay_events(after_id):
                await connection.send(json.dumps({
                    "type": event.type,
                    "id": event.id,
                    **event.data,
                }))
        except ValueError:
            pass

    # Register connection
    await agent.on_connect(connection, ctx)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await agent.on_message(connection, msg.data)
            elif msg.type == WSMsgType.BINARY:
                await agent.on_message(connection, msg.data)
            elif msg.type == WSMsgType.ERROR:
                pass
    except Exception:
        pass
    finally:
        await agent.on_close(connection, 1000, "Client disconnected")

    return ws


async def sse_handler(request: web.Request) -> web.StreamResponse:
    """Handle SSE connections for event streaming."""
    agent: Agent = request.app["agent"]

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )
    await response.prepare(request)

    # Handle replay on reconnect
    last_event_id = request.query.get("last_event_id") or request.headers.get("Last-Event-ID")
    if last_event_id:
        try:
            after_id = int(last_event_id)
            async for event in agent.replay_events(after_id):
                sse_msg = f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data)}\n\n"
                await response.write(sse_msg.encode())
        except ValueError:
            pass

    # Register SSE connection
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


def add_cors_headers(response: web.Response) -> web.Response:
    """Add CORS headers to a response."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


async def cors_preflight_handler(request: web.Request) -> web.Response:
    """Handle CORS preflight OPTIONS requests."""
    return add_cors_headers(web.Response())


async def chat_handler(request: web.Request) -> web.Response:
    """Handle REST chat requests.

    Streams the response internally while emitting chunks to SSE/WebSocket
    subscribers, then returns the complete response as JSON.
    """
    agent: Agent = request.app["agent"]

    try:
        body = await request.json()
        message = body.get("message", "")

        if not message:
            return add_cors_headers(web.json_response(
                {"error": "message is required"},
                status=400
            ))

        # Emit start event
        await agent.emit("chat_start", {"message": message})

        # Stream response, emitting chunks to SSE/WebSocket subscribers
        full_response = ""
        async for chunk in agent.stream(message):
            full_response += chunk
            await agent.emit("chunk", {"content": chunk})

        # Emit completion event
        await agent.emit("chat_complete", {"response": full_response})

        return add_cors_headers(web.json_response({
            "agent_id": agent.id,
            "response": full_response,
        }))

    except json.JSONDecodeError:
        return add_cors_headers(web.json_response(
            {"error": "Invalid JSON"},
            status=400
        ))
    except Exception as e:
        await agent.emit("chat_error", {"error": str(e)})
        return add_cors_headers(web.json_response(
            {"error": str(e)},
            status=500
        ))


async def state_handler(request: web.Request) -> web.Response:
    """Get agent state."""
    agent: Agent = request.app["agent"]

    return add_cors_headers(web.json_response({
        "agent_id": agent.id,
        "state": agent.state,
    }))


def _create_emit_handler(agent: "Agent"):
    """Create the emit handler that delivers events to all connected clients."""

    async def deliver(event: str, data: dict, event_id: int) -> None:
        # Deliver to WebSocket connections
        ws_msg = json.dumps({"type": event, "id": event_id, **data})
        for conn in list(agent._connections):
            try:
                await conn.send(ws_msg)
            except Exception:
                agent._connections.discard(conn)

        # Deliver to SSE connections
        sse_msg = f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
        for sse in list(agent._sse_connections):
            try:
                await sse.write(sse_msg.encode())
            except Exception:
                agent._sse_connections.discard(sse)

    return deliver


def run_agent_server(
    agent: "Agent",
    host: str = "localhost",
    port: int = 8080,
) -> None:
    """Run the agent as a server (blocking).

    Creates endpoints:
        - WS   /ws           - WebSocket connection
        - GET  /events       - SSE event stream
        - POST /chat         - REST chat endpoint
        - GET  /state        - Get agent state

    Args:
        agent: The agent to serve.
        host: Host to bind to (default: localhost).
        port: Port to bind to (default: 8080).
    """
    asyncio.run(run_agent_server_async(agent, host, port))


async def run_agent_server_async(
    agent: "Agent",
    host: str = "localhost",
    port: int = 8080,
) -> None:
    """Run the agent as a server (async).

    Args:
        agent: The agent to serve.
        host: Host to bind to (default: localhost).
        port: Port to bind to (default: 8080).
    """
    # Start the agent
    await agent.start()

    # Inject emit handler for event delivery
    agent._emit_handler = _create_emit_handler(agent)

    # Create aiohttp app
    app = web.Application()
    app["agent"] = agent

    # Routes
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_post("/chat", chat_handler)
    app.router.add_options("/chat", cors_preflight_handler)
    app.router.add_get("/state", state_handler)
    app.router.add_options("/state", cors_preflight_handler)

    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)

    print(f"Agent '{agent.id}' serving at http://{host}:{port}")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    print(f"  SSE:       http://{host}:{port}/events")
    print(f"  Chat:      POST http://{host}:{port}/chat")
    print(f"  State:     GET http://{host}:{port}/state")

    await site.start()

    # Run forever
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()
        await runner.cleanup()
