"""Request handlers for AgentServer."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web, WSMsgType

from ai_query.agents.server.connections import AioHttpConnection, AioHttpSSEConnection
from ai_query.agents.websocket import ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.server.base import AgentServer


def setup_routes(server: "AgentServer") -> web.Application:
    """Configure routes for the AgentServer application."""
    config = server._config
    base = config.base_path.rstrip("/")

    # Create aiohttp app with CORS middleware
    app = web.Application(middlewares=[create_cors_middleware(server)])

    # Handlers wrapper
    handlers = ServerHandlers(server)

    # WebSocket and SSE routes
    app.router.add_get(f"{base}/{{agent_id}}/ws", handlers.handle_websocket)
    app.router.add_get(f"{base}/{{agent_id}}/events", handlers.handle_sse)

    # REST API routes
    if config.enable_rest_api:
        app.router.add_get(f"{base}/{{agent_id}}/state", handlers.handle_get_state)
        app.router.add_get(f"{base}/{{agent_id}}/messages", handlers.handle_get_messages)
        app.router.add_put(f"{base}/{{agent_id}}/state", handlers.handle_put_state)
        app.router.add_post(f"{base}/{{agent_id}}/chat", handlers.handle_chat)
        app.router.add_post(f"{base}/{{agent_id}}/action/{{action_name}}", handlers.handle_action)
        app.router.add_post(f"{base}/{{agent_id}}/invoke", handlers.handle_invoke)
        app.router.add_post(f"{base}/{{agent_id}}", handlers.handle_request)
        app.router.add_delete(f"{base}/{{agent_id}}", handlers.handle_delete_agent)
        
        # Options handlers for CORS
        for route in [
            f"{base}/{{agent_id}}/state",
            f"{base}/{{agent_id}}/messages",
            f"{base}/{{agent_id}}/chat",
            f"{base}/{{agent_id}}/invoke",
            f"{base}/{{agent_id}}",
        ]:
            app.router.add_options(route, handlers.handle_options)

    # List agents endpoint
    if config.enable_list_agents:
        app.router.add_get("/agents", handlers.handle_list_agents)

    # Call setup hook
    server.on_app_setup(app)

    return app


def create_cors_middleware(server: "AgentServer") -> Any:
    """Create CORS middleware."""
    @web.middleware
    async def cors_middleware(request: web.Request, handler: Any) -> web.Response:
        if request.method == "OPTIONS":
            response = web.Response()
            return add_cors_headers(server, response, request)

        try:
            response = await handler(request)
            if not isinstance(response, web.WebSocketResponse):
                add_cors_headers(server, response, request)
            return response
        except web.HTTPException as ex:
            add_cors_headers(server, ex, request)
            raise

    return cors_middleware


def add_cors_headers(server: "AgentServer", response: Any, request: web.Request | None = None) -> Any:
    """Add CORS headers based on configuration."""
    config = server._config
    if config.allowed_origins:
        request_origin = request.headers.get("Origin") if request else None
        if request_origin and request_origin in config.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = request_origin
            response.headers["Vary"] = "Origin"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    return response


class ServerHandlers:
    """Container for request handlers to avoid closure overhead."""

    def __init__(self, server: "AgentServer"):
        self.server = server

    async def _check_auth(self, request: web.Request) -> None:
        if self.server._config.auth:
            allowed = await self.server._config.auth(request)
            if not allowed:
                raise web.HTTPUnauthorized(text="Authentication required")

    async def handle_options(self, request: web.Request) -> web.Response:
        return add_cors_headers(self.server, web.Response(), request)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        connection = AioHttpConnection(ws, request)
        connection.agent_id = agent_id
        ctx = ConnectionContext(request=request, metadata=dict(request.query))

        # Track connection
        meta = self.server._agents[agent_id]
        meta.connection_count += 1
        meta.last_activity = time.time()

        agent.enqueue("connect", None, connection=connection, ctx=ctx)

        # Replay events
        last_event_id_str = request.query.get("last_event_id")
        if last_event_id_str:
            try:
                last_id = int(last_event_id_str)
                async def replay():
                    async for event in agent.replay_events(last_id):
                        await connection.send_event(event.type, event.data)
                asyncio.create_task(replay())
            except ValueError:
                pass

        try:
            async for msg in ws:
                meta.last_activity = time.time()
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if isinstance(data, dict) and data.get("type") == "action":
                            # Handle action call
                            self._handle_ws_action(agent, connection, ctx, data)
                        else:
                            agent.enqueue("message", msg.data, connection=connection)
                    except json.JSONDecodeError:
                        agent.enqueue("message", msg.data, connection=connection)
                elif msg.type == WSMsgType.BINARY:
                    agent.enqueue("message", msg.data, connection=connection)
                elif msg.type == WSMsgType.ERROR:
                    agent.enqueue("error", ws.exception(), connection=connection)
        except Exception as e:
            agent.enqueue("error", e, connection=connection)
        finally:
            agent.enqueue("close", (1000, "Disconnected"), connection=connection)
            meta.connection_count -= 1
            meta.last_activity = time.time()

        return ws

    def _handle_ws_action(self, agent: Any, conn: Any, ctx: Any, data: dict) -> None:
        name = data.get("name")
        params = data.get("params", {})
        call_id = data.get("call_id")

        async def run_action():
            future = asyncio.get_running_loop().create_future()
            agent.enqueue("action", {"name": name, "params": params}, connection=conn, ctx=ctx, future=future)
            try:
                result = await future
                await conn.send(json.dumps({
                    "type": "action_result", "call_id": call_id, "result": result
                }))
            except Exception as e:
                await conn.send(json.dumps({
                    "type": "action_result", "call_id": call_id, "error": str(e)
                }))

        asyncio.create_task(run_action())

    async def handle_sse(self, request: web.Request) -> web.StreamResponse:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        
        # Add CORS headers manually since StreamResponse is prepared early
        response = web.StreamResponse(headers=headers)
        add_cors_headers(self.server, response, request)
        await response.prepare(request)

        agent._sse_connections.add(response)
        meta = self.server._agents[agent_id]

        # Replay
        last_id = request.query.get("last_event_id")
        if last_id:
            try:
                lid = int(last_id)
                conn = AioHttpSSEConnection(response, request)
                async for event in agent.replay_events(lid):
                    await conn.send_event(event.type, event.data)
            except ValueError:
                pass

        try:
            while True:
                await asyncio.sleep(30)
                meta.last_activity = time.time()
                if response.task is None or response.task.done():
                    break
                try:
                    await response.write(b": keepalive\n\n")
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            agent._sse_connections.discard(response)

        return response

    async def handle_chat(self, request: web.Request) -> web.Response | web.StreamResponse:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        self.server._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="Invalid JSON")

        message = body.get("message", "")
        if not message:
            raise web.HTTPBadRequest(text="Missing 'message'")

        # Streaming check
        if request.query.get("stream", "").lower() == "true":
            response = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })
            add_cors_headers(self.server, response, request)
            await response.prepare(request)

            stream_req = {"action": "chat", "message": message, "payload": body.get("payload", {})}
            try:
                async for chunk in agent.handle_request_stream(stream_req):
                    await response.write(chunk.encode())
            except Exception as e:
                await response.write(f"event: error\ndata: {str(e)}\n\n".encode())
            return response

        result = await agent.handle_request({"action": "chat", "message": message})
        return web.json_response(result)

    async def handle_invoke(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        self.server._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="Invalid JSON")

        payload = body.get("payload", body)
        result = await agent.handle_request({"action": "invoke", "payload": payload})
        return web.json_response(result)

    async def handle_action(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        action_name = request.match_info["action_name"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        self.server._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except Exception:
            body = {}

        result = await agent.handle_request({
            "action": "action",
            "name": action_name,
            "params": body
        })
        status = 400 if "error" in result else 200
        return web.json_response(result, status=status)

    async def handle_request(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        agent = self.server.get_or_create(agent_id)

        if agent._state is None:
            await agent.start()
            await self.server.on_agent_create(agent)

        self.server._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="Invalid JSON")

        result = await agent.handle_request(body)
        return web.json_response(result)

    async def handle_get_state(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        if agent_id not in self.server._agents:
            raise web.HTTPNotFound()
        
        agent = self.server._agents[agent_id].agent
        try:
            return web.json_response(agent.state)
        except Exception:
            raise web.HTTPInternalServerError(text="State not serializable")

    async def handle_get_messages(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        if agent_id not in self.server._agents:
            raise web.HTTPNotFound()

        agent = self.server._agents[agent_id].agent
        msgs = [{"role": m.role, "content": m.content} for m in agent.messages]
        return web.json_response({"messages": msgs})

    async def handle_put_state(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        if agent_id not in self.server._agents:
            raise web.HTTPNotFound()

        agent = self.server._agents[agent_id].agent
        try:
            new_state = await request.json()
            await agent.set_state(new_state)
            return web.json_response({"status": "ok"})
        except Exception:
            raise web.HTTPBadRequest()

    async def handle_delete_agent(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        agent_id = request.match_info["agent_id"]
        if agent_id not in self.server._agents:
            raise web.HTTPNotFound()

        await self.server.evict(agent_id)
        return web.json_response({"status": "evicted", "agent_id": agent_id})

    async def handle_list_agents(self, request: web.Request) -> web.Response:
        await self._check_auth(request)
        data = []
        for aid, meta in self.server._agents.items():
            data.append({
                "id": aid,
                "connections": meta.connection_count,
                "last_activity": meta.last_activity
            })
        return web.json_response({"agents": data})
