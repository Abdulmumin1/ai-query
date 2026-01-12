"""Multi-agent router with WebSocket, SSE, and REST API support."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, TypeVar

from aiohttp import web, WSMsgType

from ai_query.agents.websocket import Connection, ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.base import Agent
    from ai_query.agents.transport import AgentTransport
    from ai_query.agents.events import EventBus

State = TypeVar("State")


@dataclass
class AgentServerConfig:
    """Configuration for AgentServer lifecycle and security.
    
    Attributes:
        idle_timeout: Seconds before evicting idle agents (None = never).
        max_agents: Maximum concurrent agents (None = unlimited).
        auth: Async function to validate requests. Return True to allow, False to reject.
        allowed_origins: List of allowed CORS origins (None = allow all).
        base_path: Base path for agent routes (default: "/agent").
        enable_rest_api: Enable state REST endpoints (GET/PUT /agent/{id}/state).
        enable_list_agents: Enable GET /agents endpoint (security risk, off by default).
    
    Example:
        config = AgentServerConfig(
            idle_timeout=300,  # 5 minutes
            max_agents=100,
            auth=my_auth_function,
            allowed_origins=["https://myapp.com"],
        )
    """
    
    # Lifecycle
    idle_timeout: float | None = 300.0
    max_agents: int | None = None
    
    # Security
    auth: Callable[[web.Request], Awaitable[bool]] | None = None
    allowed_origins: list[str] | None = None
    
    # Routes
    base_path: str = "/agent"
    enable_rest_api: bool = True
    enable_list_agents: bool = False


@dataclass
class _AgentMeta:
    """Internal metadata for tracking agent lifecycle."""
    agent: Any  # Agent instance
    last_activity: float = field(default_factory=time.time)
    connection_count: int = 0


class AioHttpConnection(Connection):
    """Wraps aiohttp WebSocket in our Connection interface."""
    
    def __init__(self, ws: web.WebSocketResponse, request: web.Request):
        self._ws = ws
        self._request = request
        self.username: str | None = None
        self.agent_id: str | None = None
    
    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws.send_bytes(message)
        else:
            await self._ws.send_str(message)
    
    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, message=reason.encode())


class AgentServer(Generic[State]):
    """Multi-agent WebSocket server with routing.
    
    Routes clients to independent agent instances based on URL path.
    Each agent maintains its own state, connections, and message history.
    
    Endpoints:
        - GET  {base_path}/{id}/ws     → WebSocket connection
        - GET  {base_path}/{id}/events → SSE for AI streaming
        - GET  {base_path}/{id}/state  → Get agent state (if REST enabled)
        - PUT  {base_path}/{id}/state  → Update agent state (if REST enabled)
        - DELETE {base_path}/{id}      → Evict agent (if REST enabled)
        - GET  /agents                 → List active agents (if enabled)
    
    Example:
        class ChatRoom(ChatAgent, InMemoryAgent):
            system = "You are helpful"
        
        # Start multi-agent server
        AgentServer(ChatRoom).serve(port=8080)
        
        # Clients connect to:
        # ws://localhost:8080/agent/room-1/ws
        # ws://localhost:8080/agent/room-2/ws
    """
    
    def __init__(
        self,
        agent_cls: type["Agent[State]"],
        config: AgentServerConfig | None = None,
        transport: "AgentTransport | None" = None,
        event_bus: "EventBus | None" = None,
    ):
        """Initialize the agent server.
        
        Args:
            agent_cls: The Agent class to instantiate for each ID.
            config: Optional configuration for lifecycle and security.
            transport: Optional custom transport for agent-to-agent communication.
                If not provided, LocalTransport is used.
            event_bus: Optional custom event bus for pub/sub.
                If not provided, LocalEventBus is used.
        """
        self._agent_cls = agent_cls
        self._config = config or AgentServerConfig()
        self._agents: dict[str, _AgentMeta] = {}
        self._eviction_task: asyncio.Task | None = None
        self._transport = transport
        self._event_bus = event_bus
        self._transport_initialized = False
        self._event_bus_initialized = False
    
    # ─── Core API ────────────────────────────────────────────────────────
    
    def get_or_create(self, agent_id: str) -> "Agent[State]":
        """Get or lazily create an agent by ID.
        
        Args:
            agent_id: Unique identifier for the agent.
            
        Returns:
            The agent instance.
            
        Raises:
            web.HTTPTooManyRequests: If max_agents limit is reached.
        """
        if agent_id in self._agents:
            meta = self._agents[agent_id]
            meta.last_activity = time.time()
            return meta.agent
        
        # Check max agents limit
        if (
            self._config.max_agents is not None
            and len(self._agents) >= self._config.max_agents
        ):
            raise web.HTTPTooManyRequests(
                text=f"Maximum number of agents ({self._config.max_agents}) reached"
            )
        
        # Create new agent
        agent = self._agent_cls(agent_id)
        
        # Inject transport and event bus
        if self._transport is None and not self._transport_initialized:
            from ai_query.agents.transport import LocalTransport
            self._transport = LocalTransport(self)
            self._transport_initialized = True
        if self._event_bus is None and not self._event_bus_initialized:
            from ai_query.agents.events import LocalEventBus
            self._event_bus = LocalEventBus()
            self._event_bus_initialized = True
        
        agent._transport = self._transport
        agent._event_bus = self._event_bus
        
        self._agents[agent_id] = _AgentMeta(agent=agent)
        return agent
    
    async def evict(self, agent_id: str) -> None:
        """Evict an agent, closing all connections and removing it.

        Args:
            agent_id: The agent ID to evict.
        """
        if agent_id not in self._agents:
            return

        meta = self._agents[agent_id]
        agent = meta.agent

        # Call lifecycle hook
        await self.on_agent_evict(agent)

        # Stop the agent's message processor
        await agent.stop()

        # Close all connections
        for conn in list(agent._connections):
            try:
                await conn.close(code=1001, reason="Agent evicted")
            except Exception:
                pass
        agent._connections.clear()

        # Close SSE connections
        for sse in list(agent._sse_connections):
            try:
                await sse.write_eof()
            except Exception:
                pass
        agent._sse_connections.clear()

        # Remove from registry
        del self._agents[agent_id]
    
    def list_agents(self) -> list[str]:
        """List all active agent IDs.
        
        Returns:
            List of agent IDs.
        """
        return list(self._agents.keys())
    
    # ─── Lifecycle Hooks ─────────────────────────────────────────────────
    
    async def on_agent_create(self, agent: "Agent") -> None:
        """Called when a new agent is created.
        
        Override this to add custom initialization logic.
        
        Args:
            agent: The newly created agent.
        """
        pass
    
    async def on_agent_evict(self, agent: "Agent") -> None:
        """Called when an agent is about to be evicted.
        
        Override this to add custom cleanup logic.
        
        Args:
            agent: The agent being evicted.
        """
        pass
    
    # ─── Request Handlers ────────────────────────────────────────────────
    
    async def _check_auth(self, request: web.Request) -> None:
        """Check authentication if configured."""
        if self._config.auth is not None:
            allowed = await self._config.auth(request)
            if not allowed:
                raise web.HTTPUnauthorized(text="Authentication required")
    
    def _add_cors_headers(self, response: web.Response) -> web.Response:
        """Add CORS headers if configured."""
        if self._config.allowed_origins:
            origin = ",".join(self._config.allowed_origins)
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    
    async def _handle_options(self, request: web.Request) -> web.Response:
        """Handle CORS preflight requests."""
        return self._add_cors_headers(web.Response())
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        await self._check_auth(request)
        
        agent_id = request.match_info["agent_id"]
        agent = self.get_or_create(agent_id)
        
        # Start agent if needed
        if agent._state is None:
            await agent.start()
            await self.on_agent_create(agent)
        
        # Prepare WebSocket
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Create connection and context
        connection = AioHttpConnection(ws, request)
        connection.agent_id = agent_id
        ctx = ConnectionContext(
            request=request,
            metadata=dict(request.query),
        )
        
        # Track connection
        meta = self._agents[agent_id]
        meta.connection_count += 1
        meta.last_activity = time.time()
        
        # Connect
        agent.enqueue("connect", None, connection=connection, ctx=ctx)

        try:
            async for msg in ws:
                meta.last_activity = time.time()
                if msg.type == WSMsgType.TEXT:
                    agent.enqueue("message", msg.data, connection=connection)
                elif msg.type == WSMsgType.BINARY:
                    agent.enqueue("message", msg.data, connection=connection)
                elif msg.type == WSMsgType.ERROR:
                    agent.enqueue("error", ws.exception(), connection=connection)
        except Exception as e:
            agent.enqueue("error", e, connection=connection)
        finally:
            agent.enqueue("close", (1000, "Client disconnected"), connection=connection)
            meta.connection_count -= 1
            meta.last_activity = time.time()
        
        return ws
    
    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE connections for AI streaming."""
        await self._check_auth(request)
        
        agent_id = request.match_info["agent_id"]
        agent = self.get_or_create(agent_id)
        
        # Start agent if needed
        if agent._state is None:
            await agent.start()
            await self.on_agent_create(agent)
        
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": (
                    ",".join(self._config.allowed_origins)
                    if self._config.allowed_origins
                    else "*"
                ),
            }
        )
        await response.prepare(request)
        
        # Register SSE connection
        agent._sse_connections.add(response)
        meta = self._agents[agent_id]
        
        try:
            while True:
                await asyncio.sleep(30)
                meta.last_activity = time.time()
                await response.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            agent._sse_connections.discard(response)
        
        return response
    
    async def _handle_get_state(self, request: web.Request) -> web.Response:
        """Handle GET /agent/{id}/state."""
        await self._check_auth(request)
        
        agent_id = request.match_info["agent_id"]
        
        if agent_id not in self._agents:
            raise web.HTTPNotFound(text=f"Agent '{agent_id}' not found")
        
        agent = self._agents[agent_id].agent
        
        try:
            state_json = json.dumps(agent.state)
        except (TypeError, ValueError) as e:
            raise web.HTTPInternalServerError(text=f"State not serializable: {e}")
        
        response = web.Response(
            text=state_json,
            content_type="application/json",
        )
        return self._add_cors_headers(response)
    
    async def _handle_put_state(self, request: web.Request) -> web.Response:
        """Handle PUT /agent/{id}/state."""
        await self._check_auth(request)
        
        agent_id = request.match_info["agent_id"]
        
        if agent_id not in self._agents:
            raise web.HTTPNotFound(text=f"Agent '{agent_id}' not found")
        
        agent = self._agents[agent_id].agent
        
        try:
            new_state = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="Invalid JSON in request body")
        
        await agent.set_state(new_state)
        
        response = web.Response(
            text=json.dumps({"status": "ok"}),
            content_type="application/json",
        )
        return self._add_cors_headers(response)
    
    async def _handle_delete_agent(self, request: web.Request) -> web.Response:
        """Handle DELETE /agent/{id}."""
        await self._check_auth(request)
        
        agent_id = request.match_info["agent_id"]
        
        if agent_id not in self._agents:
            raise web.HTTPNotFound(text=f"Agent '{agent_id}' not found")
        
        await self.evict(agent_id)
        
        response = web.Response(
            text=json.dumps({"status": "evicted", "agent_id": agent_id}),
            content_type="application/json",
        )
        return self._add_cors_headers(response)
    
    async def _handle_list_agents(self, request: web.Request) -> web.Response:
        """Handle GET /agents."""
        await self._check_auth(request)

        agents_data = []
        for agent_id, meta in self._agents.items():
            agents_data.append({
                "id": agent_id,
                "connections": meta.connection_count,
                "last_activity": meta.last_activity,
            })

        response = web.Response(
            text=json.dumps({"agents": agents_data}),
            content_type="application/json",
        )
        return self._add_cors_headers(response)

    async def _handle_chat(self, request: web.Request) -> web.Response | web.StreamResponse:
        """Handle POST /agent/{id}/chat - serverless-style chat endpoint."""
        await self._check_auth(request)

        agent_id = request.match_info["agent_id"]
        agent = self.get_or_create(agent_id)

        # Start agent if needed
        if agent._state is None:
            await agent.start()
            await self.on_agent_create(agent)

        # Update activity
        if agent_id in self._agents:
            self._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="Invalid JSON in request body")

        message = body.get("message", "")
        if not message:
            raise web.HTTPBadRequest(text="Missing 'message' field")

        # Check for streaming request
        is_streaming = request.query.get("stream", "").lower() == "true"

        if is_streaming:
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            self._add_cors_headers(response)
            await response.prepare(request)

            stream_request = {
                "action": "chat",
                "message": message,
                "payload": body.get("payload", {})
            }

            try:
                async for chunk in agent.handle_request_stream(stream_request):
                    await response.write(chunk.encode())
            except Exception as e:
                # If stream already started, we can't change status code
                # Send error event
                await response.write(f"event: error\ndata: {str(e)}\n\n".encode())

            return response

        # Use handle_request for consistent behavior
        result = await agent.handle_request({
            "action": "chat",
            "message": message
        })

        response = web.Response(
            text=json.dumps(result),
            content_type="application/json",
        )
        return self._add_cors_headers(response)

    async def _handle_invoke(self, request: web.Request) -> web.Response:
        """Handle POST /agent/{id}/invoke - serverless-style invoke endpoint."""
        await self._check_auth(request)

        agent_id = request.match_info["agent_id"]
        agent = self.get_or_create(agent_id)

        # Start agent if needed
        if agent._state is None:
            await agent.start()
            await self.on_agent_create(agent)

        # Update activity
        if agent_id in self._agents:
            self._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="Invalid JSON in request body")

        payload = body.get("payload", body)

        # Use handle_request for consistent behavior
        result = await agent.handle_request({
            "action": "invoke",
            "payload": payload
        })

        response = web.Response(
            text=json.dumps(result),
            content_type="application/json",
        )
        return self._add_cors_headers(response)

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle POST /agent/{id} - generic request handler.

        Accepts the same format as agent.handle_request():
        - {"action": "chat", "message": "..."}
        - {"action": "invoke", "payload": {...}}
        - {"action": "state"}
        """
        await self._check_auth(request)

        agent_id = request.match_info["agent_id"]
        agent = self.get_or_create(agent_id)

        # Start agent if needed
        if agent._state is None:
            await agent.start()
            await self.on_agent_create(agent)

        # Update activity
        if agent_id in self._agents:
            self._agents[agent_id].last_activity = time.time()

        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="Invalid JSON in request body")

        result = await agent.handle_request(body)

        response = web.Response(
            text=json.dumps(result),
            content_type="application/json",
        )
        return self._add_cors_headers(response)
    
    # ─── Eviction Loop ───────────────────────────────────────────────────
    
    async def _eviction_loop(self) -> None:
        """Background task to evict idle agents."""
        if self._config.idle_timeout is None:
            return
        
        check_interval = min(60.0, self._config.idle_timeout / 2)
        
        while True:
            await asyncio.sleep(check_interval)
            now = time.time()
            
            for agent_id in list(self._agents.keys()):
                meta = self._agents.get(agent_id)
                if meta is None:
                    continue
                
                # Only evict if no connections and idle timeout exceeded
                if (
                    meta.connection_count == 0
                    and now - meta.last_activity > self._config.idle_timeout
                ):
                    print(f"Evicting idle agent: {agent_id}")
                    await self.evict(agent_id)
    
    # ─── Server ──────────────────────────────────────────────────────────
    
    def serve(
        self,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start the multi-agent server (blocking).
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        asyncio.run(self.serve_async(host, port))
    
    async def serve_async(
        self,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start the multi-agent server (async).
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        base = self._config.base_path.rstrip("/")
        
        # Create aiohttp app
        app = web.Application()
        
        # WebSocket and SSE routes
        app.router.add_get(f"{base}/{{agent_id}}/ws", self._handle_websocket)
        app.router.add_get(f"{base}/{{agent_id}}/events", self._handle_sse)
        
        # REST API routes
        if self._config.enable_rest_api:
            app.router.add_get(f"{base}/{{agent_id}}/state", self._handle_get_state)
            app.router.add_put(f"{base}/{{agent_id}}/state", self._handle_put_state)
            app.router.add_post(f"{base}/{{agent_id}}/chat", self._handle_chat)
            app.router.add_post(f"{base}/{{agent_id}}/invoke", self._handle_invoke)
            app.router.add_post(f"{base}/{{agent_id}}", self._handle_request)
            app.router.add_delete(f"{base}/{{agent_id}}", self._handle_delete_agent)
            app.router.add_options(f"{base}/{{agent_id}}/state", self._handle_options)
            app.router.add_options(f"{base}/{{agent_id}}/chat", self._handle_options)
            app.router.add_options(f"{base}/{{agent_id}}/invoke", self._handle_options)
            app.router.add_options(f"{base}/{{agent_id}}", self._handle_options)
        
        # List agents endpoint
        if self._config.enable_list_agents:
            app.router.add_get("/agents", self._handle_list_agents)
        
        # Start eviction loop
        if self._config.idle_timeout is not None:
            self._eviction_task = asyncio.create_task(self._eviction_loop())
        
        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        
        print(f"AgentServer running at http://{host}:{port}")
        print(f"  WebSocket: ws://{host}:{port}{base}/{{agent_id}}/ws")
        print(f"  SSE:       http://{host}:{port}{base}/{{agent_id}}/events")
        if self._config.enable_rest_api:
            print(f"  REST API:")
            print(f"    POST   {base}/{{agent_id}}        - Generic request handler")
            print(f"    POST   {base}/{{agent_id}}/chat   - Chat endpoint")
            print(f"    POST   {base}/{{agent_id}}/invoke - Invoke endpoint")
            print(f"    GET    {base}/{{agent_id}}/state  - Get state")
            print(f"    PUT    {base}/{{agent_id}}/state  - Update state")
            print(f"    DELETE {base}/{{agent_id}}        - Evict agent")
        if self._config.enable_list_agents:
            print(f"  List:      http://{host}:{port}/agents")
        if self._config.idle_timeout:
            print(f"  Idle timeout: {self._config.idle_timeout}s")
        if self._config.max_agents:
            print(f"  Max agents: {self._config.max_agents}")
        
        await site.start()
        
        # Run forever
        try:
            await asyncio.Event().wait()
        finally:
            if self._eviction_task:
                self._eviction_task.cancel()
            await runner.cleanup()
