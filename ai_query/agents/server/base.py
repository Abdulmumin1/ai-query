"""Core AgentServer implementation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from aiohttp import web

from ai_query.agents.registry import AgentRegistry
from ai_query.agents.server.connections import AioHttpConnection, AioHttpSSEConnection
from ai_query.agents.server.types import AgentMeta, AgentServerConfig
from ai_query.agents.transport import LocalTransport
from ai_query.agents.events import LocalEventBus
from ai_query.agents.websocket import ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent
    from ai_query.agents.transport import AgentTransport
    from ai_query.agents.events import EventBus


class AgentServer:
    """Multi-agent WebSocket server with routing.

    Routes clients to independent agent instances based on URL path.
    Uses AgentRegistry to determine which agent class to instantiate for a given ID.
    """

    def __init__(
        self,
        agent_cls_or_registry: type[Agent] | AgentRegistry,
        config: AgentServerConfig | None = None,
        transport: AgentTransport | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize the agent server.

        Args:
            agent_cls_or_registry: Either a specific Agent class (legacy mode)
                                   or an AgentRegistry instance (recommended).
            config: Optional configuration for lifecycle and security.
            transport: Optional custom transport for agent-to-agent communication.
                If not provided, LocalTransport is used.
            event_bus: Optional custom event bus for pub/sub.
                If not provided, LocalEventBus is used.
        """
        # Handle registry vs single class
        if isinstance(agent_cls_or_registry, AgentRegistry):
            self.registry = agent_cls_or_registry
        else:
            self.registry = AgentRegistry()
            # Register the single class to match all IDs
            self.registry.register(".*", agent_cls_or_registry)

        self._config = config or AgentServerConfig()
        self._agents: dict[str, AgentMeta] = {}
        self._eviction_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._transport = transport
        self._event_bus = event_bus
        self._transport_initialized = False
        self._event_bus_initialized = False

    # ─── Core API ────────────────────────────────────────────────────────

    def get_or_create(self, agent_id: str) -> Agent:
        """Get or lazily create an agent by ID using the Registry."""
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

        # Resolve agent class from registry
        target = self.registry.resolve(agent_id)
        
        # If target is a Transport (remote agent), we can't instantiate it locally.
        # This server only hosts LOCAL agents. 
        # Ideally, we should proxy the request if it's remote, but for now we assume
        # this server is only for local hosting.
        if not isinstance(target, type):
             # It's an instance (Transport), not a class
             raise ValueError(f"Agent ID '{agent_id}' resolves to a remote transport, not a local class.")
        
        agent_cls = target
        agent = agent_cls(agent_id)

        # Inject transport, event bus, and emit handler
        if self._transport is None and not self._transport_initialized:
            self._transport = LocalTransport(self)
            self._transport_initialized = True
        if self._event_bus is None and not self._event_bus_initialized:
            self._event_bus = LocalEventBus()
            self._event_bus_initialized = True

        agent._transport = self._transport
        agent._event_bus = self._event_bus
        agent._emit_handler = self._create_emit_handler(agent)

        self._agents[agent_id] = AgentMeta(agent=agent)
        return agent

    def _create_emit_handler(self, agent: "Agent") -> Callable[[str, dict, int], Awaitable[None]]:
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

    async def evict(self, agent_id: str) -> None:
        """Evict an agent, closing all connections and removing it."""
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

        # Close SSE connections gracefully
        for sse in list(agent._sse_connections):
            try:
                # Force close without waiting - handlers will detect and exit
                if not sse.task.done() if sse.task else True:
                    await sse.write_eof()
            except Exception:
                pass
        agent._sse_connections.clear()

        # Remove from registry
        del self._agents[agent_id]

    def list_agents(self) -> list[str]:
        """List all active agent IDs."""
        return list(self._agents.keys())

    # ─── Lifecycle Hooks ─────────────────────────────────────────────────

    async def on_agent_create(self, agent: "Agent") -> None:
        """Called when a new agent is created. Override for custom logic."""
        pass

    async def on_agent_evict(self, agent: "Agent") -> None:
        """Called when an agent is about to be evicted. Override for custom logic."""
        pass

    def on_app_setup(self, app: web.Application) -> None:
        """Hook called after the app is created but before serving."""
        pass
        
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
                    # print(f"Evicting idle agent: {agent_id}")
                    await self.evict(agent_id)

    async def _shutdown(self, runner: web.AppRunner) -> None:
        """Gracefully shut down the server and all agents."""
        # Cancel eviction task
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass

        # Evict all agents
        agent_ids = list(self._agents.keys())
        if agent_ids:
            print(f"Stopping {len(agent_ids)} agent(s)...")
            for agent_id in agent_ids:
                try:
                    await self.evict(agent_id)
                except Exception as e:
                    print(f"Error evicting agent {agent_id}: {e}")

        # Cleanup aiohttp runner
        await runner.cleanup()
        print("Server stopped")

    async def shutdown(self) -> None:
        """Manually trigger server shutdown."""
        if self._shutdown_event:
            self._shutdown_event.set()

    # ─── Server ──────────────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        """Create and return the configured aiohttp Application."""
        # This will be implemented by importing handlers from a separate module
        # to keep this file clean. But for circular dependency reasons, we might
        # need to define handlers here or inject them.
        
        # For now, let's keep the core server logic here and move handlers 
        # to a mixin or just keep them here if we want to avoid complex imports.
        # Given the instruction to break down files, I'll use a mixin approach
        # or just import the handlers function.
        
        # Actually, to properly break the file, we should inherit from a base
        # that has the handlers, or put the handlers in this file but import
        # the request handling logic from another file.
        
        # Let's import the route setup logic.
        from ai_query.agents.server.handlers import setup_routes
        return setup_routes(self)

    def serve(
        self,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start the multi-agent server (blocking)."""
        asyncio.run(self.serve_async(host, port))

    async def serve_async(
        self,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start the multi-agent server (async)."""
        app = self.create_app()
        base = self._config.base_path.rstrip("/")

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
        
        await site.start()

        # Set up graceful shutdown
        self._shutdown_event = asyncio.Event()

        def signal_handler() -> None:
            print("\nShutting down gracefully...")
            event = self._shutdown_event
            if event is not None:
                event.set()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        import signal
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)
        except (NotImplementedError, AttributeError):
            # Windows or non-main thread
            pass

        print("Press Ctrl+C to stop")

        try:
            if self._shutdown_event:
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown_event = None
            await self._shutdown(runner)
