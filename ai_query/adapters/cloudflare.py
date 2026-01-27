import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Type, Union

# Try to import workers.DurableObject (Cloudflare runtime)
try:
    from workers import DurableObject
except ImportError:

    class DurableObject:  # type: ignore
        def __init__(self, ctx: Any, env: Any):
            self.ctx = ctx
            self.env = env


try:
    import js
    from pyodide.ffi import to_js, create_proxy
except ImportError:
    js = None
    to_js = lambda x: x
    create_proxy = lambda x: x

from ai_query.agents.agent import Agent
from ai_query.agents.storage.cloudflare import DurableObjectStorage
from ai_query.agents.transport.cloudflare import DurableObjectTransport
from ai_query.agents.websocket import Connection, ConnectionContext


class WebSocketBridge(Connection):
    """Bridges a Cloudflare WebSocket to the Agent Connection protocol."""

    def __init__(self, ws: Any):
        self._ws = ws

    async def send(self, data: Union[str, bytes]) -> None:
        """Send data to the client."""
        # Check if ws is still open? Cloudflare WS object handles this.
        self._ws.send(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""
        self._ws.close(code, reason)


class AgentDO(DurableObject):
    """Base class for Cloudflare Durable Objects hosting an Agent.

    Subclass this and set `agent_class` to your Agent implementation.
    Inherits from `workers.DurableObject`.

    Implements WebSocket Hibernation API: https://developers.cloudflare.com/durable-objects/best-practices/websockets/#websocket-hibernation-api
    """

    agent_class: Type[Agent]

    def __init__(self, ctx: Any, env: Any):
        super().__init__(ctx, env)
        self.ctx = ctx
        self.env = env

        # The `id` is available on the context object in the new Python runtime
        self.id = str(ctx.id) if hasattr(ctx, "id") else "unknown"

        self.storage = DurableObjectStorage(ctx.storage)
        self.transport = DurableObjectTransport(env)

        # Provider options extraction
        provider_options: Dict[str, Any] = {}
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            val = getattr(env, key, None)
            if val:
                provider_options[key] = val

        self.agent = self.agent_class(
            id=self.id,
            storage=self.storage,
            transport=self.transport,
            provider_options=provider_options,
        )
        if hasattr(self.agent_class, "model") and not self.agent.model:
            self.agent.model = self.agent_class.model
        self.agent.env = env  # type: ignore

        # Inject emit handler for WebSocket and SSE delivery
        async def cloudflare_emit_handler(
            event_type: str, data: Dict[str, Any], event_id: int
        ) -> None:
            payload = json.dumps({"id": event_id, "type": event_type, "data": data})

            # 1. Broadcast to all accepted WebSockets (Hibernation API)
            for ws in self.ctx.getWebSockets():
                try:
                    ws.send(payload)
                except Exception:
                    pass

            # 2. Broadcast to all active SSE connections
            # We convert to SSE format: "event: <type>\ndata: <json>\n\n"
            sse_payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            for sse_queue in self.agent._sse_connections:
                try:
                    sse_queue.put_nowait(sse_payload)
                except Exception:
                    pass

        self.agent._emit_handler = cloudflare_emit_handler

    async def fetch(self, request: Any) -> Any:
        try:
            # Lazy start: Ensure agent is initialized
            if self.agent._state is None:
                await self.agent.start()

            headers = request.headers
            upgrade = headers.get("Upgrade")
            if upgrade == "websocket":
                return self.handle_websocket_upgrade(request)

            if request.method == "POST":
                body = await request.json()
                data = body.to_py() if hasattr(body, "to_py") else body
                result = await self.agent.handle_request(data)

                # Ensure background tasks (like emit) are completed before hibernation
                self.ctx.waitUntil(self._drain_mailbox())

                return self._json_response(result)
            elif request.method == "GET":
                return self._json_response({"agent_id": self.agent.id})

            return js.Response.new("Method not allowed", {"status": 405})

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error in DO fetch: {e}")
            return self._json_response({"error": str(e)}, status=500)

    def handle_websocket_upgrade(self, request: Any) -> Any:
        """Handle WebSocket upgrade using Hibernation API."""
        # Use Pythonic way to unpack JS iterable
        client, server = js.WebSocketPair.new().object_values()

        # Accept the connection. This attaches the WebSocket to the Durable Object.
        self.ctx.acceptWebSocket(server)

        bridge = WebSocketBridge(server)
        # We assume root path for connection context as standard convention
        ctx = ConnectionContext(path="/", headers={}, query_params={})

        # Trigger the initial on_connect event asynchronously
        # We use waitUntil to ensure on_connect finishes (e.g. if it emits events)
        connect_task = asyncio.create_task(self.agent.on_connect(bridge, ctx))
        self.ctx.waitUntil(connect_task)

        return js.Response.new(
            None,
            js.Object.fromEntries(to_js({"status": 101, "webSocket": client})),
        )

    # --- WebSocket Hibernation Events ---
    # These methods are called by the runtime when events occur on accepted WebSockets.

    async def webSocketMessage(self, ws: Any, message: Any) -> None:
        """Called when a message is received from a WebSocket."""
        # Ensure agent is running. If the DO was hibernating, this will reload state.
        if not self.agent._running:
            await self.agent.start()

        bridge = WebSocketBridge(ws)
        # message might be string or bytes
        await self.agent.on_message(bridge, message)

        # Ensure any side-effects from processing the message are completed
        self.ctx.waitUntil(self._drain_mailbox())

    async def webSocketClose(
        self, ws: Any, code: int, reason: str, wasClean: bool
    ) -> None:
        """Called when a WebSocket is closed."""
        bridge = WebSocketBridge(ws)
        await self.agent.on_close(bridge, code, reason)

    async def webSocketError(self, ws: Any, error: Any) -> None:
        """Called when a WebSocket error occurs."""
        # Errors are typically handled by the connection closing or client retrying
        pass

    def _json_response(self, data: Any, status: int = 200) -> Any:
        return js.Response.new(
            json.dumps(data),
            js.Object.fromEntries(
                to_js(
                    {"status": status, "headers": {"Content-Type": "application/json"}}
                )
            ),
        )

    async def _drain_mailbox(self) -> None:
        """Wait until the agent's mailbox is empty."""
        # join() blocks until all items in the queue have been processed and task_done() is called.
        await self.agent._mailbox.join()

    async def alarm(self) -> None:
        if not self.agent._processor_task or self.agent._processor_task.done():
            self.agent._processor_task = asyncio.create_task(
                self.agent._process_mailbox()
            )


class CloudflareRegistry:
    def __init__(self, env: Any):
        self.env = env
        self.routes: List[tuple[re.Pattern, Any]] = []

    def register(self, pattern: str, binding: Any) -> None:
        self.routes.append((re.compile(pattern), binding))

    async def handle_request(self, request: Any) -> Any:
        url = js.URL.new(request.url)
        path = url.pathname

        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "agent":
            agent_id = parts[1]
            binding = None
            for pattern, b in self.routes:
                if pattern.match(agent_id):
                    binding = b
                    break

            if binding:
                stub = binding.getByName(agent_id)
                return await stub.fetch(request)

        return js.Response.new(
            "Agent not found", js.Object.fromEntries(to_js({"status": 404}))
        )
