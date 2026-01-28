import asyncio
import json
import re
import os
from typing import Any, Dict, List, Optional, Type, Union

# Force Cloudflare runtime environment for BaseProvider detection
# This ensures providers know to rely on pyodide-http patching
os.environ["WORKER_RUNTIME"] = "cloudflare"

# Try to import workers (Cloudflare runtime)
try:
    from workers import DurableObject, Response
except ImportError:
    Response = None

    class DurableObject:  # type: ignore
        def __init__(self, ctx: Any, env: Any):
            self.ctx = ctx
            self.env = env


try:
    import js
    from pyodide.ffi import to_js, create_proxy, create_once_callable
except ImportError:
    js = None
    to_js = lambda x: x
    create_proxy = lambda x: x
    create_once_callable = lambda x: x

from ai_query.agents.agent import Agent
from ai_query.agents.storage.cloudflare import DurableObjectStorage
from ai_query.agents.transport.cloudflare import DurableObjectTransport
from ai_query.agents.websocket import Connection, ConnectionContext


def _safe_wait_until(ctx: Any, awaitable: Any) -> None:
    """Wrap awaitable for waitUntil to prevent borrowed proxy destruction.

    In Pyodide, when a Python coroutine is passed to JavaScript's waitUntil,
    it creates a "borrowed proxy" that gets destroyed when the Python function
    returns. Using create_once_callable creates a persistent proxy that
    auto-destroys after being called once.
    """
    try:
        # Wrap the awaitable in create_once_callable to prevent proxy destruction
        # This creates a persistent reference that survives function return
        wrapped = create_once_callable(lambda: awaitable)
        ctx.waitUntil(wrapped())
    except Exception:
        # Fallback to raw awaitable if wrapping fails
        ctx.waitUntil(awaitable)


class WebSocketBridge(Connection):
    """Bridges a Cloudflare WebSocket to the Agent Connection protocol."""

    def __init__(self, ws: Any, parent: Any = None):
        self._ws = ws
        self._parent = parent  # Keep parent alive to prevent borrowed proxy destruction

    async def send(self, data: Union[str, bytes]) -> None:
        """Send data to the client."""
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

                # Parse URL path to determine action type
                url = js.URL.new(request.url)
                path = url.pathname
                parts = path.strip("/").split("/")

                # Path format: /agent/<agent_id>/<action>
                # If action is in the path but not in body, inject it
                if len(parts) >= 3 and "action" not in data:
                    path_action = parts[2]  # e.g., "invoke", "chat"
                    data["action"] = path_action

                # Check if this is a streaming request
                if data.get("stream"):
                    return await self._handle_stream_request(data)

                result = await self.agent.handle_request(data)

                # Ensure background tasks (like emit) are completed before hibernation
                _safe_wait_until(self.ctx, self._drain_mailbox())

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
        # Create WebSocketPair and keep a reference to prevent "borrowed proxy" errors.
        # We strictly follow the docs using object_values(), but ensuring the parent object
        # stays alive during the unpacking.
        ws_pair = js.WebSocketPair.new()
        client, server = ws_pair.object_values()

        self.ctx.acceptWebSocket(server)

        # Pass ws_pair as parent to keep it alive
        bridge = WebSocketBridge(server, parent=ws_pair)

        url = js.URL.new(request.url)
        path = url.pathname

        headers = {}
        if hasattr(request.headers, "items"):
            headers = dict(request.headers.items())
        else:
            # Fallback for potential JS Headers object or other mappings
            try:
                headers = dict(request.headers)
            except Exception:
                pass

        query_params = js.Object.fromEntries(url.searchParams).to_py()

        ctx = ConnectionContext(
            request=None,
            metadata={
                "path": path,
                "headers": headers,
                "query_params": query_params,
            },
        )

        connect_task = asyncio.create_task(self.agent.on_connect(bridge, ctx))
        _safe_wait_until(self.ctx, connect_task)

        return js.Response.new(
            None,
            js.Object.fromEntries(to_js({"status": 101, "webSocket": client})),
        )

    # --- WebSocket Events ---
  
    async def webSocketMessage(self, ws: Any, message: Any) -> None:
        """Called when a message is received from a WebSocket."""
        if not self.agent._running:
            await self.agent.start()

        bridge = WebSocketBridge(ws)
        await self.agent.on_message(bridge, message)

        _safe_wait_until(self.ctx, self._drain_mailbox())

    async def webSocketClose(
        self, ws: Any, code: int, reason: str, wasClean: bool
    ) -> None:
        """Called when a WebSocket is closed."""
        bridge = WebSocketBridge(ws)
        await self.agent.on_close(bridge, code, reason)

    async def webSocketError(self, ws: Any, error: Any) -> None:
        """Called when a WebSocket error occurs."""
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

    async def _handle_stream_request(self, data: Dict[str, Any]) -> Any:
        """Handle a streaming request and return an SSE response.

        Uses ReadableStream to stream SSE events from handle_request_stream.
        """
        # Create a TransformStream for SSE
        transform_stream = js.TransformStream.new()
        readable = transform_stream.readable
        writable = transform_stream.writable
        writer = writable.getWriter()

        async def stream_task():
            try:
                async for sse_chunk in self.agent.handle_request_stream(data):
                    # Write raw SSE chunk (already formatted by handle_request_stream)
                    await writer.write(js.TextEncoder.new().encode(sse_chunk))
            except Exception as e:
                error_msg = f"event: error\ndata: {str(e)}\n\n"
                await writer.write(js.TextEncoder.new().encode(error_msg))
            finally:
                await writer.close()
                _safe_wait_until(self.ctx, self._drain_mailbox())

        # Start streaming task in background
        task = asyncio.create_task(stream_task())
        _safe_wait_until(self.ctx, task)

        # Return SSE response with readable stream
        return js.Response.new(
            readable,
            js.Object.fromEntries(
                to_js(
                    {
                        "status": 200,
                        "headers": {
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    }
                )
            ),
        )

    async def _drain_mailbox(self) -> None:
        """Wait until the agent's mailbox is empty."""
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
