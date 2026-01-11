"""Cloudflare Durable Objects agent."""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

from ai_query.agents.base import Agent
from ai_query.agents.transport import AgentTransport
from ai_query.agents.websocket import Connection, ConnectionContext
from ai_query.types import Message

State = TypeVar("State")


class DOTransport(AgentTransport):
    """Cloudflare Durable Object transport via stub.fetch().
    
    Enables agent-to-agent communication between Durable Objects.
    Automatically used when agents inherit from DurableObjectAgent.
    
    Example:
        # Agent A calls Agent B (both are DOs)
        result = await self.invoke("agent-b", {"task": "summarize"})
    """
    
    def __init__(self, agent: "DurableObjectAgent", env: Any, binding_name: str = "AGENTS"):
        """Initialize DO transport.
        
        Args:
            agent: The agent using this transport.
            env: Cloudflare environment bindings.
            binding_name: Name of the DO binding in env (default: "AGENTS").
        """
        self._agent = agent
        self._env = env
        self._binding_name = binding_name
    
    async def invoke(
        self, 
        agent_id: str, 
        payload: dict[str, Any], 
        timeout: float = 30.0
    ) -> dict[str, Any]:
        """Invoke another Durable Object agent.
        
        Args:
            agent_id: The target agent's name/ID.
            payload: The request payload.
            timeout: Timeout in seconds (handled by CF runtime).
        
        Returns:
            Response from the target agent.
        """
        # Get the DO binding
        binding = getattr(self._env, self._binding_name, None)
        if binding is None:
            raise RuntimeError(
                f"DO binding '{self._binding_name}' not found in env. "
                "Check your wrangler.toml configuration."
            )
        
        # Get stub for target agent
        stub = binding.get(binding.idFromName(agent_id))
        
        # Make the invoke request
        response = await stub.fetch(
            "http://internal/invoke",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload),
        )
        
        return await response.json()


class DurableObjectAgent(Agent[State], Generic[State]):
    """
    Agent designed for Cloudflare Durable Objects.
    
    Uses the Durable Object's built-in storage for state persistence
    and integrates with the DO WebSocket API.
    
    Note: This class is designed to be used within a Cloudflare Worker
    environment. The ctx and env parameters are provided by the DO runtime.
    
    Example:
        class MyBot(ChatAgent, DurableObjectAgent):
            initial_state = {"sessions": []}
            
            async def on_message(self, conn, message):
                response = await self.chat(message)
                await conn.send(response)
        
        # In wrangler.toml:
        # [[durable_objects.bindings]]
        # name = "MY_BOT"
        # class_name = "MyBot"
    """
    
    def __init__(self, ctx: Any, env: Any, do_binding: str = "AGENTS"):
        """
        Initialize the Durable Object agent.
        
        Args:
            ctx: The Durable Object context provided by Cloudflare.
            env: The environment bindings.
            do_binding: Name of the DO binding for agent-to-agent calls (default: "AGENTS").
        """
        # Get ID from the DO context
        agent_id = str(ctx.id) if hasattr(ctx, "id") else "do-agent"
        super().__init__(agent_id, env=env)
        
        self._ctx = ctx
        self._storage = ctx.storage
        
        # Auto-configure DO transport
        self._transport = DOTransport(self, env, do_binding)
    
    async def _load_state(self) -> State | None:
        """Load state from Durable Object storage."""
        return await self._storage.get("state")
    
    async def _save_state(self, state: State) -> None:
        """Save state to Durable Object storage."""
        await self._storage.put("state", state)
    
    async def _load_messages(self) -> list[Message]:
        """Load messages from Durable Object storage."""
        messages_data = await self._storage.get("messages")
        if messages_data:
            return [Message(**msg) for msg in messages_data]
        return []
    
    async def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to Durable Object storage."""
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        await self._storage.put("messages", messages_data)
    
    def sql(self, query: str, *params: Any) -> list[dict[str, Any]]:
        """
        Execute SQL against the Durable Object's embedded SQLite.
        
        Note: Requires Durable Objects SQL API (check CF docs for availability).
        
        Args:
            query: The SQL query to execute.
            *params: Query parameters.
            
        Returns:
            List of rows as dictionaries.
        """
        # The DO storage.sql API returns results directly
        return self._storage.sql(query, *params)
    
    # ─── Durable Object HTTP/WebSocket Handling ─────────────────────────
    
    async def fetch(self, request: Any) -> Any:
        """
        Handle HTTP requests to this Durable Object.
        
        This is the main entry point called by the DO runtime.
        Automatically handles WebSocket upgrades.
        
        Args:
            request: The incoming HTTP request.
            
        Returns:
            An HTTP Response object.
        """
        # Start the agent if not started
        if self._state is None:
            await self.start()
        
        # Check for WebSocket upgrade
        upgrade = request.headers.get("Upgrade", "")
        if upgrade.lower() == "websocket":
            return await self._handle_websocket(request)
        
        # Handle invoke requests from other DOs
        url = request.url
        if "/invoke" in url:
            return await self._handle_invoke_request(request)
        
        # Delegate to on_request for regular HTTP
        return await self.on_request(request)
    
    async def _handle_invoke_request(self, request: Any) -> Any:
        """Handle invoke() calls from other Durable Objects."""
        try:
            payload = await request.json()
            result = await self.handle_invoke(payload)
            return Response(  # type: ignore
                json.dumps(result),
                status=200,
                headers={"Content-Type": "application/json"},
            )
        except NotImplementedError as e:
            return Response(  # type: ignore
                json.dumps({"error": str(e)}),
                status=501,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            return Response(  # type: ignore
                json.dumps({"error": str(e)}),
                status=500,
                headers={"Content-Type": "application/json"},
            )
    
    async def on_request(self, request: Any) -> Any:
        """
        Handle regular HTTP requests.
        
        Override this to implement custom HTTP endpoints.
        
        Args:
            request: The HTTP request.
            
        Returns:
            An HTTP Response.
        """
        # Default implementation - return 200 OK
        # Users should override this
        return Response("OK", status=200)  # type: ignore
    
    async def _handle_websocket(self, request: Any) -> Any:
        """Handle WebSocket upgrade requests."""
        # Create WebSocket pair
        pair = WebSocketPair()  # type: ignore
        client, server = pair
        
        # Accept the WebSocket
        self._ctx.acceptWebSocket(server)
        
        # Create connection context
        ctx = ConnectionContext(request=request)
        
        # Call on_connect hook
        await self.on_connect(server, ctx)
        
        # Return upgrade response
        return Response(None, status=101, webSocket=client)  # type: ignore
    
    # ─── Durable Object WebSocket Callbacks ─────────────────────────────
    # These are called by the DO runtime
    
    async def webSocketMessage(self, ws: Connection, message: str | bytes) -> None:
        """
        Called by DO runtime when a WebSocket message is received.
        
        This delegates to the on_message hook.
        """
        await self.on_message(ws, message)
    
    async def webSocketClose(
        self, ws: Connection, code: int, reason: str, wasClean: bool
    ) -> None:
        """
        Called by DO runtime when a WebSocket is closed.
        
        This delegates to the on_close hook.
        """
        await self.on_close(ws, code, reason)
    
    async def webSocketError(self, ws: Connection, error: Any) -> None:
        """
        Called by DO runtime when a WebSocket error occurs.
        
        This delegates to the on_error hook.
        """
        await self.on_error(ws, Exception(str(error)))
