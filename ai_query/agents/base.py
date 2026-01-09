"""Abstract base Agent class with state management and WebSocket support."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ai_query.types import Message
from ai_query.agents.websocket import Connection, ConnectionContext

State = TypeVar("State")


class Agent(ABC, Generic[State]):
    """
    Abstract base class for AI agents.
    
    Provides:
    - Persistent state management (`state`, `set_state`)
    - Message history (`messages`)
    - WebSocket connection handling (`on_connect`, `on_message`, `on_close`)
    - Lifecycle hooks (`on_start`, `on_state_update`)
    
    To create a custom agent, extend this class along with a storage backend
    like InMemoryAgent, SQLiteAgent, or DurableObjectAgent.
    
    Example:
        class MyBot(ChatAgent, InMemoryAgent):
            initial_state = {"counter": 0}
            
            async def on_message(self, conn, msg):
                response = await self.chat(msg)
                await conn.send(response)
    """
    
    initial_state: State = {}  # type: ignore  # Override in subclass
    
    def __init__(self, agent_id: str, *, env: Any = None):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent instance.
            env: Optional environment bindings (for Cloudflare Durable Objects).
        """
        self._id = agent_id
        self._state: State | None = None
        self._messages: list[Message] = []
        self._connections: set[Connection] = set()
        self._sse_connections: set[Any] = set()  # SSE stream responses
        self.env = env
    
    @property
    def id(self) -> str:
        """The agent's unique identifier."""
        return self._id
    
    # ─── Abstract Storage Methods ───────────────────────────────────────
    
    @abstractmethod
    async def _load_state(self) -> State | None:
        """Load state from storage. Implement in backend-specific subclass."""
        ...
    
    @abstractmethod
    async def _save_state(self, state: State) -> None:
        """Save state to storage. Implement in backend-specific subclass."""
        ...
    
    @abstractmethod
    async def _load_messages(self) -> list[Message]:
        """Load message history. Implement in backend-specific subclass."""
        ...
    
    @abstractmethod
    async def _save_messages(self, messages: list[Message]) -> None:
        """Save message history. Implement in backend-specific subclass."""
        ...
    
    # ─── State API ──────────────────────────────────────────────────────
    
    @property
    def state(self) -> State:
        """Current agent state.
        
        Raises:
            RuntimeError: If the agent hasn't been started yet.
        """
        if self._state is None:
            raise RuntimeError(
                "Agent not started. Call 'await agent.start()' or use "
                "'async with agent:' context manager."
            )
        return self._state
    
    async def set_state(self, state: State) -> None:
        """
        Update the agent's state.
        
        This will:
        1. Update the in-memory state
        2. Persist to storage
        3. Call on_state_update hook
        4. Broadcast the new state to all connected clients
        
        Args:
            state: The new state to set.
        """
        self._state = state
        await self._save_state(state)
        self.on_state_update(state, source="server")
        await self._broadcast_state(state)
    
    # ─── Message API ────────────────────────────────────────────────────
    
    @property
    def messages(self) -> list[Message]:
        """Conversation history for this agent."""
        return self._messages
    
    async def save_messages(self, messages: list[Message]) -> None:
        """Persist messages to storage."""
        self._messages = messages
        await self._save_messages(messages)
    
    async def clear_messages(self) -> None:
        """Clear the conversation history."""
        self._messages = []
        await self._save_messages([])
    
    # ─── WebSocket Lifecycle Hooks ──────────────────────────────────────
    
    async def on_connect(self, connection: Connection, ctx: ConnectionContext) -> None:
        """Called when a WebSocket client connects.
        
        Override this to handle new connections. The default implementation
        adds the connection to the internal connection set.
        
        Args:
            connection: The WebSocket connection.
            ctx: Context containing the original request and metadata.
        """
        self._connections.add(connection)
    
    async def on_message(self, connection: Connection, message: str | bytes) -> None:
        """Called when a message is received from a WebSocket client.
        
        Override this to handle incoming messages.
        
        Args:
            connection: The WebSocket connection that sent the message.
            message: The message content (string or bytes).
        """
        pass
    
    async def on_close(
        self, connection: Connection, code: int, reason: str
    ) -> None:
        """Called when a WebSocket client disconnects.
        
        Override this to handle disconnections. The default implementation
        removes the connection from the internal connection set.
        
        Args:
            connection: The WebSocket connection that closed.
            code: The close code.
            reason: The close reason.
        """
        self._connections.discard(connection)
    
    async def on_error(self, connection: Connection, error: Exception) -> None:
        """Called when a WebSocket error occurs.
        
        Override this to handle errors.
        
        Args:
            connection: The WebSocket connection where the error occurred.
            error: The exception that was raised.
        """
        pass
    
    # ─── Agent Lifecycle Hooks ──────────────────────────────────────────
    
    async def on_start(self) -> None:
        """Called when the agent starts.
        
        Override this for initialization logic that needs to run after
        state has been loaded.
        """
        pass
    
    def on_state_update(self, state: State, source: str | Connection) -> None:
        """Called when state changes from any source.
        
        Override this to react to state updates.
        
        Args:
            state: The new state.
            source: "server" if updated by the agent, or a Connection if
                   updated by a client.
        """
        pass
    
    # ─── Broadcast to Connections ───────────────────────────────────────
    
    async def broadcast(self, message: str | bytes) -> None:
        """Send a message to all connected WebSocket clients.
        
        Args:
            message: The message to broadcast.
        """
        for conn in list(self._connections):
            try:
                await conn.send(message)
            except Exception:
                # Connection may have closed, remove it
                self._connections.discard(conn)
    
    async def _broadcast_state(self, state: State) -> None:
        """Broadcast state update to all connected clients.
        
        Sends a JSON message with type "state" and the new state data.
        """
        if self._connections:
            try:
                message = json.dumps({"type": "state", "data": state})
                await self.broadcast(message)
            except (TypeError, ValueError):
                # State is not JSON serializable, skip broadcast
                pass
    
    # ─── SSE Streaming ─────────────────────────────────────────────────
    
    async def stream_to_sse(self, event: str, data: str) -> None:
        """Send an SSE event to all connected SSE clients.
        
        Args:
            event: The event type (e.g., "ai_chunk", "ai_start", "ai_end").
            data: The event data to send.
        """
        message = f"event: {event}\ndata: {data}\n\n"
        for conn in list(self._sse_connections):
            try:
                await conn.write(message.encode())
            except Exception:
                self._sse_connections.discard(conn)
    
    # ─── Agent Lifecycle ────────────────────────────────────────────────
    
    async def start(self) -> None:
        """Initialize the agent.
        
        This loads state and messages from storage, sets initial state
        if none exists, and calls the on_start hook.
        
        Must be called before interacting with the agent, or use the
        async context manager.
        """
        loaded_state = await self._load_state()
        self._state = loaded_state if loaded_state is not None else self.initial_state
        self._messages = await self._load_messages()
        await self.on_start()
    
    async def __aenter__(self) -> "Agent[State]":
        """Async context manager entry - starts the agent."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - closes all connections."""
        for conn in list(self._connections):
            try:
                await conn.close()
            except Exception:
                pass
        self._connections.clear()
    
    # ─── WebSocket Server ──────────────────────────────────────────────
    
    def serve(
        self,
        host: str = "localhost",
        port: int = 8080,
        path: str = "/ws",
    ) -> None:
        """Start a WebSocket server for this agent.
        
        Uses aiohttp by default. This is a blocking call that runs forever.
        
        Args:
            host: Host to bind to (default: localhost).
            port: Port to bind to (default: 8080).
            path: WebSocket endpoint path (default: /ws).
        
        Example:
            class MyBot(ChatAgent, InMemoryAgent):
                system = "Hello!"
            
            MyBot("my-bot").serve(port=8080)
        """
        from ai_query.agents.server import run_agent_server
        run_agent_server(self, host=host, port=port, path=path)
    
    async def serve_async(
        self,
        host: str = "localhost",
        port: int = 8080,
        path: str = "/ws",
    ) -> None:
        """Start a WebSocket server for this agent (async version).
        
        Args:
            host: Host to bind to (default: localhost).
            port: Port to bind to (default: 8080).
            path: WebSocket endpoint path (default: /ws).
        """
        from ai_query.agents.server import run_agent_server_async
        await run_agent_server_async(self, host=host, port=port, path=path)
