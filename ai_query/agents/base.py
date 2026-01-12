"""Abstract base Agent class with state management and WebSocket support.

Implements the Actor model: each agent has a mailbox (queue) and processes
messages sequentially, eliminating race conditions.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, TYPE_CHECKING, AsyncIterator

from ai_query.types import Message
from ai_query.agents.websocket import Connection, ConnectionContext

if TYPE_CHECKING:
    from ai_query.agents.transport import AgentTransport
    from ai_query.agents.events import EventBus
    from ai_query.agents.message import IncomingMessage

State = TypeVar("State")


@dataclass
class _Envelope:
    """Internal message envelope for the actor mailbox."""
    kind: str  # "message", "invoke", "connect", "close", "error"
    payload: Any
    connection: Connection | None = None
    ctx: ConnectionContext | None = None
    future: asyncio.Future | None = None  # For invoke responses


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

        # Actor primitives (injected by AgentServer or manually)
        self._transport: "AgentTransport | None" = None
        self._event_bus: "EventBus | None" = None

        # Actor mailbox for sequential message processing
        self._mailbox: asyncio.Queue[_Envelope] = asyncio.Queue()
        self._processor_task: asyncio.Task | None = None
        self._running = False
    
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

    # ─── Actor Mailbox (Sequential Processing) ─────────────────────────

    async def _process_mailbox(self) -> None:
        """Process messages from the mailbox sequentially.

        This is the actor's main loop. It processes one message at a time,
        ensuring no concurrent state modifications.
        """
        while self._running:
            try:
                envelope = await self._mailbox.get()
            except asyncio.CancelledError:
                break

            try:
                result = await self._handle_envelope(envelope)
                if envelope.future is not None and not envelope.future.done():
                    envelope.future.set_result(result)
            except Exception as e:
                if envelope.future is not None and not envelope.future.done():
                    envelope.future.set_exception(e)
                # For non-invoke messages, call error handler
                elif envelope.connection is not None:
                    await self.on_error(envelope.connection, e)
            finally:
                self._mailbox.task_done()

    async def _handle_envelope(self, envelope: _Envelope) -> Any:
        """Route envelope to the appropriate handler."""
        if envelope.kind == "connect":
            await self.on_connect(envelope.connection, envelope.ctx)  # type: ignore
        elif envelope.kind == "message":
            await self.on_message(envelope.connection, envelope.payload)  # type: ignore
        elif envelope.kind == "close":
            code, reason = envelope.payload
            await self.on_close(envelope.connection, code, reason)  # type: ignore
        elif envelope.kind == "error":
            await self.on_error(envelope.connection, envelope.payload)  # type: ignore
        elif envelope.kind == "invoke":
            return await self.handle_invoke(envelope.payload)
        return None

    def enqueue(
        self,
        kind: str,
        payload: Any,
        connection: Connection | None = None,
        ctx: ConnectionContext | None = None,
    ) -> None:
        """Enqueue a message for sequential processing.

        Args:
            kind: Message type ("message", "connect", "close", "error").
            payload: The message payload.
            connection: Associated WebSocket connection.
            ctx: Connection context (for "connect" kind).
        """
        self._mailbox.put_nowait(_Envelope(
            kind=kind,
            payload=payload,
            connection=connection,
            ctx=ctx,
        ))

    async def enqueue_invoke(
        self,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Enqueue an invoke and wait for the result.

        Args:
            payload: The invoke payload.
            timeout: Maximum time to wait for response.

        Returns:
            The response from handle_invoke.

        Raises:
            asyncio.TimeoutError: If the invoke times out.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        self._mailbox.put_nowait(_Envelope(
            kind="invoke",
            payload=payload,
            future=future,
        ))

        return await asyncio.wait_for(future, timeout=timeout)
    
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
    
    # ─── Actor Communication ─────────────────────────────────────────────
    
    async def invoke(
        self, 
        agent_id: str, 
        payload: dict[str, Any],
        timeout: float = 30.0
    ) -> dict[str, Any]:
        """Call another agent and wait for response.
        
        Sends a request to another agent via the configured transport and
        waits for the response.
        
        Args:
            agent_id: The target agent's identifier.
            payload: The request payload to send.
            timeout: Maximum time to wait for response in seconds.
        
        Returns:
            The response from the target agent.
        
        Raises:
            RuntimeError: If no transport is configured.
        
        Example:
            result = await self.invoke("agent:researcher", {
                "task": "search",
                "query": "AI news"
            })
        """
        if self._transport is None:
            raise RuntimeError(
                "No transport configured. Use AgentServer or set _transport manually."
            )
        return await self._transport.invoke(agent_id, payload, timeout)
    
    async def emit(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event to subscribers.
        
        Events are namespaced with the agent ID: "agent-id:event-name".
        If no event bus is configured, this is a no-op.
        
        Args:
            event: The event name (e.g., "task.complete").
            data: The event payload.
        
        Example:
            await self.emit("analysis.complete", {
                "result": analysis_result
            })
        """
        if self._event_bus is not None:
            await self._event_bus.emit(f"{self.id}:{event}", data)
    
    async def on_receive(self, message: "IncomingMessage") -> Any:
        """Unified handler for all incoming messages.
        
        Override this for a single handler that processes messages from
        any source (WebSocket clients, other agents, HTTP requests).
        
        The default implementation routes to legacy handlers for backward
        compatibility.
        
        Args:
            message: The incoming message with source metadata.
        
        Returns:
            Response for agent invokes, None for client messages.
        
        Example:
            async def on_receive(self, message: IncomingMessage) -> Any:
                if message.source == "client":
                    response = await self.chat(message.content)
                    await message.reply(response)
                elif message.source == "agent":
                    return await self.process_task(message.content)
        """
        if message.source == "client" and message._connection is not None:
            await self.on_message(message._connection, message.content)
        elif message.source == "agent":
            return await self.handle_invoke(
                message.content if isinstance(message.content, dict) else {}
            )
    
    async def handle_invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle an invoke() call from another agent.
        
        Override this to process requests from other agents.
        
        Args:
            payload: The request payload from the calling agent.
        
        Returns:
            Response to send back to the calling agent.
        
        Raises:
            NotImplementedError: If not overridden.
        
        Example:
            async def handle_invoke(self, payload: dict) -> dict:
                task = payload.get("task")
                if task == "summarize":
                    result = await self.summarize(payload["text"])
                    return {"summary": result}
                return {"error": "Unknown task"}
        """
        raise NotImplementedError(
            f"Agent {self.id} does not implement handle_invoke(). "
            "Override this method to handle invoke() calls from other agents."
        )
    
    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Serverless request lifecycle handler.
        
        Use this for stateless serverless environments (Lambda, Cloud Run).
        Handles the full lifecycle: load state → process → save state → respond.
        
        Args:
            request: The request with 'action' and action-specific fields.
                - action='chat': requires 'message'
                - action='invoke': requires 'payload'
                - action='state': returns current state
        
        Returns:
            Response dict with action-specific results.
        
        Example (AWS Lambda):
            def handler(event, context):
                agent = MyAgent(event["agent_id"])
                return asyncio.run(agent.handle_request(event))
        """
        # Ensure agent is started
        if self._state is None:
            await self.start()
        
        action = request.get("action", "chat")
        
        if action == "chat":
            # Import here to avoid circular import
            if hasattr(self, "chat"):
                message = request.get("message", "")
                response = await self.chat(message)  # type: ignore
                return {"agent_id": self.id, "response": response}
            return {"error": "Agent does not support chat"}
        
        elif action == "invoke":
            payload = request.get("payload", {})
            result = await self.handle_invoke(payload)
            return {"agent_id": self.id, "result": result}

        elif action == "state":
            return {"agent_id": self.id, "state": self.state}

        return {"error": f"Unknown action: {action}"}

    async def handle_request_stream(self, request: dict[str, Any]) -> "AsyncIterator[str]":
        """Serverless streaming request handler.

        Handles streaming requests, primarily for 'chat' action.
        Yields SSE-formatted events:
        - event: start (empty data)
        - event: chunk (text delta)
        - event: end (full accumulated text)
        - event: error (error message)

        Args:
            request: The request with 'action' and action-specific fields.

        Yields:
            SSE formatted strings.
        """
        # Ensure agent is started
        if self._state is None:
            await self.start()

        action = request.get("action", "chat")

        if action == "chat":
            if not hasattr(self, "stream_chat"):
                yield "event: error\ndata: Agent does not support stream_chat\n\n"
                return

            message = request.get("message", "")

            try:
                # Start event
                yield "event: start\ndata: \n\n"

                full_text = ""
                # Stream chunks
                async for chunk in self.stream_chat(message):  # type: ignore
                    full_text += chunk
                    # SSE format: data must be a single line or multiple data: lines
                    # We escape newlines to keep it simple JSON string would be safer but this is raw text stream
                    safe_chunk = chunk.replace("\n", "\\n")
                    yield f"event: chunk\ndata: {safe_chunk}\n\n"

                # End event with full text
                # Use JSON to safely encode newlines
                safe_full = json.dumps(full_text)
                yield f"event: end\ndata: {safe_full}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"

        else:
            yield f"event: error\ndata: Streaming not supported for action: {action}\n\n"

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
        if none exists, starts the message processing loop, and calls
        the on_start hook.

        Must be called before interacting with the agent, or use the
        async context manager.
        """
        loaded_state = await self._load_state()
        self._state = loaded_state if loaded_state is not None else self.initial_state
        self._messages = await self._load_messages()

        # Start the actor message processing loop
        self._running = True
        self._processor_task = asyncio.create_task(self._process_mailbox())

        await self.on_start()

    async def stop(self) -> None:
        """Stop the agent's message processing loop.

        Drains remaining messages and cancels the processor task.
        """
        self._running = False

        if self._processor_task is not None:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def __aenter__(self) -> "Agent[State]":
        """Async context manager entry - starts the agent."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - stops agent and closes connections."""
        # Stop the message processor
        await self.stop()

        # Close all WebSocket connections
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
    
    @classmethod
    def serve_many(
        cls,
        host: str = "localhost",
        port: int = 8080,
        config: "AgentServerConfig | None" = None,
    ) -> None:
        """Start a multi-agent server for this agent class.
        
        Each client connects to a unique agent instance via URL path:
        - ws://{host}:{port}/agent/{agent_id}/ws (WebSocket)
        - http://{host}:{port}/agent/{agent_id}/events (SSE)
        - http://{host}:{port}/agent/{agent_id}/state (REST API)
        
        This is a blocking call that runs forever.
        
        Args:
            host: Host to bind to (default: localhost).
            port: Port to bind to (default: 8080).
            config: Optional AgentServerConfig for lifecycle and security.
        
        Example:
            class ChatRoom(ChatAgent, InMemoryAgent):
                system = "You are helpful"
            
            # Clients connect to ws://localhost:8080/agent/room-1/ws
            ChatRoom.serve_many(port=8080)
        """
        from ai_query.agents.router import AgentServer, AgentServerConfig
        AgentServer(cls, config=config).serve(host=host, port=port)

