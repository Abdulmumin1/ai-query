"""Transport-agnostic Agent with event emission and optional persistence."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    TYPE_CHECKING,
    TypeVar,
    Union,
    Set,
    List,
    Dict,
)

from ai_query.agents.storage import MemoryStorage, Storage
from ai_query.agents.websocket import Connection, ConnectionContext
from ai_query.model import LanguageModel
from ai_query.types import AbortError, AbortSignal, Message

if TYPE_CHECKING:
    from ai_query.agents.events import EventBus
    from ai_query.agents.transport import AgentTransport
    from ai_query.types import ProviderOptions, StopCondition, ToolSet


Content = Union[str, List[Any]]

# Type alias for emit handler callback
EmitHandler = Callable[[str, Dict[str, Any], int], Awaitable[None]]

# Generic TypeVar for type-safe agent proxies. T can be any subclass of Agent.
T = TypeVar("T", bound="Agent")


@dataclass
class Context:
    """Context for the current operation being processed by the agent."""

    connection: Union[Connection, None] = None
    ctx: Union[ConnectionContext, None] = None


def action(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark an agent method as an externally callable action."""
    setattr(func, "__is_action__", True)
    return func


@dataclass
class _Envelope:
    kind: str
    payload: Any
    connection: Union[Connection, None] = None
    ctx: Union[ConnectionContext, None] = None
    future: Union[asyncio.Future, None] = None


@dataclass
class Event:
    """An emitted event with ID for replay support."""

    id: int
    type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "data": self.data}


class AgentCallProxy(Generic[T]):
    """A type-safe proxy for making fluent calls to another agent.

    This class enables static analysis and autocompletion for remote agent calls.
    """

    def __init__(self, agent: "Agent", target_id: str):
        self._agent = agent
        self._target_id = target_id

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        """Captures the method name and returns an awaitable RPC function."""

        async def method_proxy(**kwargs: Any) -> Any:
            """The inner function that is awaited to perform the RPC call."""
            if self._agent._transport is None:
                raise RuntimeError(
                    "No transport configured. Agents must be managed by an AgentServer "
                    "or provided a transport manually to use call()."
                )

            # The payload format is compatible with the agent's handle_invoke method
            payload = {"method": name, "params": kwargs}
            result = await self._agent._transport.invoke(self._target_id, payload)

            if "error" in result:
                raise RuntimeError(f"Agent call failed: {result['error']}")

            return result.get("result")

        return method_proxy


class Agent:
    """A transport-agnostic AI agent with event emission and optional persistence.

    The Agent emits events during its work. How those events are delivered to
    clients is determined by the transport layer (serve(), AgentServer, etc.).

    Example:
        class ResearchBot(Agent):
            enable_event_log = True  # Enable replay on reconnect

            async def research(self, query: str):
                await self.emit("status", {"text": "Searching..."})
                results = await self.search(query)
                await self.emit("results", {"count": len(results)})

        # Start server - handles WS, SSE, HTTP automatically
        bot = ResearchBot("assistant", storage=SQLiteStorage("bot.db"))
        bot.serve(port=8080)
    """

    # Class-level flag to enable event persistence for replay
    enable_event_log: bool = False

    def __init__(
        self,
        id: str,
        *,
        model: Union[LanguageModel, None] = None,
        system: str = "You are a helpful assistant.",
        tools: Union[ToolSet, None] = None,
        storage: Union[Storage, None] = None,
        initial_state: Union[Dict[str, Any], None] = None,
        stop_when: Union[StopCondition, List[StopCondition], None] = None,
        provider_options: Union[ProviderOptions, None] = None,
        transport: Union[AgentTransport, None] = None,
        event_bus: Union[EventBus, None] = None,
    ) -> None:
        self._id = id
        self._storage = storage or MemoryStorage()
        self._initial_state = initial_state or {}

        self.model = model
        self.system = system
        self.tools = tools or {}
        self.stop_when = stop_when
        self.provider_options = provider_options

        # Agent state
        self._state: Union[Dict[str, Any], None] = None
        self._messages: List[Message] = []

        # Event log for durability & replay
        self._event_log: List[Dict[str, Any]] = []
        self._event_counter: int = 0

        # Communication
        self._transport = transport
        self._event_bus = event_bus

        # Emit handler - injected by transport layer (serve, AgentServer, etc.)
        # Signature: async def handler(event_type: str, data: dict, event_id: int)
        self._emit_handler: Union[EmitHandler, None] = None

        # Connection tracking (used by serve() internally)
        self._connections: Set[Connection] = set()
        self._sse_connections: Set[Any] = set()

        # Actor mailbox for serialized message processing
        self._mailbox: asyncio.Queue[_Envelope] = asyncio.Queue()
        self._processor_task: Union[asyncio.Task, None] = None
        self._running = False
        self._start_lock = asyncio.Lock()

        # Current context (only valid during message processing)
        self.context = Context()

        # Action registry
        self._action_map: Dict[str, Callable[..., Awaitable[Any]]] = {}

    # ─── Properties ────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._id

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def state(self) -> Dict[str, Any]:
        if self._state is None:
            raise RuntimeError(
                "Agent not started. Call 'await agent.start()' or use "
                "'async with agent:' context manager."
            )
        return self._state

    @property
    def messages(self) -> List[Message]:
        return self._messages

    # ─── Event Emission (The Core API) ─────────────────────────────────────

    async def emit(self, event: str, data: Dict[str, Any]) -> int:
        """Emit an event. Returns the event ID."""
        # Assign ID
        self._event_counter += 1
        event_id = self._event_counter

        # Log event
        event_record = {"id": event_id, "type": event, "data": data}
        self._event_log.append(event_record)

        # Persist if enabled
        if self.enable_event_log:
            await self._storage.set(f"{self._id}:event_log", self._event_log)

        # Deliver via handler (set by transport layer)
        if self._emit_handler:
            await self._emit_handler(event, data, event_id)

        return event_id

    async def replay_events(self, after_id: int = 0) -> AsyncIterator[Event]:
        """Yield events after the given ID for replay."""
        for event_record in self._event_log:
            if event_record["id"] > after_id:
                yield Event(
                    id=event_record["id"],
                    type=event_record["type"],
                    data=event_record["data"],
                )

    async def clear_event_log(self) -> None:
        """Clear the event log."""
        self._event_log = []
        self._event_counter = 0
        if self.enable_event_log:
            await self._storage.delete(f"{self._id}:event_log")

    # ─── State Management ──────────────────────────────────────────────────

    async def set_state(self, state: Dict[str, Any]) -> None:
        """Set the agent's state and persist to storage."""
        self._state = state
        await self._storage.set(f"{self._id}:state", state)
        # Emit state change event
        await self.emit("state", {"state": state})

    async def update_state(self, **kwargs: Any) -> None:
        """Merge updates into the agent's state."""
        new_state = {**self.state, **kwargs}
        await self.set_state(new_state)

    async def clear(self) -> None:
        """Clear message history."""
        self._messages = []
        await self._storage.set(f"{self._id}:messages", [])

    # ─── Chat & Streaming ──────────────────────────────────────────────────

    async def chat(
        self,
        message: Content,
        *,
        signal: Union[AbortSignal, None] = None,
    ) -> str:
        """Send a message and get a response."""
        if signal:
            signal.throw_if_aborted()

        if self.model is None:
            from ai_query.providers.google import google

            self.model = google("gemini-2.0-flash")

        msg_content = message if isinstance(message, str) else message
        self._messages.append(Message(role="user", content=msg_content))

        if signal:
            gen_task = asyncio.create_task(self._do_chat())
            abort_task = asyncio.create_task(signal.wait())

            done, pending = await asyncio.wait(
                [gen_task, abort_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if abort_task in done:
                self._messages.pop()
                raise AbortError(signal.reason)

            return gen_task.result()
        else:
            return await self._do_chat()

    async def _do_chat(self) -> str:
        from ai_query import stream_text
        from ai_query.providers.google import google

        model = self.model or google("gemini-2.0-flash")

        result = stream_text(
            model=model,
            system=self.system,
            messages=self._messages,
            tools=self.tools if self.tools else None,
            stop_when=self.stop_when,
            provider_options=self.provider_options,
        )

        full_response = ""
        async for chunk in result.text_stream:
            full_response += chunk

        self._messages.append(Message(role="assistant", content=full_response))
        await self._storage.set(
            f"{self._id}:messages", [m.to_dict() for m in self._messages]
        )
        return full_response

    async def stream(
        self,
        message: Content,
        *,
        signal: Union[AbortSignal, None] = None,
    ) -> AsyncIterator[str]:
        """Stream a response chunk by chunk."""
        if signal:
            signal.throw_if_aborted()

        if self.model is None:
            from ai_query.providers.google import google

            self.model = google("gemini-2.0-flash")

        # Ensure model is set for type checker
        assert self.model is not None

        msg_content = message if isinstance(message, str) else message
        self._messages.append(Message(role="user", content=msg_content))

        from ai_query import stream_text

        result = stream_text(
            model=self.model,
            system=self.system,
            messages=self._messages,
            tools=self.tools if self.tools else None,
            stop_when=self.stop_when,
            provider_options=self.provider_options,
        )

        full_response = ""
        async for chunk in result.text_stream:
            if signal and signal.aborted:
                self._messages.pop()
                raise AbortError(signal.reason)
            full_response += chunk
            yield chunk

        self._messages.append(Message(role="assistant", content=full_response))
        await self._storage.set(
            f"{self._id}:messages", [m.to_dict() for m in self._messages]
        )

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the agent, loading state from storage.
        
        This method is idempotent and concurrency-safe.
        """
        # Hot State Optimization: Check if already started
        if self._state is not None:
             return

        async with self._start_lock:
             # Double-check inside lock
             if self._state is not None:
                 return

             # Load state
             state = await self._storage.get(f"{self._id}:state")
             self._state = state if state is not None else self._initial_state

             # Load messages
             messages_data = await self._storage.get(f"{self._id}:messages")
             if messages_data:
                 self._messages = [
                     Message(role=m["role"], content=m["content"]) for m in messages_data
                 ]

             # Load event log if persistence is enabled
             if self.enable_event_log:
                 event_log_data = await self._storage.get(f"{self._id}:event_log")
                 if event_log_data:
                     self._event_log = event_log_data
                     self._event_counter = max(
                         (e.get("id", 0) for e in self._event_log), default=0
                     )

             # Build action map
             for name in dir(self):
                 attr = getattr(self, name)
                 if hasattr(attr, "__is_action__"):
                     self._action_map[name] = attr

             self._running = True
             self._processor_task = asyncio.create_task(self._process_mailbox())
             await self.on_start()

    async def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        if self._processor_task is not None:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
        await self.on_stop()

    async def on_start(self) -> None:
        """Override to run logic when agent starts."""
        pass

    async def on_stop(self) -> None:
        """Override to run logic when agent stops."""
        pass

    async def __aenter__(self) -> "Agent":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ─── Connection Handling (used by serve/transport) ─────────────────────

    async def on_connect(self, connection: Connection, ctx: ConnectionContext) -> None:
        """Called when a client connects. Override for custom logic."""
        self._connections.add(connection)

    async def on_message(self, connection: Connection, message: Union[str, bytes]) -> None:
        """Called when a message is received. Override to handle messages."""
        pass

    async def on_close(self, connection: Connection, code: int, reason: str) -> None:
        """Called when a client disconnects. Override for cleanup."""
        self._connections.discard(connection)

    # ─── Actor Mailbox ─────────────────────────────────────────────────────

    async def _process_mailbox(self) -> None:
        """Process enqueued messages serially (actor pattern)."""
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
            finally:
                self._mailbox.task_done()

    async def _handle_envelope(self, envelope: _Envelope) -> Any:
        # Set context for this operation
        self.context.connection = envelope.connection
        self.context.ctx = envelope.ctx

        try:
            if envelope.kind == "connect":
                assert envelope.connection is not None
                assert envelope.ctx is not None
                await self.on_connect(envelope.connection, envelope.ctx)
            elif envelope.kind == "message":
                assert envelope.connection is not None
                await self.on_message(envelope.connection, envelope.payload)
            elif envelope.kind == "close":
                assert envelope.connection is not None
                code, reason = envelope.payload
                await self.on_close(envelope.connection, code, reason)
            elif envelope.kind == "action":
                name = envelope.payload["name"]
                params = envelope.payload["params"]
                if name not in self._action_map:
                    raise ValueError(f"Action '{name}' not found on agent {self.id}")

                res = self._action_map[name](**params)
                if asyncio.iscoroutine(res):
                    return await res
                return res
            return None
        finally:
            # Clear context
            self.context.connection = None
            self.context.ctx = None

    def enqueue(
        self,
        kind: str,
        payload: Any = None,
        connection: Union[Connection, None] = None,
        ctx: Union[ConnectionContext, None] = None,
        future: Union[asyncio.Future, None] = None,
    ) -> None:
        """Enqueue a message for serial processing."""
        self._mailbox.put_nowait(
            _Envelope(
                kind=kind,
                payload=payload,
                connection=connection,
                ctx=ctx,
                future=future,
            )
        )

    # ─── Server ────────────────────────────────────────────────────────────

    def serve(
        self,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start a server with WebSocket, SSE, and REST endpoints."""
        from ai_query.agents.server import AgentServer

        server = AgentServer(self.__class__)
        server.serve(host=host, port=port)

    # ─── Serverless Request Handling ───────────────────────────────────────

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a serverless request."""
        if self._state is None:
            await self.start()

        action = request.get("action", "chat")

        if action == "chat":
            message = request.get("message", "")
            response = await self.chat(message)
            return {"agent_id": self.id, "response": response}

        elif action == "action":
            name = request.get("name")
            params = request.get("params", {})
            if not name:
                return {"error": "Missing 'name' in action request"}

            # Enqueue to ensure sequential execution
            future = asyncio.get_running_loop().create_future()
            self.enqueue("action", {"name": name, "params": params}, future=future)
            result = await future
            return {"agent_id": self.id, "result": result}

        elif action == "invoke":
            payload = request.get("payload", {})
            # Try to handle as action if it has 'method'
            if "method" in payload:
                method = payload["method"]
                params = payload.get("params", {})
                if method in self._action_map:
                    future = asyncio.get_running_loop().create_future()
                    self.enqueue(
                        "action", {"name": method, "params": params}, future=future
                    )
                    result = await future
                    return {"agent_id": self.id, "result": result}

            result = await self.handle_invoke(payload)
            return {"agent_id": self.id, "result": result}

        elif action == "state":
            return {"agent_id": self.id, "state": self.state}

        return {"error": f"Unknown action: {action}"}

    def call(self, agent_id: str, *, agent_cls: type[T]) -> AgentCallProxy[T]:
        """Returns a type-safe proxy for making fluent calls to another agent."""
        return AgentCallProxy(self, agent_id)

    async def handle_invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming call from another agent."""
        method = payload.get("method")
        params = payload.get("params", {})

        if not method:
            return {"error": "Missing 'method' in call payload"}

        # 1. Check if it's a registered action
        if method in self._action_map:
            try:
                # Enqueue to ensure sequential execution
                future = asyncio.get_running_loop().create_future()
                self.enqueue("action", {"name": method, "params": params}, future=future)
                result = await future
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        # 2. Legacy support: look for on_call_ method on self
        handler = getattr(self, f"on_call_{method}", None)
        if handler and callable(handler):
            try:
                res = handler(**params)
                if asyncio.iscoroutine(res):
                    result = await res
                else:
                    result = res
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        return {"error": f"Method '{method}' not implemented on agent {self.id}"}

    async def handle_request_stream(
        self, request: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Serverless streaming request handler."""
        # Ensure agent is started
        if self._state is None:
            await self.start()

        action = request.get("action", "chat")

        if action == "chat":
            message = request.get("message", "")

            try:
                # Start event
                yield "event: start\ndata: \n\n"

                full_text = ""
                # Stream chunks using the agent's stream method
                async for chunk in self.stream(message):
                    full_text += chunk
                    # SSE format: escape newlines for safety
                    safe_chunk = chunk.replace("\n", "\\n")
                    yield f"event: chunk\ndata: {safe_chunk}\n\n"

                # End event with full text
                safe_full = json.dumps(full_text)
                yield f"event: end\ndata: {safe_full}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"
        else:
            yield f"event: error\ndata: Streaming not supported for action: {action}\n\n"
