"""Transport abstractions for agent-to-agent communication."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ai_query.agents.agent import AgentEventHandler, Event
    from ai_query.agents.server import AgentServer
    from ai_query.types import AbortSignal


_active_call_observer: ContextVar[Union["_AgentCallEventObserver", None]] = (
    ContextVar("active_agent_call_event_observer", default=None)
)


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    """Retrieve late target failures after a timed-out caller has detached."""
    if not future.cancelled():
        future.exception()


class _AgentCallEventObserver:
    """Invocation-owned observer that can be detached by its caller."""

    def __init__(self, handler: "AgentEventHandler") -> None:
        self._handler: Union["AgentEventHandler", None] = handler
        self._delivery_task: Union[asyncio.Task[None], None] = None

    def raise_if_reentrant(self) -> None:
        if _active_call_observer.get() is self:
            raise RuntimeError(
                "agent call event handlers cannot re-enter emit() on the target agent"
            )

    async def notify(self, event: "Event") -> None:
        handler = self._handler
        if handler is None:
            return
        self.raise_if_reentrant()

        async def deliver() -> None:
            token = _active_call_observer.set(self)
            try:
                await handler(event)
            finally:
                _active_call_observer.reset(token)

        delivery_task = asyncio.create_task(deliver())
        self._delivery_task = delivery_task
        try:
            try:
                await delivery_task
            except asyncio.CancelledError:
                # Invocation cleanup may close the observer while detached work
                # is still attempting delivery. That is a normal lifecycle
                # boundary, not a handler failure to surface through emit().
                if self._handler is None:
                    return
                raise
        finally:
            if self._delivery_task is delivery_task:
                self._delivery_task = None

    def close(self) -> None:
        """Stop future delivery and release the caller-owned callback."""
        self._handler = None
        if self._delivery_task is not None:
            self._delivery_task.cancel()


class AgentTransport(ABC):
    """Abstract base for agent-to-agent communication.

    Transports handle how agents communicate with each other. The default
    LocalTransport works for agents in the same process. Users can implement
    custom transports for distributed scenarios (Redis, HTTP, etc.).

    Example custom transport:
        class RedisTransport(AgentTransport):
            def __init__(self, redis_url: str):
                self.redis = Redis.from_url(redis_url)

            async def invoke(self, agent_id: str, payload: dict, timeout: float) -> dict:
                # Publish to agent's channel, wait for response
                ...
    """

    supports_call_events = False

    @abstractmethod
    async def invoke(
        self,
        agent_id: str,
        payload: dict[str, Any],
        timeout: Union[float, None] = 30.0,
        signal: Union["AbortSignal", None] = None,
        on_event: Union["AgentEventHandler", None] = None,
    ) -> dict[str, Any]:
        """Send a request to another agent and wait for response.

        Args:
            agent_id: The target agent's identifier.
            payload: The request payload to send.
            timeout: Maximum time to wait for response in seconds, or ``None``
                to wait without a deadline.
            signal: Optional abort signal to cancel the request.
            on_event: Optional handler for events emitted by the target while
                this invocation is active.

        Returns:
            The response from the target agent.


        Raises:
            TimeoutError: If the agent doesn't respond within timeout.
            RuntimeError: If the agent cannot be reached.
        """
        ...


class LocalTransport(AgentTransport):
    """In-process transport via AgentServer.

    This is the default transport used when agents are running in the same
    process. It enqueues invokes to the target agent's mailbox, ensuring
    sequential processing.
    """

    supports_call_events = True

    def __init__(self, server: "AgentServer"):
        """Initialize with reference to the AgentServer.

        Args:
            server: The AgentServer managing agents.
        """
        self._server = server

    async def invoke(
        self,
        agent_id: str,
        payload: dict[str, Any],
        timeout: Union[float, None] = 30.0,
        signal: Union["AbortSignal", None] = None,
        on_event: Union["AgentEventHandler", None] = None,
    ) -> dict[str, Any]:
        """Invoke another agent, resolving via registry."""
        target = self._server.registry.resolve(agent_id)

        if not isinstance(target, type):
            # Target is a remote transport, delegate to it
            if on_event is None:
                return await target.invoke(
                    agent_id, payload, timeout=timeout, signal=signal
                )
            if not getattr(target, "supports_call_events", False):
                raise NotImplementedError(
                    f"{type(target).__name__} does not support observing invoke events"
                )
            return await target.invoke(
                agent_id,
                payload,
                timeout=timeout,
                signal=signal,
                on_event=on_event,
            )

        # Local execution: get or create the agent
        agent = self._server.get_or_create(agent_id)

        # Ensure agent is started
        if agent._state is None:
            await agent.start()

        # Enqueue the invoke and wait for response with timeout
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_future_exception)
        call_observer = _AgentCallEventObserver(on_event) if on_event else None
        agent.enqueue(
            "invoke",
            payload,
            future=future,
            signal=signal,
            call_observer=call_observer,
        )

        async def wait_for_result_or_abort() -> dict[str, Any]:
            if signal is None:
                return await asyncio.shield(future)

            abort_task = asyncio.create_task(signal.wait())
            try:
                done, _pending = await asyncio.wait(
                    (future, abort_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_task not in done:
                    return future.result()

                # Preserve terminal state and events from cooperative abort
                # cleanup, but do not let an uncooperative target hold the
                # caller or its observer indefinitely. A target that ignores
                # the signal may still return during this grace period; that
                # late success must not override the caller's abort.
                try:
                    terminal_response = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError as error:
                    raise asyncio.TimeoutError(
                        f"Agent call aborted: {signal.reason}"
                    ) from error
                if "error" in terminal_response:
                    return terminal_response
                raise asyncio.TimeoutError(f"Agent call aborted: {signal.reason}")
            finally:
                abort_task.cancel()
                try:
                    await abort_task
                except asyncio.CancelledError:
                    pass

        try:
            try:
                return await asyncio.wait_for(
                    wait_for_result_or_abort(),
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                if signal and signal.aborted:
                    # Give a cooperative target time to observe the shared signal
                    # and persist terminal state before the caller unwinds.
                    try:
                        await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
                    except (Exception, asyncio.CancelledError):
                        pass
                raise
        finally:
            if call_observer is not None:
                call_observer.close()
