"""Remote agent client proxy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, TypeVar, Union

from ai_query.agents.agent import AgentCallProxy
from ai_query.agents.transport.http import HTTPTransport
from ai_query.types import AbortSignal

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent
    from ai_query.agents.turn import TurnEvent

T = TypeVar("T", bound="Agent")


class RemoteAgent:
    """Proxy for interacting with a remote agent via HTTP.

    This class mimics the interface of a local Agent but delegates all operations
    to an underlying transport (HTTPTransport by default).
    """

    def __init__(self, transport: HTTPTransport, agent_id: str):
        self._transport = transport
        self._agent_id = agent_id

    @property
    def id(self) -> str:
        return self._agent_id

    async def chat(
        self,
        message: str,
        *,
        signal: Union[AbortSignal, None] = None,
    ) -> str:
        """Send a chat message to the remote agent."""
        return await self._transport.chat(self._agent_id, message, signal=signal)

    async def stream(
        self,
        message: str,
        *,
        signal: Union[AbortSignal, None] = None,
    ) -> AsyncIterator[str]:
        """Stream a response chunk by chunk."""
        async for chunk in self._transport.stream(
            self._agent_id, message, signal=signal
        ):
            yield chunk

    async def events(
        self,
        message: str,
        *,
        signal: Union[AbortSignal, None] = None,
    ) -> AsyncIterator["TurnEvent"]:
        """Stream typed lifecycle events for a remote agent turn."""
        async for event in self._transport.stream_events(
            self._agent_id, message, signal=signal
        ):
            yield event

    def call(
        self,
        *,
        agent_cls: Union[type[T], None] = None,
        timeout: Union[float, None] = 30.0,
        signal: Union[AbortSignal, None] = None,
    ) -> AgentCallProxy[T]:
        """Returns a type-safe proxy for making fluent calls to the remote agent."""
        return AgentCallProxy(
            self._transport,
            self._agent_id,
            timeout=timeout,
            signal=signal,
        )

    async def close(self) -> None:
        """Close the underlying transport."""
        if hasattr(self._transport, "close"):
            await self._transport.close()


def connect(url: str, headers: Union[dict[str, str], None] = None) -> RemoteAgent:
    """Connect to a remote agent.

    Args:
        url: The full URL to the agent (e.g., "https://api.example.com/agents/researcher").
        headers: Optional headers (e.g. Authorization).

    Returns:
        A RemoteAgent instance.
    """
    # Parse URL to separate base_url and agent_id
    # We assume the last part of the path is the agent_id
    # e.g. .../agents/my-agent -> base=.../agents, id=my-agent

    if url.endswith("/"):
        url = url[:-1]

    parts = url.rsplit("/", 1)
    if len(parts) == 2:
        base_url, agent_id = parts
    else:
        # Fallback for weird URLs
        base_url = url
        agent_id = "default"

    transport = HTTPTransport(base_url=base_url, headers=headers)
    return RemoteAgent(transport, agent_id)
