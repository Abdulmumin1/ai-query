"""Transport abstractions for agent-to-agent communication."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_query.agents.server import AgentServer


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
    
    @abstractmethod
    async def invoke(
        self, 
        agent_id: str, 
        payload: dict[str, Any], 
        timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send a request to another agent and wait for response.
        
        Args:
            agent_id: The target agent's identifier.
            payload: The request payload to send.
            timeout: Maximum time to wait for response in seconds.
        
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
        timeout: float = 30.0
    ) -> dict[str, Any]:
        """Invoke another agent, resolving via registry."""
        target = self._server.registry.resolve(agent_id)
        
        if not isinstance(target, type):
            # Target is a remote transport, delegate to it
            return await target.invoke(agent_id, payload, timeout=timeout)

        # Local execution: get or create the agent
        agent = self._server.get_or_create(agent_id)

        # Ensure agent is started
        if agent._state is None:
            await agent.start()

        # Enqueue the invoke and wait for response with timeout
        future = asyncio.get_running_loop().create_future()
        agent.enqueue("invoke", payload, future=future)
        return await asyncio.wait_for(future, timeout=timeout)
