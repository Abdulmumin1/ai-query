"""Transport abstractions for agent-to-agent communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_query.agents.router import AgentServer


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
        """Invoke another agent in the same process.

        Args:
            agent_id: The target agent's identifier.
            payload: The request payload.
            timeout: Maximum time to wait for response.

        Returns:
            The response from the target agent.

        Raises:
            asyncio.TimeoutError: If the agent doesn't respond within timeout.
        """
        agent = self._server.get_or_create(agent_id)

        # Ensure agent is started
        if agent._state is None:
            await agent.start()

        # Enqueue the invoke and wait for response with timeout
        return await agent.enqueue_invoke(payload, timeout=timeout)
