"""Registry for managing agent types and routing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent
    from ai_query.agents.transport import AgentTransport

# Target can be an Agent class (local execution) or a Transport (remote execution)
RegistryTarget = Union[type["Agent"], "AgentTransport"]


class AgentRegistry:
    """Registry to map agent IDs to their implementation or location.
    
    Allows serving multiple different types of agents from a single server,
    or routing calls to remote agents.
    
    Example:
        registry = AgentRegistry()
        registry.register("writer", WriterAgent)
        registry.register("chat-.*", ChatRoomAgent)
        registry.register("remote-.*", HTTPTransport("..."))
    """

    def __init__(self) -> None:
        self._routes: list[tuple[str, RegistryTarget]] = []
        self._cache: dict[str, RegistryTarget] = {}

    def register(self, pattern: str, target: RegistryTarget) -> None:
        """Register a route pattern.
        
        Args:
            pattern: Regex pattern for agent ID (e.g. "writer", "room-.*").
            target: Agent class (for local) or Transport (for remote).
        """
        # Ensure exact match if it's a simple name (not containing regex chars)
        if all(c.isalnum() or c in "-_" for c in pattern):
            regex = f"^{pattern}$"
        else:
            regex = pattern
            
        self._routes.append((regex, target))
        self._cache.clear()  # Invalidate cache

    def resolve(self, agent_id: str) -> RegistryTarget:
        """Find the target for a given agent ID.
        
        Returns:
            The matched Agent class or Transport.
            
        Raises:
            ValueError: If no matching route is found.
        """
        if agent_id in self._cache:
            return self._cache[agent_id]

        for pattern, target in self._routes:
            if re.match(pattern, agent_id):
                self._cache[agent_id] = target
                return target

        raise ValueError(f"No route found for agent ID: '{agent_id}'")
