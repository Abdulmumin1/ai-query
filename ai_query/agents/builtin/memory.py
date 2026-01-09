"""In-memory storage agent for development and testing."""

from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar

from ai_query.agents.base import Agent
from ai_query.types import Message

State = TypeVar("State")


class InMemoryAgent(Agent[State], Generic[State]):
    """
    Agent with in-memory storage.
    
    Perfect for development, testing, and simple use cases where
    persistence is not required.
    
    Note: Data is shared across all InMemoryAgent instances via class-level
    storage. Data is lost when the process exits.
    
    Example:
        class MyBot(ChatAgent, InMemoryAgent):
            initial_state = {"counter": 0}
        
        async with MyBot("bot-1") as bot:
            await bot.set_state({"counter": 1})
    """
    
    # Class-level storage shared across all instances
    _store: ClassVar[dict[str, dict[str, Any]]] = {}
    
    async def _load_state(self) -> State | None:
        """Load state from in-memory storage."""
        agent_data = self._store.get(self._id, {})
        return agent_data.get("state")
    
    async def _save_state(self, state: State) -> None:
        """Save state to in-memory storage."""
        if self._id not in self._store:
            self._store[self._id] = {}
        self._store[self._id]["state"] = state
    
    async def _load_messages(self) -> list[Message]:
        """Load messages from in-memory storage."""
        agent_data = self._store.get(self._id, {})
        return agent_data.get("messages", [])
    
    async def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to in-memory storage."""
        if self._id not in self._store:
            self._store[self._id] = {}
        self._store[self._id]["messages"] = messages
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all stored data. Useful for testing."""
        cls._store.clear()
    
    def clear(self) -> None:
        """Clear this agent's stored data."""
        if self._id in self._store:
            del self._store[self._id]
