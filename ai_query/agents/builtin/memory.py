"""In-memory storage agent for development and testing."""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar, Generic, TypeVar

from ai_query.agents.base import Agent
from ai_query.types import Message, AgentEvent

State = TypeVar("State")


class InMemoryAgent(Agent[State], Generic[State]):
    """
    Agent with in-memory storage.

    Perfect for development, testing, and simple use cases where
    persistence is not required.

    Note: Data is shared across all InMemoryAgent instances via class-level
    storage. Data is lost when the process exits. The class-level lock
    ensures thread-safe access to the shared storage.

    Example:
        class MyBot(ChatAgent, InMemoryAgent):
            initial_state = {"counter": 0}

        async with MyBot("bot-1") as bot:
            await bot.set_state({"counter": 1})
    """

    # Class-level storage shared across all instances
    _store: ClassVar[dict[str, dict[str, Any]]] = {}
    _store_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    async def _load_state(self) -> State | None:
        """Load state from in-memory storage."""
        async with self._store_lock:
            agent_data = self._store.get(self._id, {})
            return agent_data.get("state")

    async def _save_state(self, state: State) -> None:
        """Save state to in-memory storage."""
        async with self._store_lock:
            if self._id not in self._store:
                self._store[self._id] = {}
            self._store[self._id]["state"] = state

    async def _load_messages(self) -> list[Message]:
        """Load messages from in-memory storage."""
        async with self._store_lock:
            agent_data = self._store.get(self._id, {})
            return agent_data.get("messages", [])

    async def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to in-memory storage."""
        async with self._store_lock:
            if self._id not in self._store:
                self._store[self._id] = {}
            self._store[self._id]["messages"] = messages

    async def _save_event(self, event: AgentEvent) -> None:
        """Save an event to the persistent log."""
        async with self._store_lock:
            if self._id not in self._store:
                self._store[self._id] = {}

            events = self._store[self._id].get("events", [])

            # Prune expired events
            if self.event_retention > 0:
                cutoff = time.time() - self.event_retention
                events = [e for e in events if e.created_at > cutoff]

            events.append(event)
            self._store[self._id]["events"] = events

    async def _get_events(self, after_id: int | None = None) -> list[AgentEvent]:
        """Get events from the persistent log."""
        async with self._store_lock:
            agent_data = self._store.get(self._id, {})
            events = agent_data.get("events", [])

            if after_id is None:
                return events

            return [e for e in events if e.id > after_id]

    @classmethod
    async def clear_all_async(cls) -> None:
        """Clear all stored data asynchronously. Useful for testing."""
        async with cls._store_lock:
            cls._store.clear()

    @classmethod
    def clear_all(cls) -> None:
        """Clear all stored data. Useful for testing.

        Note: For async contexts, prefer clear_all_async().
        """
        cls._store.clear()

    async def clear_async(self) -> None:
        """Clear this agent's stored data asynchronously."""
        async with self._store_lock:
            if self._id in self._store:
                del self._store[self._id]

    def clear(self) -> None:
        """Clear this agent's stored data.

        Note: For async contexts, prefer clear_async().
        """
        if self._id in self._store:
            del self._store[self._id]
