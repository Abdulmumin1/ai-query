"""Event bus abstractions for pub/sub communication."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import Any, Callable, Awaitable


# Type for event handlers
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class EventBus(ABC):
    """Abstract base for event pub/sub.
    
    Event buses handle publishing and subscribing to events. The default
    LocalEventBus works for agents in the same process. Users can implement
    custom event buses for distributed scenarios (Redis pub/sub, Pusher, etc.).
    
    Example custom event bus:
        class RedisEventBus(EventBus):
            def __init__(self, redis_url: str):
                self.redis = Redis.from_url(redis_url)
            
            async def emit(self, event: str, data: dict) -> None:
                await self.redis.publish(f"events:{event}", json.dumps(data))
    """
    
    @abstractmethod
    async def emit(self, event: str, data: dict[str, Any]) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: The event name (e.g., "agent-123:task.complete").
            data: The event payload.
        """
        ...
    
    @abstractmethod
    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        """Subscribe to events matching a pattern.
        
        Args:
            pattern: Glob pattern to match events (e.g., "*:task.*").
            handler: Async function called when matching event is emitted.
        """
        ...
    
    @abstractmethod
    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from a pattern.
        
        Args:
            pattern: The pattern to unsubscribe from.
            handler: The handler to remove.
        """
        ...


class LocalEventBus(EventBus):
    """In-process event bus.
    
    Distributes events to handlers within the same process. Events are
    delivered asynchronously via asyncio tasks.
    """
    
    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = {}
    
    async def emit(self, event: str, data: dict[str, Any]) -> None:
        """Emit event to all matching subscribers.
        
        Handlers are called asynchronously and don't block the emit.
        
        Args:
            event: The event name.
            data: The event payload.
        """
        for pattern, handlers in self._handlers.items():
            if fnmatch(event, pattern):
                for handler in handlers:
                    # Fire and forget - don't await
                    asyncio.create_task(self._safe_call(handler, event, data))
    
    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        """Subscribe to events matching pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*:task.*" matches "agent-1:task.complete").
            handler: Function to call when event matches.
        """
        if pattern not in self._handlers:
            self._handlers[pattern] = []
        if handler not in self._handlers[pattern]:
            self._handlers[pattern].append(handler)
    
    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        """Remove a handler from a pattern.
        
        Args:
            pattern: The pattern to unsubscribe from.
            handler: The handler to remove.
        """
        if pattern in self._handlers:
            try:
                self._handlers[pattern].remove(handler)
            except ValueError:
                pass
    
    async def _safe_call(
        self, 
        handler: EventHandler, 
        event: str, 
        data: dict[str, Any]
    ) -> None:
        """Call handler with error handling."""
        try:
            result = handler(event, data)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as e:
            # Log but don't crash
            print(f"Event handler error for {event}: {e}")
