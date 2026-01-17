"""Storage protocol definition."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Key-value storage interface for agent state and messages.

    Implement this protocol to create custom storage backends.

    Example:
        class RedisStorage:
            def __init__(self, url: str = "redis://localhost"):
                self.client = redis.from_url(url)

            async def get(self, key: str) -> Any | None:
                value = await self.client.get(key)
                return json.loads(value) if value else None

            async def set(self, key: str, value: Any) -> None:
                await self.client.set(key, json.dumps(value))

            async def delete(self, key: str) -> None:
                await self.client.delete(key)

            async def keys(self, prefix: str = "") -> list[str]:
                return await self.client.keys(f"{prefix}*")
    """

    async def get(self, key: str) -> Any | None:
        """Get a value by key. Returns None if not found."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Set a value by key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a value by key."""
        ...

    async def keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        ...
