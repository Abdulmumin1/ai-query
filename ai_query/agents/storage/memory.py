"""In-memory storage implementation."""

from __future__ import annotations

from typing import Any


class MemoryStorage:
    """In-memory storage for development and testing.

    Data is lost when the process exits.

    Example:
        agent = Agent("assistant", storage=MemoryStorage())
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def keys(self, prefix: str = "") -> list[str]:
        if not prefix:
            return list(self._data.keys())
        return [k for k in self._data.keys() if k.startswith(prefix)]

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
