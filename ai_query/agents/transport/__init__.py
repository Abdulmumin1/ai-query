"""Transport adapters for agent communication."""

from .base import AgentTransport, LocalTransport
# We don't import HTTPTransport here to avoid hard dependency on httpx
# unless user wants it. But for DX, it's usually better to expose it.
# Let's verify dependencies.

__all__ = ["AgentTransport", "LocalTransport"]
