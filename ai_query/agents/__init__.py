"""Agent module for building stateful AI agents."""

from ai_query.agents.base import Agent
from ai_query.agents.chat import ChatAgent
from ai_query.agents.websocket import Connection, ConnectionContext
from ai_query.agents.server import AioHttpConnection
from ai_query.agents.builtin import (
    InMemoryAgent,
    SQLiteAgent,
    DurableObjectAgent,
)

__all__ = [
    # Core
    "Agent",
    "ChatAgent",
    # WebSocket types
    "Connection",
    "ConnectionContext",
    "AioHttpConnection",
    # Built-in agents
    "InMemoryAgent",
    "SQLiteAgent",
    "DurableObjectAgent",
]

