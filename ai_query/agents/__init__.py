"""Agent module for building stateful AI agents."""

from ai_query.agents.base import Agent
from ai_query.agents.chat import ChatAgent
from ai_query.agents.websocket import Connection, ConnectionContext
from ai_query.agents.server import AioHttpConnection
from ai_query.agents.router import AgentServer, AgentServerConfig
from ai_query.agents.message import IncomingMessage
from ai_query.agents.transport import AgentTransport, LocalTransport
from ai_query.agents.events import EventBus, LocalEventBus
from ai_query.agents.builtin import (
    InMemoryAgent,
    SQLiteAgent,
    DurableObjectAgent,
)

__all__ = [
    # Core
    "Agent",
    "ChatAgent",
    # Message types
    "IncomingMessage",
    # Transport
    "AgentTransport",
    "LocalTransport",
    # Events
    "EventBus",
    "LocalEventBus",
    # WebSocket types
    "Connection",
    "ConnectionContext",
    "AioHttpConnection",
    # Multi-agent server
    "AgentServer",
    "AgentServerConfig",
    # Built-in agents
    "InMemoryAgent",
    "SQLiteAgent",
    "DurableObjectAgent",
]
