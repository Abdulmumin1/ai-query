"""Agent module for building stateful AI agents."""

from ai_query.agents.agent import Agent, Event, action
from ai_query.agents.hooks import (
    AgentHooks,
    AfterStepContext,
    AfterToolCallContext,
    BeforeStepContext,
    BeforeToolCallContext,
)
from ai_query.agents.server import AgentServer, AgentServerConfig
from ai_query.agents.turn import AgentTurn, TurnEvent, TurnOptions, TurnResult
from ai_query.agents.websocket import Connection, ConnectionContext
from ai_query.types import BeforeToolCallResult, AfterToolCallResult
from ai_query.agents.storage import (
    Storage,
    MemoryStorage,
    SQLiteStorage,
)

__all__ = [
    # Core
    "Agent",
    "Event",
    "action",
    "AgentHooks",
    "BeforeStepContext",
    "AfterStepContext",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "AgentTurn",
    "TurnEvent",
    "TurnOptions",
    "TurnResult",
    # Storage
    "Storage",
    "MemoryStorage",
    "SQLiteStorage",
    # Multi-agent server
    "AgentServer",
    "AgentServerConfig",
    # WebSocket
    "Connection",
    "ConnectionContext",
]
