"""Agent module for building stateful AI agents."""

from __future__ import annotations

from importlib import import_module

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
    "RetryPolicy",
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


_EXPORTS: dict[str, tuple[str, str]] = {
    "Agent": ("ai_query.agents.agent", "Agent"),
    "Event": ("ai_query.agents.agent", "Event"),
    "action": ("ai_query.agents.agent", "action"),
    "AgentHooks": ("ai_query.agents.hooks", "AgentHooks"),
    "BeforeStepContext": ("ai_query.agents.hooks", "BeforeStepContext"),
    "AfterStepContext": ("ai_query.agents.hooks", "AfterStepContext"),
    "BeforeToolCallContext": ("ai_query.agents.hooks", "BeforeToolCallContext"),
    "AfterToolCallContext": ("ai_query.agents.hooks", "AfterToolCallContext"),
    "BeforeToolCallResult": ("ai_query.types", "BeforeToolCallResult"),
    "AfterToolCallResult": ("ai_query.types", "AfterToolCallResult"),
    "AgentTurn": ("ai_query.agents.turn", "AgentTurn"),
    "TurnEvent": ("ai_query.agents.turn", "TurnEvent"),
    "TurnOptions": ("ai_query.agents.turn", "TurnOptions"),
    "TurnResult": ("ai_query.agents.turn", "TurnResult"),
    "RetryPolicy": ("ai_query.types", "RetryPolicy"),
    "Storage": ("ai_query.agents.storage", "Storage"),
    "MemoryStorage": ("ai_query.agents.storage", "MemoryStorage"),
    "SQLiteStorage": ("ai_query.agents.storage", "SQLiteStorage"),
    "AgentServer": ("ai_query.agents.server", "AgentServer"),
    "AgentServerConfig": ("ai_query.agents.server", "AgentServerConfig"),
    "Connection": ("ai_query.agents.websocket", "Connection"),
    "ConnectionContext": ("ai_query.agents.websocket", "ConnectionContext"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
