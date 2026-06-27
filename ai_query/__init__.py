"""Main entry point for ai-query library."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    # Main functions
    "generate_text",
    "stream_text",
    "embed",
    "embed_many",
    # Agents
    "Agent",
    "Event",
    "action",
    "AgentHooks",
    "BeforeStepContext",
    "AfterStepContext",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "AgentTurn",
    "TurnEvent",
    "TurnOptions",
    "TurnResult",
    "AgentRegistry",
    "AgentServer",
    "AgentServerConfig",
    "connect",
    "HTTPTransport",
    "LocalTransport",
    # Tool decorators
    "tool",
    "Field",
    # Common types
    "LanguageModel",
    "EmbeddingModel",
    "HTTPStatusError",
    "GenerateTextResult",
    "TextStreamResult",
    "EmbedResult",
    "EmbedManyResult",
    "ReasoningConfig",
    "ReasoningEffort",
    "ReasoningEvent",
    "StreamFinishedEvent",
    "StreamReasoningEvent",
    "StreamStepFinishedEvent",
    "StreamStepStartedEvent",
    "TextDeltaEvent",
    "TextStreamEvent",
    "ToolCallReadyEvent",
    "ToolExecutionFinishedEvent",
    "ToolExecutionStartedEvent",
    "ToolResultEvent",
    "RetryEvent",
    "RetryPolicy",
    "StepControl",
    "BeforeToolCallEvent",
    "BeforeToolCallResult",
    "AfterToolCallEvent",
    "AfterToolCallResult",
    "AbortController",
    "AbortError",
    "AbortSignal",
    # Stop conditions
    "StopCondition",
    "step_count_is",
    "has_tool_call",
    # Callbacks
    "StepStartEvent",
    "StepFinishEvent",
    "OnStepStart",
    "OnStepFinish",
    "OnBeforeToolCall",
    "OnAfterToolCall",
    "OnReasoningEvent",
    "OnRetry",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    # Main functions
    "generate_text": ("ai_query.core", "generate_text"),
    "stream_text": ("ai_query.core", "stream_text"),
    "embed": ("ai_query.core", "embed"),
    "embed_many": ("ai_query.core", "embed_many"),
    # Agents
    "Agent": ("ai_query.agents.agent", "Agent"),
    "Event": ("ai_query.agents.agent", "Event"),
    "action": ("ai_query.agents.agent", "action"),
    "AgentHooks": ("ai_query.agents.hooks", "AgentHooks"),
    "BeforeStepContext": ("ai_query.agents.hooks", "BeforeStepContext"),
    "AfterStepContext": ("ai_query.agents.hooks", "AfterStepContext"),
    "BeforeToolCallContext": ("ai_query.agents.hooks", "BeforeToolCallContext"),
    "AfterToolCallContext": ("ai_query.agents.hooks", "AfterToolCallContext"),
    "AgentTurn": ("ai_query.agents.turn", "AgentTurn"),
    "TurnEvent": ("ai_query.agents.turn", "TurnEvent"),
    "TurnOptions": ("ai_query.agents.turn", "TurnOptions"),
    "TurnResult": ("ai_query.agents.turn", "TurnResult"),
    "AgentRegistry": ("ai_query.agents.registry", "AgentRegistry"),
    "AgentServer": ("ai_query.agents.server", "AgentServer"),
    "AgentServerConfig": ("ai_query.agents.server", "AgentServerConfig"),
    "connect": ("ai_query.agents.remote", "connect"),
    "HTTPTransport": ("ai_query.agents.transport.http", "HTTPTransport"),
    "LocalTransport": ("ai_query.agents.transport", "LocalTransport"),
    # Tool decorators
    "tool": ("ai_query.types", "tool"),
    "Field": ("ai_query.types", "Field"),
    # Common types
    "LanguageModel": ("ai_query.model", "LanguageModel"),
    "EmbeddingModel": ("ai_query.model", "EmbeddingModel"),
    "HTTPStatusError": ("ai_query.transport", "HTTPStatusError"),
    "GenerateTextResult": ("ai_query.types", "GenerateTextResult"),
    "TextStreamResult": ("ai_query.types", "TextStreamResult"),
    "EmbedResult": ("ai_query.types", "EmbedResult"),
    "EmbedManyResult": ("ai_query.types", "EmbedManyResult"),
    "ReasoningConfig": ("ai_query.types", "ReasoningConfig"),
    "ReasoningEffort": ("ai_query.types", "ReasoningEffort"),
    "ReasoningEvent": ("ai_query.types", "ReasoningEvent"),
    "StreamFinishedEvent": ("ai_query.types", "StreamFinishedEvent"),
    "StreamReasoningEvent": ("ai_query.types", "StreamReasoningEvent"),
    "StreamStepFinishedEvent": ("ai_query.types", "StreamStepFinishedEvent"),
    "StreamStepStartedEvent": ("ai_query.types", "StreamStepStartedEvent"),
    "TextDeltaEvent": ("ai_query.types", "TextDeltaEvent"),
    "TextStreamEvent": ("ai_query.types", "TextStreamEvent"),
    "ToolCallReadyEvent": ("ai_query.types", "ToolCallReadyEvent"),
    "ToolExecutionFinishedEvent": ("ai_query.types", "ToolExecutionFinishedEvent"),
    "ToolExecutionStartedEvent": ("ai_query.types", "ToolExecutionStartedEvent"),
    "ToolResultEvent": ("ai_query.types", "ToolResultEvent"),
    "RetryEvent": ("ai_query.types", "RetryEvent"),
    "RetryPolicy": ("ai_query.types", "RetryPolicy"),
    "StepControl": ("ai_query.types", "StepControl"),
    "BeforeToolCallEvent": ("ai_query.types", "BeforeToolCallEvent"),
    "BeforeToolCallResult": ("ai_query.types", "BeforeToolCallResult"),
    "AfterToolCallEvent": ("ai_query.types", "AfterToolCallEvent"),
    "AfterToolCallResult": ("ai_query.types", "AfterToolCallResult"),
    "AbortController": ("ai_query.types", "AbortController"),
    "AbortError": ("ai_query.types", "AbortError"),
    "AbortSignal": ("ai_query.types", "AbortSignal"),
    # Stop conditions
    "StopCondition": ("ai_query.types", "StopCondition"),
    "step_count_is": ("ai_query.types", "step_count_is"),
    "has_tool_call": ("ai_query.types", "has_tool_call"),
    # Callbacks
    "StepStartEvent": ("ai_query.types", "StepStartEvent"),
    "StepFinishEvent": ("ai_query.types", "StepFinishEvent"),
    "OnStepStart": ("ai_query.types", "OnStepStart"),
    "OnStepFinish": ("ai_query.types", "OnStepFinish"),
    "OnBeforeToolCall": ("ai_query.types", "OnBeforeToolCall"),
    "OnAfterToolCall": ("ai_query.types", "OnAfterToolCall"),
    "OnReasoningEvent": ("ai_query.types", "OnReasoningEvent"),
    "OnRetry": ("ai_query.types", "OnRetry"),
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
