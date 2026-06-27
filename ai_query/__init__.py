"""Main entry point for ai-query library."""

from __future__ import annotations

from ai_query.types import (
    AfterToolCallEvent,
    AfterToolCallResult,
    AbortController,
    AbortError,
    AbortSignal,
    BeforeToolCallEvent,
    BeforeToolCallResult,
    GenerateTextResult,
    TextStreamResult,
    EmbedResult,
    EmbedManyResult,
    tool,
    Field,
    StepControl,
    StepStartEvent,
    StepFinishEvent,
    ReasoningEvent,
    StreamFinishedEvent,
    StreamReasoningEvent,
    StreamStepFinishedEvent,
    StreamStepStartedEvent,
    TextDeltaEvent,
    TextStreamEvent,
    RetryEvent,
    RetryPolicy,
    OnReasoningEvent,
    OnRetry,
    OnBeforeToolCall,
    OnAfterToolCall,
    OnStepStart,
    OnStepFinish,
    StopCondition,
    step_count_is,
    has_tool_call,
    ReasoningConfig,
    ReasoningEffort,
)
from ai_query.model import LanguageModel, EmbeddingModel
from ai_query.transport import HTTPStatusError
from ai_query.core import (
    generate_text,
    stream_text,
    embed,
    embed_many,
)

from ai_query.agents.agent import Agent, Event, action
from ai_query.agents.hooks import (
    AgentHooks,
    AfterStepContext,
    AfterToolCallContext,
    BeforeStepContext,
    BeforeToolCallContext,
)
from ai_query.agents.turn import AgentTurn, TurnEvent, TurnOptions, TurnResult
from ai_query.agents.registry import AgentRegistry
from ai_query.agents.remote import connect
from ai_query.agents.server import AgentServer, AgentServerConfig
from ai_query.agents.transport.http import HTTPTransport
from ai_query.agents.transport import LocalTransport

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
