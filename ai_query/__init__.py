"""Main entry point for ai-query library."""

from __future__ import annotations

from ai_query.types import (
    GenerateTextResult,
    TextStreamResult,
    EmbedResult,
    EmbedManyResult,
    tool,
    Field,
    StepStartEvent,
    StepFinishEvent,
    OnStepStart,
    OnStepFinish,
    StopCondition,
    step_count_is,
    has_tool_call,
)
from ai_query.model import LanguageModel, EmbeddingModel
from ai_query.core import (
    generate_text,
    stream_text,
    embed,
    embed_many,
)

from ai_query.agents.agent import Agent, Event, action
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
    "GenerateTextResult",
    "TextStreamResult",
    "EmbedResult",
    "EmbedManyResult",
    # Stop conditions
    "StopCondition",
    "step_count_is",
    "has_tool_call",
    # Callbacks
    "StepStartEvent",
    "StepFinishEvent",
    "OnStepStart",
    "OnStepFinish",
]
