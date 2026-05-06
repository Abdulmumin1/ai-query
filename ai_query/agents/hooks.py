from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ai_query.types import (
    AbortSignal,
    AfterToolCallEvent,
    AfterToolCallResult,
    BeforeToolCallEvent,
    BeforeToolCallResult,
    OnReasoningEvent,
    ReasoningEvent,
    StepControl,
    StepFinishEvent,
    StepStartEvent,
)

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent
    from ai_query.agents.turn import AgentTurn


@dataclass
class BeforeStepContext:
    agent: "Agent[Any]"
    turn: "AgentTurn | None"
    step_number: int
    event: StepStartEvent
    signal: AbortSignal


@dataclass
class AfterStepContext:
    agent: "Agent[Any]"
    turn: "AgentTurn | None"
    step_number: int
    event: StepFinishEvent
    signal: AbortSignal


@dataclass
class BeforeToolCallContext:
    agent: "Agent[Any]"
    turn: "AgentTurn | None"
    step_number: int
    event: BeforeToolCallEvent
    signal: AbortSignal


@dataclass
class AfterToolCallContext:
    agent: "Agent[Any]"
    turn: "AgentTurn | None"
    step_number: int
    event: AfterToolCallEvent
    signal: AbortSignal


BeforeStep = Callable[
    [BeforeStepContext],
    StepControl | None | Awaitable[StepControl | None],
]
AfterStep = Callable[[AfterStepContext], None | Awaitable[None]]
BeforeToolCall = Callable[
    [BeforeToolCallContext],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolCall = Callable[
    [AfterToolCallContext],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


@dataclass
class AgentHooks:
    before_step: BeforeStep | None = None
    after_step: AfterStep | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    on_reasoning_event: OnReasoningEvent | None = None


def merge_hooks(base: AgentHooks | None, override: AgentHooks | None) -> AgentHooks:
    if base is None and override is None:
        return AgentHooks()
    if base is None:
        return AgentHooks(
            before_step=override.before_step if override else None,
            after_step=override.after_step if override else None,
            before_tool_call=override.before_tool_call if override else None,
            after_tool_call=override.after_tool_call if override else None,
            on_reasoning_event=override.on_reasoning_event if override else None,
        )
    if override is None:
        return AgentHooks(
            before_step=base.before_step,
            after_step=base.after_step,
            before_tool_call=base.before_tool_call,
            after_tool_call=base.after_tool_call,
            on_reasoning_event=base.on_reasoning_event,
        )
    return AgentHooks(
        before_step=override.before_step or base.before_step,
        after_step=override.after_step or base.after_step,
        before_tool_call=override.before_tool_call or base.before_tool_call,
        after_tool_call=override.after_tool_call or base.after_tool_call,
        on_reasoning_event=override.on_reasoning_event or base.on_reasoning_event,
    )
