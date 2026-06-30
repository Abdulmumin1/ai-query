from __future__ import annotations

import pytest

from ai_query.agents.turn import (
    ReasoningDelta,
    StepFinished,
    StepRetrying,
    StepStarted,
    TextDelta,
    TurnFailed,
    TurnFinished,
    TurnResult,
    TurnStarted,
)
from ai_query.agents.turn_codec import turn_event_from_dict, turn_event_to_dict
from ai_query.types import (
    ImagePart,
    Message,
    ReasoningEvent,
    ReasoningPart,
    StepResult,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallReadyEvent,
    ToolCallStartedEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionStartedEvent,
    ToolResult,
    ToolResultEvent,
    Usage,
)


def _sample_step() -> StepResult:
    return StepResult(
        text="done",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="inspect",
                arguments={"path": "README.md", "raw": b"input"},
                metadata={"provider": "mock"},
            )
        ],
        tool_results=[
            ToolResult(
                tool_call_id="call_1",
                tool_name="inspect",
                result={"content": b"result"},
            )
        ],
        reasoning_parts=[ReasoningPart(text="checked", data={"signature": b"sig"})],
        finish_reason="stop",
        usage=Usage(input_tokens=5, output_tokens=2, cached_tokens=3, total_tokens=7),
    )


def _sample_events():
    step = _sample_step()
    call = step.tool_calls[0]
    result = step.tool_results[0]
    usage = step.usage
    turn_result = TurnResult(
        turn_id="turn_1",
        agent_id="agent_1",
        text="done",
        finish_reason="stop",
        usage=usage,
        steps=[step],
        started_at=1.0,
        ended_at=2.0,
        output_message=Message(
            role="assistant",
            content=[ImagePart(image=b"image", media_type="image/png")],
        ),
    )
    return [
        TurnStarted(
            type="turn.started",
            turn_id="turn_1",
            agent_id="agent_1",
            message=Message(role="user", content="hello"),
            created_at=1.0,
        ),
        TextDelta(type="text.delta", text="done"),
        ReasoningDelta(
            type="reasoning.delta",
            event=ReasoningEvent(
                kind="summary",
                provider="mock",
                text="plan",
                data={"signature": b"reasoning"},
            ),
        ),
        StepStarted(type="step.started", step_number=1),
        StepRetrying(
            type="step.retrying",
            step_number=1,
            attempt=2,
            max_attempts=3,
            delay=0.25,
            error="retry",
        ),
        ToolCallStartedEvent(
            type="tool_call.started",
            step_number=1,
            index=0,
            tool_call_id="call_1",
            name="inspect",
        ),
        ToolCallDeltaEvent(
            type="tool_call.delta",
            step_number=1,
            index=0,
            tool_call_id="call_1",
            arguments_delta='{"path":"README.md"}',
        ),
        ToolCallReadyEvent(
            type="tool_call.ready", step_number=1, index=0, tool_call=call
        ),
        ToolExecutionStartedEvent(
            type="tool_execution.started", step_number=1, index=0, tool_call=call
        ),
        ToolExecutionFinishedEvent(
            type="tool_execution.finished",
            step_number=1,
            index=0,
            tool_call=call,
            tool_result=result,
            duration=0.5,
        ),
        ToolResultEvent(
            type="tool_result",
            step_number=1,
            index=0,
            tool_call=call,
            tool_result=result,
        ),
        StepFinished(type="step.finished", step_number=1, step=step, usage=usage),
        TurnFinished(type="turn.finished", result=turn_result),
        TurnFailed(
            type="turn.failed",
            error="aborted",
            error_type="AbortError",
            aborted=True,
        ),
    ]


@pytest.mark.parametrize("event", _sample_events(), ids=lambda event: event.type)
def test_turn_event_codec_round_trips_public_events(event):
    assert turn_event_from_dict(turn_event_to_dict(event)) == event


def test_turn_event_codec_rejects_unknown_event_type():
    with pytest.raises(ValueError, match="Unknown turn event type"):
        turn_event_from_dict({"type": "future.event"})


def test_turn_event_codec_rejects_non_json_tool_result():
    event = ToolResultEvent(
        type="tool_result",
        step_number=1,
        index=0,
        tool_call=ToolCall(id="call_1", name="bad", arguments={}),
        tool_result=ToolResult(
            tool_call_id="call_1",
            tool_name="bad",
            result=object(),
        ),
    )

    with pytest.raises(TypeError, match="non-JSON value object"):
        turn_event_to_dict(event)
