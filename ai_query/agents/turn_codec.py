"""JSON wire codec for public AgentTurn events."""

from __future__ import annotations

import base64
import json
from typing import Any

from ai_query.agents.turn import (
    ReasoningDelta,
    StepFinished,
    StepRetrying,
    StepStarted,
    TextDelta,
    TurnEvent,
    TurnFailed,
    TurnFinished,
    TurnResult,
    TurnStarted,
)
from ai_query.types import (
    Message,
    ReasoningEvent,
    ReasoningPart,
    StepResult,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallReadyEvent,
    ToolCallStartedEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionProgressEvent,
    ToolExecutionStartedEvent,
    ToolResult,
    ToolResultEvent,
    TurnTermination,
    Usage,
    _tool_output_value_from_dict,
    _tool_output_value_to_dict,
)


_BYTES_MARKER = "$ai_query.bytes"


def _to_wire_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {_BYTES_MARKER: base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _to_wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Turn event payload contains non-JSON value {type(value).__name__}"
    )


def _from_wire_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_BYTES_MARKER}:
            encoded = value[_BYTES_MARKER]
            if not isinstance(encoded, str):
                raise ValueError("Invalid encoded bytes in turn event payload")
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("Invalid encoded bytes in turn event payload") from exc
        return {key: _from_wire_value(item) for key, item in value.items()}
    return value


def _usage_to_dict(usage: Usage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cached_tokens,
        "total_tokens": usage.total_tokens,
    }


def _usage_from_dict(data: Any) -> Usage | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Turn event usage must be an object or null")
    return Usage(
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cached_tokens=data.get("cached_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


def _termination_to_dict(
    termination: TurnTermination | None,
) -> dict[str, Any] | None:
    if termination is None:
        return None
    return {
        "kind": termination.kind,
        "provider_finish_reason": termination.provider_finish_reason,
        "reason": termination.reason,
        "stop_condition": termination.stop_condition,
        "tool_name": termination.tool_name,
        "final_step_number": termination.final_step_number,
        "has_text": termination.has_text,
        "has_tool_calls": termination.has_tool_calls,
        "last_tool_error": termination.last_tool_error,
        "error_type": termination.error_type,
        "message": termination.message,
    }


def _termination_from_dict(data: Any) -> TurnTermination | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Turn termination must be an object or null")
    return TurnTermination(
        kind=data["kind"],
        provider_finish_reason=data.get("provider_finish_reason"),
        reason=data.get("reason"),
        stop_condition=data.get("stop_condition"),
        tool_name=data.get("tool_name"),
        final_step_number=data.get("final_step_number", 0),
        has_text=data.get("has_text", False),
        has_tool_calls=data.get("has_tool_calls", False),
        last_tool_error=data.get("last_tool_error"),
        error_type=data.get("error_type"),
        message=data.get("message"),
    )


def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": _to_wire_value(tool_call.arguments),
        "metadata": _to_wire_value(tool_call.metadata),
    }


def _tool_call_from_dict(data: Any) -> ToolCall:
    if not isinstance(data, dict):
        raise ValueError("Turn event tool_call must be an object")
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=_from_wire_value(data.get("arguments", {})),
        metadata=_from_wire_value(data.get("metadata", {})),
    )


def _tool_result_to_dict(tool_result: ToolResult | None) -> dict[str, Any] | None:
    if tool_result is None:
        return None
    return {
        "tool_call_id": tool_result.tool_call_id,
        "tool_name": tool_result.tool_name,
        "result": _to_wire_value(_tool_output_value_to_dict(tool_result.result)),
        "is_error": tool_result.is_error,
    }


def _tool_result_from_dict(data: Any) -> ToolResult | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Turn event tool_result must be an object or null")
    return ToolResult(
        tool_call_id=data["tool_call_id"],
        tool_name=data["tool_name"],
        result=_tool_output_value_from_dict(_from_wire_value(data.get("result"))),
        is_error=data.get("is_error", False),
    )


def _reasoning_to_dict(event: ReasoningEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "provider": event.provider,
        "text": event.text,
        "data": _to_wire_value(event.data),
    }


def _reasoning_from_dict(data: Any) -> ReasoningEvent:
    if not isinstance(data, dict):
        raise ValueError("Turn reasoning event must be an object")
    return ReasoningEvent(
        kind=data["kind"],
        provider=data["provider"],
        text=data.get("text"),
        data=_from_wire_value(data.get("data", {})),
    )


def _reasoning_part_to_dict(part: ReasoningPart) -> dict[str, Any]:
    return {"text": part.text, "data": _to_wire_value(part.data)}


def _reasoning_part_from_dict(data: Any) -> ReasoningPart:
    if not isinstance(data, dict):
        raise ValueError("Turn reasoning part must be an object")
    return ReasoningPart(
        text=data.get("text", ""),
        data=_from_wire_value(data.get("data", {})),
    )


def _step_to_dict(step: StepResult) -> dict[str, Any]:
    return {
        "text": step.text,
        "tool_calls": [_tool_call_to_dict(call) for call in step.tool_calls],
        "tool_results": [_tool_result_to_dict(result) for result in step.tool_results],
        "reasoning_parts": [
            _reasoning_part_to_dict(part) for part in step.reasoning_parts
        ],
        "finish_reason": step.finish_reason,
        "usage": _usage_to_dict(step.usage),
    }


def _step_from_dict(data: Any) -> StepResult:
    if not isinstance(data, dict):
        raise ValueError("Turn step must be an object")
    tool_results = [
        _tool_result_from_dict(item) for item in data.get("tool_results", [])
    ]
    return StepResult(
        text=data.get("text", ""),
        tool_calls=[_tool_call_from_dict(item) for item in data.get("tool_calls", [])],
        tool_results=[item for item in tool_results if item is not None],
        reasoning_parts=[
            _reasoning_part_from_dict(item) for item in data.get("reasoning_parts", [])
        ],
        finish_reason=data.get("finish_reason"),
        usage=_usage_from_dict(data.get("usage")),
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    return _to_wire_value(message.to_dict())


def _message_from_dict(data: Any) -> Message:
    if not isinstance(data, dict):
        raise ValueError("Turn event message must be an object")
    return Message.from_dict(_from_wire_value(data))


def _turn_result_to_dict(result: TurnResult) -> dict[str, Any]:
    return {
        "turn_id": result.turn_id,
        "agent_id": result.agent_id,
        "text": result.text,
        "finish_reason": result.finish_reason,
        "usage": _usage_to_dict(result.usage),
        "steps": [_step_to_dict(step) for step in result.steps],
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "output_message": _message_to_dict(result.output_message),
        "termination": _termination_to_dict(result.termination),
    }


def _turn_result_from_dict(data: Any) -> TurnResult:
    if not isinstance(data, dict):
        raise ValueError("Turn result must be an object")
    return TurnResult(
        turn_id=data["turn_id"],
        agent_id=data["agent_id"],
        text=data.get("text", ""),
        finish_reason=data.get("finish_reason"),
        usage=_usage_from_dict(data.get("usage")),
        steps=[_step_from_dict(step) for step in data.get("steps", [])],
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        output_message=_message_from_dict(data["output_message"]),
        termination=_termination_from_dict(data.get("termination")),
    )


def turn_event_to_dict(event: TurnEvent) -> dict[str, Any]:
    """Serialize a public turn event to its stable JSON-compatible shape."""
    if isinstance(event, TurnStarted):
        return {
            "type": event.type,
            "turn_id": event.turn_id,
            "agent_id": event.agent_id,
            "message": _message_to_dict(event.message),
            "created_at": event.created_at,
        }
    if isinstance(event, TextDelta):
        return {"type": event.type, "text": event.text}
    if isinstance(event, ReasoningDelta):
        return {"type": event.type, "event": _reasoning_to_dict(event.event)}
    if isinstance(event, StepStarted):
        return {"type": event.type, "step_number": event.step_number}
    if isinstance(event, StepRetrying):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "attempt": event.attempt,
            "max_attempts": event.max_attempts,
            "delay": event.delay,
            "error": event.error,
        }
    if isinstance(event, ToolCallStartedEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call_id": event.tool_call_id,
            "name": event.name,
        }
    if isinstance(event, ToolCallDeltaEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call_id": event.tool_call_id,
            "name_delta": event.name_delta,
            "arguments_delta": event.arguments_delta,
        }
    if isinstance(event, ToolCallReadyEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call": _tool_call_to_dict(event.tool_call),
        }
    if isinstance(event, ToolExecutionStartedEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call": _tool_call_to_dict(event.tool_call),
        }
    if isinstance(event, ToolExecutionProgressEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call": _tool_call_to_dict(event.tool_call),
            "message": event.message,
            "data": _to_wire_value(event.data),
        }
    if isinstance(event, ToolExecutionFinishedEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call": _tool_call_to_dict(event.tool_call),
            "tool_result": _tool_result_to_dict(event.tool_result),
            "duration": event.duration,
            "error": event.error,
            "aborted": event.aborted,
        }
    if isinstance(event, ToolResultEvent):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "index": event.index,
            "tool_call": _tool_call_to_dict(event.tool_call),
            "tool_result": _tool_result_to_dict(event.tool_result),
        }
    if isinstance(event, StepFinished):
        return {
            "type": event.type,
            "step_number": event.step_number,
            "step": _step_to_dict(event.step),
            "usage": _usage_to_dict(event.usage),
        }
    if isinstance(event, TurnFinished):
        return {"type": event.type, "result": _turn_result_to_dict(event.result)}
    if isinstance(event, TurnFailed):
        return {
            "type": event.type,
            "error": event.error,
            "error_type": event.error_type,
            "aborted": event.aborted,
            "termination": _termination_to_dict(event.termination),
        }
    raise TypeError(f"Unsupported turn event type: {type(event).__name__}")


def turn_event_from_dict(data: dict[str, Any]) -> TurnEvent:
    """Reconstruct a public turn event from its wire representation."""
    if not isinstance(data, dict):
        raise ValueError("Turn event payload must be an object")
    event_type = data.get("type")
    if event_type == "turn.started":
        return TurnStarted(
            type=event_type,
            turn_id=data["turn_id"],
            agent_id=data["agent_id"],
            message=_message_from_dict(data["message"]),
            created_at=data["created_at"],
        )
    if event_type == "text.delta":
        return TextDelta(type=event_type, text=data.get("text", ""))
    if event_type == "reasoning.delta":
        return ReasoningDelta(
            type=event_type,
            event=_reasoning_from_dict(data["event"]),
        )
    if event_type == "step.started":
        return StepStarted(type=event_type, step_number=data["step_number"])
    if event_type == "step.retrying":
        return StepRetrying(
            type=event_type,
            step_number=data["step_number"],
            attempt=data["attempt"],
            max_attempts=data["max_attempts"],
            delay=data["delay"],
            error=data["error"],
        )
    if event_type == "tool_call.started":
        return ToolCallStartedEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
    if event_type == "tool_call.delta":
        return ToolCallDeltaEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call_id=data.get("tool_call_id"),
            name_delta=data.get("name_delta"),
            arguments_delta=data.get("arguments_delta"),
        )
    if event_type == "tool_call.ready":
        return ToolCallReadyEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call=_tool_call_from_dict(data["tool_call"]),
        )
    if event_type == "tool_execution.started":
        return ToolExecutionStartedEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call=_tool_call_from_dict(data["tool_call"]),
        )
    if event_type == "tool_execution.progress":
        return ToolExecutionProgressEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call=_tool_call_from_dict(data["tool_call"]),
            message=data["message"],
            data=_from_wire_value(data.get("data", {})),
        )
    if event_type == "tool_execution.finished":
        return ToolExecutionFinishedEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call=_tool_call_from_dict(data["tool_call"]),
            tool_result=_tool_result_from_dict(data.get("tool_result")),
            duration=data["duration"],
            error=data.get("error"),
            aborted=data.get("aborted", False),
        )
    if event_type == "tool_result":
        tool_result = _tool_result_from_dict(data["tool_result"])
        if tool_result is None:
            raise ValueError("tool_result event requires a result")
        return ToolResultEvent(
            type=event_type,
            step_number=data["step_number"],
            index=data["index"],
            tool_call=_tool_call_from_dict(data["tool_call"]),
            tool_result=tool_result,
        )
    if event_type == "step.finished":
        return StepFinished(
            type=event_type,
            step_number=data["step_number"],
            step=_step_from_dict(data["step"]),
            usage=_usage_from_dict(data.get("usage")),
        )
    if event_type == "turn.finished":
        return TurnFinished(
            type=event_type,
            result=_turn_result_from_dict(data["result"]),
        )
    if event_type == "turn.failed":
        return TurnFailed(
            type=event_type,
            error=data.get("error", "Remote turn failed"),
            error_type=data.get("error_type"),
            aborted=data.get("aborted", False),
            termination=_termination_from_dict(data.get("termination")),
        )
    raise ValueError(f"Unknown turn event type: {event_type!r}")


def turn_event_to_sse(event: TurnEvent) -> str:
    """Encode one turn event as one SSE frame."""
    payload = json.dumps(
        turn_event_to_dict(event),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event.type}\ndata: {payload}\n\n"
