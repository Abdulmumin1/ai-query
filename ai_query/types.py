"""Core types for ai-query library."""

from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import (
    MISSING as DATACLASS_MISSING,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
)
from types import UnionType
from typing import (
    Any,
    Annotated,
    Literal,
    NotRequired,
    Required,
    Union,
    AsyncIterator,
    Callable,
    Awaitable,
    get_type_hints,
    get_origin,
    get_args,
    is_typeddict,
    overload,
    TypedDict,
)

# Message types
Role = Literal["system", "user", "assistant", "tool"]


# =============================================================================
# Field - Pydantic-style parameter metadata
# =============================================================================

# Sentinel for required fields (like Pydantic's ...)
_MISSING = object()


class Field:
    """Define metadata for a tool parameter."""

    def __init__(
        self,
        description: str = "",
        default: Any = _MISSING,
        enum: Union[list[Any], None] = None,
        min_value: Union[float, None] = None,
        max_value: Union[float, None] = None,
    ) -> None:
        self.description = description
        self.default = default
        self.enum = enum
        self.min_value = min_value
        self.max_value = max_value

    def __repr__(self) -> str:
        return f"Field(description={self.description!r}, default={self.default!r})"


def _literal_value_to_json_type(value: Any) -> Union[str, None]:
    """Map a literal value to its JSON Schema type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return None


def _typed_dict_to_json_schema(
    typed_dict_cls: type[Any],
    seen: set[Any],
    globalns: Union[dict[str, Any], None] = None,
    localns: Union[dict[str, Any], None] = None,
) -> dict[str, Any]:
    """Convert a TypedDict type to JSON Schema."""
    if typed_dict_cls in seen:
        return {"type": "object"}

    seen.add(typed_dict_cls)
    try:
        hints = get_type_hints(
            typed_dict_cls,
            globalns=globalns,
            localns=localns,
            include_extras=True,
        )
        properties: dict[str, Any] = {}
        required_keys_attr = getattr(typed_dict_cls, "__required_keys__", set())
        optional_keys_attr = getattr(typed_dict_cls, "__optional_keys__", set())
        total = getattr(typed_dict_cls, "__total__", True)
        required: list[str] = []

        for key, value_type in hints.items():
            origin = get_origin(value_type)
            is_required = total
            if origin is Required:
                is_required = True
            elif origin is NotRequired:
                is_required = False
            elif key in required_keys_attr and key not in optional_keys_attr:
                is_required = True
            elif key in optional_keys_attr and key not in required_keys_attr:
                is_required = False

            if origin in (Required, NotRequired):
                value_type = get_args(value_type)[0]

            properties[key] = _python_type_to_json_schema(
                value_type,
                seen,
                globalns=globalns,
                localns=localns,
            )
            if is_required:
                required.append(key)

        return {"type": "object", "properties": properties, "required": required}
    finally:
        seen.remove(typed_dict_cls)


def _dataclass_to_json_schema(
    dataclass_cls: type[Any],
    seen: set[Any],
    globalns: Union[dict[str, Any], None] = None,
    localns: Union[dict[str, Any], None] = None,
) -> dict[str, Any]:
    """Convert a dataclass type to JSON Schema."""
    if dataclass_cls in seen:
        return {"type": "object"}

    seen.add(dataclass_cls)
    try:
        hints = get_type_hints(
            dataclass_cls,
            globalns=globalns,
            localns=localns,
            include_extras=True,
        )
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field_info in dataclass_fields(dataclass_cls):
            field_type = hints.get(field_info.name, Any)
            properties[field_info.name] = _python_type_to_json_schema(
                field_type,
                seen,
                globalns=globalns,
                localns=localns,
            )

            if (
                field_info.default is DATACLASS_MISSING
                and field_info.default_factory is DATACLASS_MISSING
            ):
                required.append(field_info.name)

        return {"type": "object", "properties": properties, "required": required}
    finally:
        seen.remove(dataclass_cls)


def _python_type_to_json_schema(
    py_type: Any,
    seen: Union[set[Any], None] = None,
    *,
    globalns: Union[dict[str, Any], None] = None,
    localns: Union[dict[str, Any], None] = None,
) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    seen = seen or set()
    if globalns is None and localns is None:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame else None
            if caller is not None:
                globalns = caller.f_globals
                localns = caller.f_locals
        finally:
            del frame
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Annotated:
        return (
            _python_type_to_json_schema(
                args[0],
                seen,
                globalns=globalns,
                localns=localns,
            )
            if args
            else {}
        )

    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(
                non_none[0],
                seen,
                globalns=globalns,
                localns=localns,
            )
        return {
            "anyOf": [
                _python_type_to_json_schema(
                    t,
                    seen,
                    globalns=globalns,
                    localns=localns,
                )
                for t in non_none
            ]
        }

    if origin in (list, set, frozenset):
        item_schema = (
            _python_type_to_json_schema(
                args[0],
                seen,
                globalns=globalns,
                localns=localns,
            )
            if args
            else {}
        )
        return {"type": "array", "items": item_schema}

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {
                "type": "array",
                "items": _python_type_to_json_schema(
                    args[0],
                    seen,
                    globalns=globalns,
                    localns=localns,
                ),
            }
        if args:
            prefix_items = [
                _python_type_to_json_schema(
                    arg,
                    seen,
                    globalns=globalns,
                    localns=localns,
                )
                for arg in args
            ]
            return {
                "type": "array",
                "prefixItems": prefix_items,
                "minItems": len(prefix_items),
                "maxItems": len(prefix_items),
            }
        return {"type": "array"}

    if origin is dict:
        schema: dict[str, Any] = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _python_type_to_json_schema(
                args[1],
                seen,
                globalns=globalns,
                localns=localns,
            )
        return schema

    if origin is Literal:
        literal_types = {
            literal_type
            for literal_type in (_literal_value_to_json_type(value) for value in args)
            if literal_type is not None
        }
        if len(literal_types) == 1:
            return {"type": literal_types.pop(), "enum": list(args)}
        return {
            "anyOf": [
                {
                    "const": value,
                    **(
                        {"type": literal_type}
                        if (literal_type := _literal_value_to_json_type(value)) is not None
                        else {}
                    ),
                }
                for value in args
            ]
        }

    if py_type is Any:
        return {}

    if is_typeddict(py_type):
        return _typed_dict_to_json_schema(
            py_type,
            seen,
            globalns=globalns,
            localns=localns,
        )

    if inspect.isclass(py_type) and is_dataclass(py_type):
        return _dataclass_to_json_schema(
            py_type,
            seen,
            globalns=globalns,
            localns=localns,
        )

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if py_type in type_map:
        return {"type": type_map[py_type]}

    return {"type": "string"}


# =============================================================================
# Tool Types
# =============================================================================


@dataclass
class Tool:
    description: str
    parameters: dict[str, Any]
    execute: Union[Callable[..., Any], Callable[..., Awaitable[Any]]]

    async def run(self, **kwargs: Any) -> Any:
        sig = inspect.signature(self.execute)
        bound_args = sig.bind_partial(**kwargs)

        for name, param in sig.parameters.items():
            if name not in bound_args.arguments and isinstance(param.default, Field):
                if param.default.default is not _MISSING:
                    kwargs[name] = param.default.default
        
        result = self.execute(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


@overload
def tool(func: Callable[..., Any]) -> Tool: ...

@overload
def tool(*, description: Union[str, None] = None) -> Callable[[Callable[..., Any]], Tool]: ...

@overload
def tool(
    *,
    description: str,
    parameters: dict[str, Any],
    execute: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
) -> Tool: ...


def tool(
    func: Union[Callable[..., Any], None] = None,
    *,
    description: Union[str, None] = None,
    parameters: Union[dict[str, Any], None] = None,
    execute: Union[Callable[..., Any], Callable[..., Awaitable[Any]], None] = None,
) -> Union[Tool, Callable[[Callable[..., Any]], Tool]]:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame else None
        definition_localns = dict(caller.f_locals) if caller is not None else None
    finally:
        del frame

    if execute is not None and parameters is not None and description is not None:
        return Tool(description=description, parameters=parameters, execute=execute)

    def _create_tool_from_function(fn: Callable[..., Any]) -> Tool:
        tool_description = description or ""
        if not tool_description and fn.__doc__:
            tool_description = fn.__doc__.strip().split("\n")[0].strip()
        if not tool_description:
            tool_description = f"Execute the {fn.__name__} function"

        try:
            hints = get_type_hints(
                fn,
                globalns=fn.__globals__,
                localns=definition_localns,
            )
        except Exception:
            hints = {}

        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "return"):
                continue

            param_type = hints.get(param_name, str)
            prop = _python_type_to_json_schema(
                param_type,
                globalns=fn.__globals__,
                localns=definition_localns,
            )

            if isinstance(param.default, Field):
                field_meta = param.default
                if field_meta.description:
                    prop["description"] = field_meta.description
                if field_meta.enum:
                    prop["enum"] = field_meta.enum
                if field_meta.min_value is not None:
                    prop["minimum"] = field_meta.min_value
                if field_meta.max_value is not None:
                    prop["maximum"] = field_meta.max_value
                if field_meta.default is _MISSING:
                    required.append(param_name)
            elif param.default is inspect.Parameter.empty:
                required.append(param_name)
            
            properties[param_name] = prop

        schema = {"type": "object", "properties": properties, "required": required}
        return Tool(description=tool_description, parameters=schema, execute=fn)

    if func is not None:
        return _create_tool_from_function(func)
    else:
        return _create_tool_from_function

ToolSet = dict[str, Tool]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool = False


@dataclass
class ToolCallPart:
    type: Literal["tool_call"] = "tool_call"
    tool_call: Union[ToolCall, None] = None


@dataclass
class ToolResultPart:
    type: Literal["tool_result"] = "tool_result"
    tool_result: Union[ToolResult, None] = None


@dataclass
class ReasoningPart:
    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Stop Conditions
# =============================================================================


@dataclass
class StepResult:
    text: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    reasoning_parts: list[ReasoningPart] = field(default_factory=list)
    finish_reason: Union[str, None] = None
    usage: Union["Usage", None] = None


TurnTerminationKind = Literal[
    "completed",
    "stop_condition",
    "step_limit",
    "hook_stop",
    "tool_terminated",
    "aborted",
    "failed",
]


@dataclass(frozen=True)
class TurnTermination:
    kind: TurnTerminationKind
    provider_finish_reason: Union[str, None] = None
    reason: Union[str, None] = None
    stop_condition: Union[str, None] = None
    tool_name: Union[str, None] = None
    final_step_number: int = 0
    has_text: bool = False
    has_tool_calls: bool = False
    last_tool_error: Union[str, None] = None
    error_type: Union[str, None] = None
    message: Union[str, None] = None


def build_turn_termination(
    kind: TurnTerminationKind,
    *,
    steps: list[StepResult],
    text: str = "",
    provider_finish_reason: Union[str, None] = None,
    reason: Union[str, None] = None,
    stop_condition: Union[str, None] = None,
    tool_name: Union[str, None] = None,
    final_step_number: Union[int, None] = None,
    has_tool_calls: Union[bool, None] = None,
    error_type: Union[str, None] = None,
    message: Union[str, None] = None,
) -> TurnTermination:
    last_tool_error = None
    for step in reversed(steps):
        for result in reversed(step.tool_results):
            if result.is_error:
                last_tool_error = str(result.result)
                break
        if last_tool_error is not None:
            break

    return TurnTermination(
        kind=kind,
        provider_finish_reason=provider_finish_reason,
        reason=reason,
        stop_condition=stop_condition,
        tool_name=tool_name,
        final_step_number=(
            len(steps) if final_step_number is None else final_step_number
        ),
        has_text=bool(text.strip()),
        has_tool_calls=(
            any(step.tool_calls for step in steps)
            if has_tool_calls is None
            else has_tool_calls
        ),
        last_tool_error=last_tool_error,
        error_type=error_type,
        message=message,
    )


@dataclass(frozen=True)
class StopDecision:
    stop: bool = True
    name: Union[str, None] = None
    reason: Union[str, None] = None


StopConditionResult = Union[bool, StopDecision]
StopCondition = Callable[
    [list[StepResult]],
    Union[StopConditionResult, Awaitable[StopConditionResult]],
]


def step_count_is(count: int) -> StopCondition:
    def condition(steps: list[StepResult]) -> bool:
        return len(steps) >= count
    condition._ai_query_termination_kind = "step_limit"  # type: ignore[attr-defined]
    condition._ai_query_stop_name = f"step_count_is({count})"  # type: ignore[attr-defined]
    condition._ai_query_stop_reason = f"Reached the step limit of {count}"  # type: ignore[attr-defined]
    return condition


def has_tool_call(tool_name: str) -> StopCondition:
    def condition(steps: list[StepResult]) -> bool:
        if not steps:
            return False
        last_step = steps[-1]
        return any(tc.name == tool_name for tc in last_step.tool_calls)
    condition._ai_query_termination_kind = "stop_condition"  # type: ignore[attr-defined]
    condition._ai_query_stop_name = f"has_tool_call({tool_name!r})"  # type: ignore[attr-defined]
    condition._ai_query_stop_reason = f"Matched tool call {tool_name!r}"  # type: ignore[attr-defined]
    return condition


# =============================================================================
# Step Callbacks
# =============================================================================


@dataclass
class StepStartEvent:
    step_number: int
    messages: list["Message"]
    tools: Union[ToolSet, None]


@dataclass
class StepControl:
    inject_messages: list["Message"] = field(default_factory=list)
    stop: bool = False
    stop_reason: Union[str, None] = None


@dataclass
class StepFinishEvent:
    step_number: int
    step: StepResult
    text: str
    usage: Union[Usage, None]
    steps: list[StepResult]


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    initial_delay: float = 0.5
    max_delay: float = 8.0
    backoff: float = 2.0
    jitter: bool = True
    retry_on: Callable[[Exception], bool] | None = None


@dataclass
class RetryEvent:
    step_number: int
    attempt: int
    max_attempts: int
    delay: float
    error: str
    exception: Exception


OnStepStart = Callable[[StepStartEvent], Union[StepControl, None, Awaitable[Union[StepControl, None]]]]
OnStepFinish = Callable[[StepFinishEvent], Union[None, Awaitable[None]]]
OnRetry = Callable[[RetryEvent], Union[None, Awaitable[None]]]


@dataclass
class BeforeToolCallEvent:
    step_number: int
    tool_call: ToolCall
    messages: list["Message"]


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallEvent:
    step_number: int
    tool_call: ToolCall
    tool_result: ToolResult
    messages: list["Message"]


@dataclass
class AfterToolCallResult:
    result: Any | None = None
    is_error: bool | None = None
    terminate: bool | None = None
    terminate_reason: str | None = None


OnBeforeToolCall = Callable[
    [BeforeToolCallEvent],
    Union[
        BeforeToolCallResult,
        None,
        Awaitable[Union[BeforeToolCallResult, None]],
    ],
]
OnAfterToolCall = Callable[
    [AfterToolCallEvent],
    Union[
        AfterToolCallResult,
        None,
        Awaitable[Union[AfterToolCallResult, None]],
    ],
]


@dataclass
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImagePart:
    type: Literal["image"] = "image"
    image: Union[str, bytes] = b""
    media_type: Union[str, None] = None


@dataclass
class FilePart:
    type: Literal["file"] = "file"
    data: Union[str, bytes] = b""
    media_type: str = ""
    filename: Union[str, None] = None


ToolOutputPart = Union[TextPart, ImagePart, FilePart]


class UnsupportedToolOutputError(ValueError):
    """Raised when a provider endpoint cannot represent a rich tool output."""


@dataclass(repr=False)
class ToolOutput:
    """Explicit rich content returned by a tool for the model's next turn."""

    content: list[ToolOutputPart]

    def __post_init__(self) -> None:
        unsupported = [
            type(part).__name__
            for part in self.content
            if not isinstance(part, (TextPart, ImagePart, FilePart))
        ]
        if unsupported:
            raise TypeError(
                "ToolOutput content only supports TextPart, ImagePart, and "
                f"FilePart; got {', '.join(unsupported)}"
            )

    def __repr__(self) -> str:
        counts = {
            "text": sum(isinstance(part, TextPart) for part in self.content),
            "image": sum(isinstance(part, ImagePart) for part in self.content),
            "file": sum(isinstance(part, FilePart) for part in self.content),
        }
        return (
            "ToolOutput("
            f"text_parts={counts['text']}, image_parts={counts['image']}, "
            f"file_parts={counts['file']}, payloads=<redacted>)"
        )


_TOOL_OUTPUT_MARKER = "$ai_query.tool_output"
_TOOL_OUTPUT_BYTES_MARKER = "$ai_query.bytes"


def _tool_output_value_to_dict(value: Any) -> Any:
    if not isinstance(value, ToolOutput):
        return value

    content: list[dict[str, Any]] = []
    for part in value.content:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
            continue

        raw_value = part.image if isinstance(part, ImagePart) else part.data
        if isinstance(raw_value, bytes):
            raw_value = {
                _TOOL_OUTPUT_BYTES_MARKER: base64.b64encode(raw_value).decode("ascii")
            }
        part_dict: dict[str, Any] = {
            "type": part.type,
            "image" if isinstance(part, ImagePart) else "data": raw_value,
        }
        if part.media_type:
            part_dict["media_type"] = part.media_type
        if isinstance(part, FilePart) and part.filename:
            part_dict["filename"] = part.filename
        content.append(part_dict)

    return {_TOOL_OUTPUT_MARKER: {"content": content}}


def _tool_output_value_from_dict(value: Any) -> Any:
    if not isinstance(value, dict) or set(value) != {_TOOL_OUTPUT_MARKER}:
        return value
    payload = value[_TOOL_OUTPUT_MARKER]
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
        raise ValueError("Invalid serialized ToolOutput")

    content: list[ToolOutputPart] = []
    for item in payload["content"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid serialized ToolOutput part")
        part_type = item.get("type")
        if part_type == "text":
            content.append(TextPart(text=item.get("text", "")))
            continue
        if part_type not in {"image", "file"}:
            raise ValueError(f"Invalid serialized ToolOutput part type: {part_type!r}")

        key = "image" if part_type == "image" else "data"
        raw_value = item.get(key, b"")
        if isinstance(raw_value, dict) and set(raw_value) == {_TOOL_OUTPUT_BYTES_MARKER}:
            encoded = raw_value[_TOOL_OUTPUT_BYTES_MARKER]
            if not isinstance(encoded, str):
                raise ValueError("Invalid encoded bytes in ToolOutput")
            try:
                raw_value = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("Invalid encoded bytes in ToolOutput") from exc

        if part_type == "image":
            content.append(
                ImagePart(image=raw_value, media_type=item.get("media_type"))
            )
        else:
            content.append(
                FilePart(
                    data=raw_value,
                    media_type=item.get("media_type", ""),
                    filename=item.get("filename"),
                )
            )
    return ToolOutput(content=content)


ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
]


@dataclass
class Message:
    role: Role
    content: Union[str, list[ContentPart]]

    @staticmethod
    def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "metadata": tool_call.metadata,
        }

    @staticmethod
    def _tool_result_to_dict(tool_result: ToolResult) -> dict[str, Any]:
        return {
            "tool_call_id": tool_result.tool_call_id,
            "tool_name": tool_result.tool_name,
            "result": _tool_output_value_to_dict(tool_result.result),
            "is_error": tool_result.is_error,
        }

    @classmethod
    def _part_to_dict(cls, part: Any) -> dict[str, Any]:
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "tool_call" and isinstance(part.get("tool_call"), ToolCall):
                return {
                    "type": "tool_call",
                    "tool_call": cls._tool_call_to_dict(part["tool_call"]),
                }
            if part_type == "tool_result" and isinstance(
                part.get("tool_result"), ToolResult
            ):
                return {
                    "type": "tool_result",
                    "tool_result": cls._tool_result_to_dict(part["tool_result"]),
                }
            return part

        part_dict: dict[str, Any] = {"type": part.type}
        if isinstance(part, TextPart):
            part_dict["text"] = part.text
        elif isinstance(part, ImagePart):
            part_dict["image"] = part.image
            if part.media_type:
                part_dict["media_type"] = part.media_type
        elif isinstance(part, FilePart):
            part_dict["data"] = part.data
            part_dict["media_type"] = part.media_type
            if part.filename:
                part_dict["filename"] = part.filename
        elif isinstance(part, ToolCallPart):
            if part.tool_call:
                part_dict["tool_call"] = cls._tool_call_to_dict(part.tool_call)
        elif isinstance(part, ToolResultPart):
            if part.tool_result:
                part_dict["tool_result"] = cls._tool_result_to_dict(part.tool_result)
        elif isinstance(part, ReasoningPart):
            part_dict["text"] = part.text
            if part.data:
                part_dict["data"] = part.data

        return part_dict

    @staticmethod
    def _part_from_dict(part: Any) -> ContentPart | dict[str, Any]:
        if not isinstance(part, dict):
            return part

        part_type = part.get("type")
        if part_type == "text":
            return TextPart(text=part.get("text", ""))
        if part_type == "image":
            return ImagePart(
                image=part.get("image", b""),
                media_type=part.get("media_type"),
            )
        if part_type == "file":
            return FilePart(
                data=part.get("data", b""),
                media_type=part.get("media_type", ""),
                filename=part.get("filename"),
            )
        if part_type == "tool_call":
            tool_call = part.get("tool_call")
            if isinstance(tool_call, dict):
                tool_call = ToolCall(
                    id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=tool_call.get("arguments", {}),
                    metadata=tool_call.get("metadata", {}),
                )
            return ToolCallPart(tool_call=tool_call)
        if part_type == "tool_result":
            tool_result = part.get("tool_result")
            if isinstance(tool_result, dict):
                tool_result = ToolResult(
                    tool_call_id=tool_result["tool_call_id"],
                    tool_name=tool_result["tool_name"],
                    result=_tool_output_value_from_dict(tool_result.get("result")),
                    is_error=tool_result.get("is_error", False),
                )
            return ToolResultPart(tool_result=tool_result)
        if part_type == "reasoning":
            data = part.get("data")
            if not isinstance(data, dict):
                data = {}
            return ReasoningPart(
                text=part.get("text", ""),
                data=data,
            )

        return part

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        content = data["content"]
        if isinstance(content, str):
            return cls(role=data["role"], content=content)

        return cls(
            role=data["role"],
            content=[cls._part_from_dict(part) for part in content],
        )

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        content_list = [self._part_to_dict(part) for part in self.content]

        return {"role": self.role, "content": content_list}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cache_miss_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerateTextResult:
    text: str
    steps: list[StepResult] = field(default_factory=list)
    reasoning_parts: list[ReasoningPart] = field(default_factory=list)
    finish_reason: Union[str, None] = None
    usage: Union[Usage, None] = None
    response: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    termination: Union[TurnTermination, None] = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    @property
    def tool_results(self) -> list[ToolResult]:
        results = []
        for step in self.steps:
            results.extend(step.tool_results)
        return results


@dataclass
class AgentEvent:
    id: int
    type: str
    data: dict[str, Any]
    created_at: float


@dataclass
class StreamTextResult:
    text: str
    finish_reason: Union[str, None] = None
    usage: Union[Usage, None] = None


ReasoningEventKind = Literal["summary", "delta", "signature", "state"]
StreamReasoningEventType = Literal[
    "reasoning.summary",
    "reasoning.delta",
    "reasoning.signature",
    "reasoning.state",
]


@dataclass
class ReasoningEvent:
    kind: ReasoningEventKind
    provider: str
    text: Union[str, None] = None
    data: dict[str, Any] = field(default_factory=dict)


OnReasoningEvent = Callable[[ReasoningEvent], Union[None, Awaitable[None]]]


ToolCallStreamEventKind = Literal["start", "delta"]


@dataclass
class ToolCallStreamEvent:
    kind: ToolCallStreamEventKind
    index: int
    tool_call_id: Union[str, None] = None
    name: Union[str, None] = None
    name_delta: Union[str, None] = None
    arguments_delta: Union[str, None] = None


@dataclass
class StreamChunk:
    text: str = ""
    is_final: bool = False
    usage: Union[Usage, None] = None
    finish_reason: Union[str, None] = None
    tool_calls: Union[list[ToolCall], None] = None
    reasoning_events: Union[list[ReasoningEvent], None] = None
    tool_call_events: Union[list[ToolCallStreamEvent], None] = None


@dataclass
class TextDeltaEvent:
    type: Literal["text.delta"]
    text: str
    step_number: int


@dataclass
class StreamReasoningEvent:
    type: StreamReasoningEventType
    event: ReasoningEvent
    step_number: int


@dataclass
class ToolCallStartedEvent:
    type: Literal["tool_call.started"]
    step_number: int
    index: int
    tool_call_id: Union[str, None]
    name: Union[str, None]


@dataclass
class ToolCallDeltaEvent:
    type: Literal["tool_call.delta"]
    step_number: int
    index: int
    tool_call_id: Union[str, None]
    name_delta: Union[str, None] = None
    arguments_delta: Union[str, None] = None


@dataclass
class ToolCallReadyEvent:
    type: Literal["tool_call.ready"]
    step_number: int
    index: int
    tool_call: ToolCall


@dataclass
class ToolExecutionStartedEvent:
    type: Literal["tool_execution.started"]
    step_number: int
    index: int
    tool_call: ToolCall


@dataclass
class ToolExecutionFinishedEvent:
    type: Literal["tool_execution.finished"]
    step_number: int
    index: int
    tool_call: ToolCall
    tool_result: Union[ToolResult, None]
    duration: float
    error: Union[str, None] = None
    aborted: bool = False


@dataclass
class ToolResultEvent:
    type: Literal["tool_result"]
    step_number: int
    index: int
    tool_call: ToolCall
    tool_result: ToolResult


@dataclass
class StreamStepStartedEvent:
    type: Literal["step.started"]
    step_number: int
    messages: list[Message]
    tools: Union[ToolSet, None]


@dataclass
class StreamStepFinishedEvent:
    type: Literal["step.finished"]
    step_number: int
    step: StepResult
    text: str
    usage: Union[Usage, None]
    steps: list[StepResult]


@dataclass
class StreamFinishedEvent:
    type: Literal["stream.finished"]
    text: str
    finish_reason: Union[str, None]
    usage: Union[Usage, None]
    steps: list[StepResult]
    termination: Union[TurnTermination, None] = None


TextStreamEvent = Union[
    TextDeltaEvent,
    StreamReasoningEvent,
    ToolCallStartedEvent,
    ToolCallDeltaEvent,
    ToolCallReadyEvent,
    ToolExecutionStartedEvent,
    ToolExecutionFinishedEvent,
    ToolResultEvent,
    StreamStepStartedEvent,
    StreamStepFinishedEvent,
    StreamFinishedEvent,
]


class TextStreamResult:
    def __init__(
        self,
        stream: AsyncIterator[TextStreamEvent],
        steps: Union[list[StepResult], None] = None,
    ) -> None:
        self._stream = stream
        self._events: list[TextStreamEvent] = []
        self._chunks: list[str] = []
        self._usage: Union[Usage, None] = None
        self._finish_reason: Union[str, None] = None
        self._termination: Union[TurnTermination, None] = None
        self._steps: list[StepResult] = steps if steps is not None else []
        self._done = False
        self._error: BaseException | None = None
        self._pull_lock = asyncio.Lock()

    def _record_event(self, event: TextStreamEvent) -> None:
        self._events.append(event)
        if isinstance(event, TextDeltaEvent):
            self._chunks.append(event.text)
        elif isinstance(event, StreamFinishedEvent):
            self._usage = event.usage
            self._finish_reason = event.finish_reason
            self._termination = event.termination

    async def _iterate_events(self) -> AsyncIterator[TextStreamEvent]:
        index = 0
        while True:
            if index < len(self._events):
                event = self._events[index]
                index += 1
                yield event
                continue

            if self._done:
                if self._error is not None:
                    raise self._error
                return

            async with self._pull_lock:
                if index < len(self._events) or self._done:
                    continue
                try:
                    event = await anext(self._stream)
                except StopAsyncIteration:
                    self._done = True
                except BaseException as exc:
                    self._error = exc
                    self._done = True
                else:
                    self._record_event(event)

    async def _consume_stream(self) -> AsyncIterator[str]:
        async for event in self._iterate_events():
            if isinstance(event, TextDeltaEvent):
                yield event.text

    @property
    def event_stream(self) -> AsyncIterator[TextStreamEvent]:
        return self._iterate_events()

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._consume_stream()

    async def _wait_for_completion(self) -> None:
        if not self._done:
            async for _ in self._iterate_events():
                pass

    @property
    async def text(self) -> str:
        await self._wait_for_completion()
        return "".join(self._chunks)

    @property
    async def usage(self) -> Union[Usage, None]:
        await self._wait_for_completion()
        return self._usage

    @property
    async def finish_reason(self) -> Union[str, None]:
        await self._wait_for_completion()
        return self._finish_reason

    @property
    async def termination(self) -> Union[TurnTermination, None]:
        await self._wait_for_completion()
        return self._termination

    @property
    async def steps(self) -> list[StepResult]:
        await self._wait_for_completion()
        return self._steps

    def __aiter__(self) -> AsyncIterator[str]:
        return self.text_stream

ProviderOptions = dict[str, dict[str, Any]]

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class ReasoningConfig(TypedDict, total=False):
    effort: ReasoningEffort
    budget: int


# =============================================================================
# Embedding Types
# =============================================================================


@dataclass
class EmbeddingUsage:
    tokens: int = 0


@dataclass
class EmbedResult:
    value: str
    embedding: list[float]
    usage: EmbeddingUsage
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedManyResult:
    values: list[str]
    embeddings: list[list[float]]
    usage: EmbeddingUsage
    provider_metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abort / Cancellation
# =============================================================================


class AbortError(Exception):
    def __init__(self, reason: Union[str, None] = None):
        self.reason = reason
        self.termination: Union[TurnTermination, None] = None
        super().__init__(reason or "Operation aborted")


class AbortSignal:
    def __init__(self) -> None:
        self._aborted = False
        self._reason: Union[str, None] = None
        self._listeners: list[Callable[[], None]] = []
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> Union[str, None]:
        return self._reason

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise AbortError(self._reason)

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)
        if self._aborted:
            callback()

    def _abort(self, reason: Union[str, None] = None) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
        self._event.set()
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                pass

    async def wait(self) -> None:
        await self._event.wait()

    @staticmethod
    def timeout(seconds: float) -> "AbortSignal":
        controller = AbortController()
        async def _timeout() -> None:
            await asyncio.sleep(seconds)
            controller.abort(f"Timeout after {seconds}s")
        asyncio.create_task(_timeout())
        return controller.signal


class AbortController:
    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self, reason: Union[str, None] = None) -> None:
        self._signal._abort(reason)
