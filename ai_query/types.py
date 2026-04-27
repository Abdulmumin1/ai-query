"""Core types for ai-query library."""

from __future__ import annotations

import asyncio
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
    TYPE_CHECKING,
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


# =============================================================================
# Stop Conditions
# =============================================================================


@dataclass
class StepResult:
    text: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    finish_reason: Union[str, None] = None


StopCondition = Callable[[list[StepResult]], Union[bool, Awaitable[bool]]]


def step_count_is(count: int) -> StopCondition:
    def condition(steps: list[StepResult]) -> bool:
        return len(steps) >= count
    return condition


def has_tool_call(tool_name: str) -> StopCondition:
    def condition(steps: list[StepResult]) -> bool:
        if not steps:
            return False
        last_step = steps[-1]
        return any(tc.name == tool_name for tc in last_step.tool_calls)
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
class StepFinishEvent:
    step_number: int
    step: StepResult
    text: str
    usage: Usage
    steps: list[StepResult]


OnStepStart = Callable[[StepStartEvent], Union[None, Awaitable[None]]]
OnStepFinish = Callable[[StepFinishEvent], Union[None, Awaitable[None]]]


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


ContentPart = Union[TextPart, ImagePart, FilePart, ToolCallPart, ToolResultPart]


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
            "result": tool_result.result,
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
        elif isinstance(part, ToolCallPart):
            if part.tool_call:
                part_dict["tool_call"] = cls._tool_call_to_dict(part.tool_call)
        elif isinstance(part, ToolResultPart):
            if part.tool_result:
                part_dict["tool_result"] = cls._tool_result_to_dict(part.tool_result)

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
                    result=tool_result.get("result"),
                    is_error=tool_result.get("is_error", False),
                )
            return ToolResultPart(tool_result=tool_result)

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
    total_tokens: int = 0


@dataclass
class GenerateTextResult:
    text: str
    steps: list[StepResult] = field(default_factory=list)
    finish_reason: Union[str, None] = None
    usage: Union[Usage, None] = None
    response: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

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


@dataclass
class ReasoningEvent:
    kind: ReasoningEventKind
    provider: str
    text: Union[str, None] = None
    data: dict[str, Any] = field(default_factory=dict)


OnReasoningEvent = Callable[[ReasoningEvent], Union[None, Awaitable[None]]]


@dataclass
class StreamChunk:
    text: str = ""
    is_final: bool = False
    usage: Union[Usage, None] = None
    finish_reason: Union[str, None] = None
    tool_calls: Union[list[ToolCall], None] = None
    reasoning_events: Union[list[ReasoningEvent], None] = None


class TextStreamResult:
    def __init__(self, stream: AsyncIterator[StreamChunk], steps: Union[list[StepResult], None] = None) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._usage: Union[Usage, None] = None
        self._finish_reason: Union[str, None] = None
        self._steps: list[StepResult] = steps if steps is not None else []
        self._done = False
        self._done_event = asyncio.Event()
        self._consumed = False

    async def _consume_stream(self) -> AsyncIterator[str]:
        if self._consumed:
            for chunk in self._chunks:
                yield chunk
            return

        self._consumed = True
        async for chunk in self._stream:
            if chunk.is_final:
                self._usage = chunk.usage
                self._finish_reason = chunk.finish_reason
            elif chunk.text:
                self._chunks.append(chunk.text)
                yield chunk.text

        self._done = True
        self._done_event.set()

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._consume_stream()

    async def _wait_for_completion(self) -> None:
        if not self._done:
            if not self._consumed:
                async for _ in self._consume_stream():
                    pass
            else:
                await self._done_event.wait()

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
