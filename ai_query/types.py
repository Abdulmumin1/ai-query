"""Core types for ai-query library."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Union,
    AsyncIterator,
    Callable,
    Awaitable,
    TYPE_CHECKING,
    get_type_hints,
    get_origin,
    get_args,
    overload,
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


def _python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(t) for t in non_none]}

    if origin is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    if origin is Literal:
        return {"type": "string", "enum": list(args)}

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
    if execute is not None and parameters is not None and description is not None:
        return Tool(description=description, parameters=parameters, execute=execute)

    def _create_tool_from_function(fn: Callable[..., Any]) -> Tool:
        tool_description = description or ""
        if not tool_description and fn.__doc__:
            tool_description = fn.__doc__.strip().split("\n")[0].strip()
        if not tool_description:
            tool_description = f"Execute the {fn.__name__} function"

        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "return"):
                continue

            param_type = hints.get(param_name, str)
            prop = _python_type_to_json_schema(param_type)

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


ContentPart = Union[TextPart, ImagePart, FilePart]


@dataclass
class Message:
    role: Role
    content: Union[str, list[ContentPart]]

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        
        content_list = []
        for p in self.content:
            if isinstance(p, dict):
                content_list.append(p)
            else:
                part_dict: dict[str, Any] = {"type": p.type}
                if isinstance(p, TextPart):
                    part_dict["text"] = p.text
                elif isinstance(p, ImagePart):
                    part_dict["image"] = p.image
                    if p.media_type:
                        part_dict["media_type"] = p.media_type
                elif isinstance(p, FilePart):
                    part_dict["data"] = p.data
                    part_dict["media_type"] = p.media_type
                content_list.append(part_dict)
        
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


@dataclass
class StreamChunk:
    text: str = ""
    is_final: bool = False
    usage: Union[Usage, None] = None
    finish_reason: Union[str, None] = None
    tool_calls: Union[list[ToolCall], None] = None


class TextStreamResult:
    def __init__(self, stream: AsyncIterator[StreamChunk], steps: Union[list[StepResult], None] = None) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._usage: Union[Usage, None] = None
        self._finish_reason: Union[str, None] = None
        self._steps: list[StepResult] = steps or []
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
