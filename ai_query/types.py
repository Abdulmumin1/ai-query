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
    """Define metadata for a tool parameter.

    Similar to Pydantic's Field, this allows you to specify descriptions
    and defaults for tool parameters in a clean, declarative way.

    Example:
        @tool(description="Search the web")
        async def search(
            query: str = Field(description="The search query"),
            limit: int = Field(description="Max results", default=10)
        ) -> str:
            ...
    """

    def __init__(
        self,
        description: str = "",
        default: Any = _MISSING,
        enum: list[Any] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
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

    # Handle Optional[X] (Union[X, None])
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        # Union of multiple types - use anyOf
        return {"anyOf": [_python_type_to_json_schema(t) for t in non_none]}

    # Handle List[X]
    if origin is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    # Handle Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Handle Literal["a", "b", "c"]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Basic type mappings
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

    # Default to string for unknown types
    return {"type": "string"}


# =============================================================================
# Tool Types
# =============================================================================


@dataclass
class Tool:
    """A tool that can be called by the AI model.

    Tools allow the AI to perform actions like fetching data, calling APIs,
    or executing code. When the AI decides to use a tool, the execute function
    is called with the parsed arguments.

    Example using decorator (recommended):
        >>> @tool(description="Get weather for a location")
        ... async def get_weather(
        ...     location: str = Field(description="City name")
        ... ) -> str:
        ...     return f"Weather in {location}: sunny"

    Example using factory function (legacy):
        >>> weather_tool = tool(
        ...     description="Get weather for a location",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {"type": "string", "description": "City name"}
        ...         },
        ...         "required": ["location"]
        ...     },
        ...     execute=lambda location: {"temp": 72, "condition": "sunny"}
        ... )
    """

    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., Any] | Callable[..., Awaitable[Any]]

    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool, handling both sync and async functions."""
        # Replace Field objects with their actual default values if they weren't provided
        sig = inspect.signature(self.execute)
        bound_args = sig.bind_partial(**kwargs)

        # We need to check all parameters that have a Field default
        for name, param in sig.parameters.items():
            if name not in bound_args.arguments and isinstance(param.default, Field):
                if param.default.default is not _MISSING:
                    kwargs[name] = param.default.default
                else:
                    # If it's a Field with no default, but it's not in kwargs,
                    # Python will raise a TypeError anyway if we don't provide it,
                    # or it will use the Field object itself as the default.
                    # We should let the normal Python call handle the missing arg error.
                    pass

        result = self.execute(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


@overload
def tool(
    func: Callable[..., Any],
) -> Tool: ...


@overload
def tool(
    *,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


@overload
def tool(
    *,
    description: str,
    parameters: dict[str, Any],
    execute: Callable[..., Any] | Callable[..., Awaitable[Any]],
) -> Tool: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execute: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Create a tool from a function using decorator syntax or factory function.

    Can be used in three ways:

    1. Simple decorator (infers everything from function):
        >>> @tool
        ... async def search(query: str = Field(description="Search query")) -> str:
        ...     '''Search the web for information.'''
        ...     return "results"

    2. Decorator with description:
        >>> @tool(description="Search the web")
        ... async def search(query: str = Field(description="Search query")) -> str:
        ...     return "results"

    3. Factory function (legacy, explicit parameters):
        >>> search_tool = tool(
        ...     description="Search the web",
        ...     parameters={"type": "object", "properties": {...}},
        ...     execute=search_fn
        ... )

    Args:
        func: The function to wrap (when used as @tool without parentheses).
        description: Tool description. If not provided, uses the function's docstring.
        parameters: JSON Schema for parameters (legacy). If not provided, inferred from type hints.
        execute: Function to execute (legacy). If not provided, uses the decorated function.

    Returns:
        A Tool instance.
    """
    # Legacy factory function mode: tool(description=..., parameters=..., execute=...)
    if execute is not None and parameters is not None and description is not None:
        return Tool(description=description, parameters=parameters, execute=execute)

    def _create_tool_from_function(fn: Callable[..., Any]) -> Tool:
        """Build a Tool by inspecting the function signature and type hints."""
        # Get description from argument or docstring
        tool_description = description or ""
        if not tool_description and fn.__doc__:
            # Use first line of docstring
            tool_description = fn.__doc__.strip().split("\n")[0].strip()
        if not tool_description:
            tool_description = f"Execute the {fn.__name__} function"

        # Get type hints
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        sig = inspect.signature(fn)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type from hints, default to string
            param_type = hints.get(param_name, str)
            # Skip return type annotation
            if param_name == "return":
                continue

            prop = _python_type_to_json_schema(param_type)

            # Check if default is a Field
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
                # Required if no default provided
                if field_meta.default is _MISSING:
                    required.append(param_name)
            elif param.default is inspect.Parameter.empty:
                # No default = required
                required.append(param_name)

            properties[param_name] = prop

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return Tool(description=tool_description, parameters=schema, execute=fn)

    # Decorator mode: @tool or @tool(description="...")
    if func is not None:
        # Called as @tool without parentheses
        return _create_tool_from_function(func)
    else:
        # Called as @tool(...) with arguments
        return _create_tool_from_function


# Type alias for a collection of tools
ToolSet = dict[str, Tool]


@dataclass
class ToolCall:
    """A tool call made by the AI model."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Provider-specific metadata (e.g., Google's thought_signature)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool = False


@dataclass
class ToolCallPart:
    """A tool call content part (for assistant messages)."""

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall | None = None


@dataclass
class ToolResultPart:
    """A tool result content part (for tool messages)."""

    type: Literal["tool_result"] = "tool_result"
    tool_result: ToolResult | None = None


# =============================================================================
# Stop Conditions
# =============================================================================


@dataclass
class StepResult:
    """Result from a single step in the generation loop."""

    text: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    finish_reason: str | None = None


# Type for stop condition functions
StopCondition = Callable[[list[StepResult]], Union[bool, Awaitable[bool]]]


def step_count_is(count: int) -> StopCondition:
    """Stop when the step count reaches the specified number.

    Args:
        count: Number of steps after which to stop.

    Returns:
        A stop condition function.

    Example:
        >>> result = await generate_text(
        ...     model=openai("gpt-4"),
        ...     tools={"search": search_tool},
        ...     stop_when=step_count_is(5),  # Max 5 iterations
        ...     prompt="Research this topic"
        ... )
    """
    def condition(steps: list[StepResult]) -> bool:
        return len(steps) >= count

    return condition


def has_tool_call(tool_name: str) -> StopCondition:
    """Stop when a specific tool is called.

    Args:
        tool_name: Name of the tool that triggers the stop.

    Returns:
        A stop condition function.

    Example:
        >>> result = await generate_text(
        ...     model=openai("gpt-4"),
        ...     tools={"search": search_tool, "final_answer": answer_tool},
        ...     stop_when=has_tool_call("final_answer"),
        ...     prompt="Research and answer"
        ... )
    """
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
    """Event passed to on_step_start callback.

    Provides context about the step that's about to run, allowing inspection
    or modification of messages before the model is called.

    Attributes:
        step_number: The 1-indexed step number (1 for first step, 2 for second, etc.).
        messages: The current conversation history that will be sent to the model.
            This list can be modified to alter what the model sees.
        tools: The available tools for this step, or None if no tools.
    """

    step_number: int
    messages: list["Message"]
    tools: "ToolSet | None"


@dataclass
class StepFinishEvent:
    """Event passed to on_step_finish callback.

    Provides information about the completed step and accumulated state.

    Attributes:
        step_number: The 1-indexed step number that just completed.
        step: The StepResult for this specific step (text, tool_calls, tool_results).
        text: Accumulated text from all steps so far.
        usage: Accumulated token usage from all steps so far.
        steps: List of all StepResults from steps completed so far.
    """

    step_number: int
    step: StepResult
    text: str
    usage: "Usage"
    steps: list[StepResult]


# Type aliases for step callback functions
OnStepStart = Callable[[StepStartEvent], Union[None, Awaitable[None]]]
OnStepFinish = Callable[[StepFinishEvent], Union[None, Awaitable[None]]]


@dataclass
class TextPart:
    """Text content part."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImagePart:
    """Image content part."""

    type: Literal["image"] = "image"
    image: str | bytes = b""  # base64 string, bytes, or URL
    media_type: str | None = None


@dataclass
class FilePart:
    """File content part."""

    type: Literal["file"] = "file"
    data: str | bytes = b""
    media_type: str = ""


ContentPart = Union[TextPart, ImagePart, FilePart]


@dataclass
class Message:
    """A message in the conversation."""

    role: Role
    content: str | list[ContentPart]

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        content_list = []
        for p in self.content:
            if isinstance(p, dict):
                content_list.append(p)
            else:
                # Handle ContentPart objects
                part_dict = {"type": p.type}
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

        return {
            "role": self.role,
            "content": content_list,
        }


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerateTextResult:
    """Result from generate_text call."""

    text: str
    steps: list[StepResult] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    response: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls made across all steps."""
        calls = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    @property
    def tool_results(self) -> list[ToolResult]:
        """All tool results from all steps."""
        results = []
        for step in self.steps:
            results.extend(step.tool_results)
        return results


@dataclass
class StreamTextResult:
    """Final result from stream_text after streaming completes."""

    text: str
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass
class StreamChunk:
    """A chunk from streaming - either text content or final metadata."""

    text: str = ""
    is_final: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


class TextStreamResult:
    """Result object from stream_text with both stream and metadata.

    Similar to ai-sdk's streamText result, provides access to:
    - text_stream: AsyncIterator yielding text chunks
    - text: Awaitable that resolves to full text after streaming
    - usage: Awaitable that resolves to Usage after streaming
    - finish_reason: Awaitable that resolves to finish reason after streaming
    - steps: Awaitable that resolves to list of StepResult after streaming

    Example:
        >>> result = stream_text(model=google("gemini-2.0-flash"), prompt="Hi")
        >>> async for chunk in result.text_stream:
        ...     print(chunk, end="")
        >>> print(await result.usage)  # Usage after stream completes
        >>> print(await result.text)   # Full accumulated text
        >>> print(await result.steps)  # All steps with tool calls

    Or iterate directly:
        >>> async for chunk in stream_text(model=google("gemini-2.0-flash"), prompt="Hi"):
        ...     print(chunk, end="")
    """

    def __init__(self, stream: AsyncIterator[StreamChunk], steps: list[StepResult] | None = None) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._usage: Usage | None = None
        self._finish_reason: str | None = None
        self._steps: list[StepResult] = steps or []
        self._done = False
        self._done_event = asyncio.Event()
        self._consumed = False

    async def _consume_stream(self) -> AsyncIterator[str]:
        """Consume the stream, collecting chunks and yielding text."""
        if self._consumed:
            # If already consumed, just yield from collected chunks
            for chunk in self._chunks:
                yield chunk
            return

        self._consumed = True
        async for chunk in self._stream:
            if chunk.is_final:
                # Final chunk contains metadata
                self._usage = chunk.usage
                self._finish_reason = chunk.finish_reason
            elif chunk.text:
                self._chunks.append(chunk.text)
                yield chunk.text

        self._done = True
        self._done_event.set()

    @property
    def text_stream(self) -> AsyncIterator[str]:
        """Async iterator yielding text chunks as they arrive."""
        return self._consume_stream()

    async def _wait_for_completion(self) -> None:
        """Wait for the stream to complete."""
        if not self._done:
            # If not consumed yet, we need to consume it
            if not self._consumed:
                async for _ in self._consume_stream():
                    pass
            else:
                await self._done_event.wait()

    @property
    async def text(self) -> str:
        """Get the full accumulated text after streaming completes."""
        await self._wait_for_completion()
        return "".join(self._chunks)

    @property
    async def usage(self) -> Usage | None:
        """Get usage statistics after streaming completes."""
        await self._wait_for_completion()
        return self._usage

    @property
    async def finish_reason(self) -> str | None:
        """Get the finish reason after streaming completes."""
        await self._wait_for_completion()
        return self._finish_reason

    @property
    async def steps(self) -> list[StepResult]:
        """Get all steps including tool calls after streaming completes."""
        await self._wait_for_completion()
        return self._steps

    def __aiter__(self) -> AsyncIterator[str]:
        """Allow direct iteration: async for chunk in stream_text(...)"""
        return self.text_stream


# Provider options type - allows provider-specific configuration
# Example: {"google": {"safety_settings": {...}}, "anthropic": {"top_k": 10}}
ProviderOptions = dict[str, dict[str, Any]]
