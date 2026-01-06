"""Core types for ai-query library."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal, Union, AsyncIterator, Callable, Awaitable, TYPE_CHECKING

# Message types
Role = Literal["system", "user", "assistant", "tool"]


# =============================================================================
# Tool Types
# =============================================================================


@dataclass
class Tool:
    """A tool that can be called by the AI model.

    Tools allow the AI to perform actions like fetching data, calling APIs,
    or executing code. When the AI decides to use a tool, the execute function
    is called with the parsed arguments.

    Example:
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
        result = self.execute(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def tool(
    *,
    description: str,
    parameters: dict[str, Any],
    execute: Callable[..., Any] | Callable[..., Awaitable[Any]],
) -> Tool:
    """Create a tool definition.

    Args:
        description: What the tool does (sent to the AI model).
        parameters: JSON Schema defining the tool's input parameters.
        execute: Function to run when the tool is called. Can be sync or async.
            Arguments are passed as keyword arguments matching the schema.

    Returns:
        A Tool instance.

    Example:
        >>> search_tool = tool(
        ...     description="Search the web",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {"type": "string", "description": "Search query"}
        ...         },
        ...         "required": ["query"]
        ...     },
        ...     execute=lambda query: {"results": ["result1", "result2"]}
        ... )
    """
    return Tool(description=description, parameters=parameters, execute=execute)


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
    finish_reason: str | None = None
    usage: Usage | None = None
    response: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


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

    Example:
        >>> result = stream_text(model=google("gemini-2.0-flash"), prompt="Hi")
        >>> async for chunk in result.text_stream:
        ...     print(chunk, end="")
        >>> print(await result.usage)  # Usage after stream completes
        >>> print(await result.text)   # Full accumulated text

    Or iterate directly:
        >>> async for chunk in stream_text(model=google("gemini-2.0-flash"), prompt="Hi"):
        ...     print(chunk, end="")
    """

    def __init__(self, stream: AsyncIterator[StreamChunk]) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._usage: Usage | None = None
        self._finish_reason: str | None = None
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

    def __aiter__(self) -> AsyncIterator[str]:
        """Allow direct iteration: async for chunk in stream_text(...)"""
        return self.text_stream


# Provider options type - allows provider-specific configuration
# Example: {"google": {"safety_settings": {...}}, "anthropic": {"top_k": 10}}
ProviderOptions = dict[str, dict[str, Any]]
