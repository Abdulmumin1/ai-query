"""Core types for ai-query library."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Union, AsyncIterator, Coroutine

# Message types
Role = Literal["system", "user", "assistant", "tool"]


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
