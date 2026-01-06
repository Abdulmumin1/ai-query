"""Core types for ai-query library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

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
        return {
            "role": self.role,
            "content": [
                {"type": p.type, **({"text": p.text} if isinstance(p, TextPart) else {})}
                for p in self.content
            ],
        }


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
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


# Provider options type - allows provider-specific configuration
# Example: {"google": {"safety_settings": {...}}, "anthropic": {"top_k": 10}}
ProviderOptions = dict[str, dict[str, Any]]
