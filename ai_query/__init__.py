"""Main entry point for ai-query library."""

from __future__ import annotations

from typing import Any, AsyncIterator

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    TextPart,
    ImagePart,
    FilePart,
    Usage,
    TextStreamResult,
)
from ai_query.model import LanguageModel
from ai_query.providers.base import BaseProvider
from ai_query.providers.openai import OpenAIProvider, openai
from ai_query.providers.anthropic import AnthropicProvider, anthropic
from ai_query.providers.google import GoogleProvider, google


async def generate_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
    provider_options: ProviderOptions | None = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate text using an AI model.

    This is the main function for text generation. It supports three input modes:
    1. Simple prompt: Just pass `prompt` for quick queries
    2. System + prompt: Pass `system` and `prompt` for guided generation
    3. Full messages: Pass `messages` for complete conversation control

    Args:
        model: A LanguageModel instance created by a provider function
            (e.g., openai("gpt-4"), anthropic("claude-sonnet-4-20250514"), google("gemini-2.0-flash")).
        prompt: Simple text prompt (mutually exclusive with messages).
        system: System prompt to guide model behavior.
        messages: Full conversation history as Message objects or dicts.
        provider_options: Provider-specific options.
            Example: {"google": {"safety_settings": {...}}}
        **kwargs: Additional parameters (max_tokens, temperature, etc.).

    Returns:
        GenerateTextResult containing:
            - text: The generated text
            - finish_reason: Why generation stopped
            - usage: Token usage statistics
            - response: Raw response data
            - provider_metadata: Provider-specific metadata

    Examples:
        Simple prompt:
        >>> from ai_query import generate_text, openai
        >>> result = await generate_text(
        ...     model=openai("gpt-4"),
        ...     prompt="What is the capital of France?"
        ... )
        >>> print(result.text)

        With system prompt:
        >>> from ai_query import generate_text, anthropic
        >>> result = await generate_text(
        ...     model=anthropic("claude-sonnet-4-20250514"),
        ...     system="You are a helpful assistant.",
        ...     prompt="Explain quantum computing simply."
        ... )

        Full conversation:
        >>> from ai_query import generate_text, google
        >>> result = await generate_text(
        ...     model=google("gemini-2.0-flash"),
        ...     messages=[
        ...         {"role": "system", "content": "You are a poet."},
        ...         {"role": "user", "content": "Write a haiku about coding."}
        ...     ]
        ... )

        With provider options:
        >>> result = await generate_text(
        ...     model=google("gemini-2.0-flash"),
        ...     prompt="Tell me a story",
        ...     provider_options={
        ...         "google": {
        ...             "safety_settings": {"HARM_CATEGORY_VIOLENCE": "BLOCK_NONE"}
        ...         }
        ...     }
        ... )
    """
    # Build messages list
    final_messages: list[Message] = []

    if messages is not None:
        # Convert dict messages to Message objects if needed
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        # Build from prompt and system
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Generate using the model's provider
    return await model.provider.generate(
        model=model.model_id,
        messages=final_messages,
        provider_options=provider_options,
        **kwargs,
    )


def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
    provider_options: ProviderOptions | None = None,
    **kwargs: Any,
) -> TextStreamResult:
    """Stream text from an AI model.

    Returns a TextStreamResult object that provides both the text stream
    and metadata (usage, finish_reason) after streaming completes.

    Args:
        model: A LanguageModel instance created by a provider function
            (e.g., openai("gpt-4"), anthropic("claude-sonnet-4-20250514"), google("gemini-2.0-flash")).
        prompt: Simple text prompt (mutually exclusive with messages).
        system: System prompt to guide model behavior.
        messages: Full conversation history as Message objects or dicts.
        provider_options: Provider-specific options.
        **kwargs: Additional parameters (max_tokens, temperature, etc.).

    Returns:
        TextStreamResult with:
            - text_stream: AsyncIterator yielding text chunks
            - text: Awaitable for full text after completion
            - usage: Awaitable for Usage stats after completion
            - finish_reason: Awaitable for finish reason after completion

    Examples:
        Simple streaming (direct iteration):
        >>> async for chunk in stream_text(model=openai("gpt-4"), prompt="Hi"):
        ...     print(chunk, end="", flush=True)

        With usage access:
        >>> result = stream_text(model=google("gemini-2.0-flash"), prompt="Hi")
        >>> async for chunk in result.text_stream:
        ...     print(chunk, end="", flush=True)
        >>> usage = await result.usage
        >>> print(f"Tokens: {usage.total_tokens}")
    """
    # Build messages list
    final_messages: list[Message] = []

    if messages is not None:
        # Convert dict messages to Message objects if needed
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        # Build from prompt and system
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Create the stream from the provider
    stream = model.provider.stream(
        model=model.model_id,
        messages=final_messages,
        provider_options=provider_options,
        **kwargs,
    )

    # Return TextStreamResult wrapping the stream
    return TextStreamResult(stream)


__all__ = [
    # Main functions
    "generate_text",
    "stream_text",
    # Provider factory functions
    "openai",
    "anthropic",
    "google",
    # Types
    "LanguageModel",
    "GenerateTextResult",
    "TextStreamResult",
    "Message",
    "ProviderOptions",
    "TextPart",
    "ImagePart",
    "FilePart",
    "Usage",
    # Base class for custom providers
    "BaseProvider",
    # Built-in provider classes (for advanced usage)
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
]
