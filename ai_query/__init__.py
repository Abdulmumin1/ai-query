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


async def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
    provider_options: ProviderOptions | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Stream text from an AI model.

    This function streams text chunks as they arrive from the model.
    It supports the same input modes as generate_text.

    Args:
        model: A LanguageModel instance created by a provider function
            (e.g., openai("gpt-4"), anthropic("claude-sonnet-4-20250514"), google("gemini-2.0-flash")).
        prompt: Simple text prompt (mutually exclusive with messages).
        system: System prompt to guide model behavior.
        messages: Full conversation history as Message objects or dicts.
        provider_options: Provider-specific options.
        **kwargs: Additional parameters (max_tokens, temperature, etc.).

    Yields:
        Text chunks as they arrive from the model.

    Examples:
        Simple streaming:
        >>> from ai_query import stream_text, openai
        >>> async for chunk in stream_text(
        ...     model=openai("gpt-4"),
        ...     prompt="Write a short story"
        ... ):
        ...     print(chunk, end="", flush=True)

        With system prompt:
        >>> async for chunk in stream_text(
        ...     model=anthropic("claude-sonnet-4-20250514"),
        ...     system="You are a storyteller.",
        ...     prompt="Tell me a tale."
        ... ):
        ...     print(chunk, end="", flush=True)
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

    # Stream using the model's provider
    async for chunk in model.provider.stream(
        model=model.model_id,
        messages=final_messages,
        provider_options=provider_options,
        **kwargs,
    ):
        yield chunk


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
