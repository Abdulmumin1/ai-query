"""Main entry point for ai-query library."""

from __future__ import annotations

from typing import Any

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    TextPart,
    ImagePart,
    FilePart,
    Usage,
)
from ai_query.providers.base import BaseProvider
from ai_query.providers.openai import OpenAIProvider
from ai_query.providers.anthropic import AnthropicProvider
from ai_query.providers.google import GoogleProvider

# Global provider registry
_providers: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}

# Cached provider instances
_provider_instances: dict[str, BaseProvider] = {}


def register_provider(name: str, provider_class: type[BaseProvider]) -> None:
    """Register a new provider adapter.

    Args:
        name: Provider identifier (e.g., "mistral", "cohere").
        provider_class: Class implementing BaseProvider.

    Example:
        >>> from ai_query import register_provider
        >>> from my_providers import MistralProvider
        >>> register_provider("mistral", MistralProvider)
    """
    _providers[name] = provider_class


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider identifier.
        **kwargs: Provider-specific configuration.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider is not registered.
    """
    if name not in _providers:
        available = ", ".join(_providers.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")

    # Create new instance with config
    return _providers[name](**kwargs)


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse model string to extract provider and model name.

    Args:
        model: Model string like "openai/gpt-4" or "anthropic/claude-sonnet-4-20250514".

    Returns:
        Tuple of (provider_name, model_name).

    Raises:
        ValueError: If model string format is invalid.
    """
    if "/" not in model:
        raise ValueError(
            f"Invalid model format: {model}. "
            "Expected format: 'provider/model' (e.g., 'openai/gpt-4')"
        )

    parts = model.split("/", 1)
    return parts[0], parts[1]


async def generate_text(
    *,
    model: str,
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
        model: Model identifier in format "provider/model".
            Examples: "openai/gpt-4", "anthropic/claude-sonnet-4-20250514", "google/gemini-2.0-flash"
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
        >>> result = await generate_text(
        ...     model="openai/gpt-4",
        ...     prompt="What is the capital of France?"
        ... )
        >>> print(result.text)

        With system prompt:
        >>> result = await generate_text(
        ...     model="anthropic/claude-sonnet-4-20250514",
        ...     system="You are a helpful assistant.",
        ...     prompt="Explain quantum computing simply."
        ... )

        Full conversation:
        >>> result = await generate_text(
        ...     model="google/gemini-2.0-flash",
        ...     messages=[
        ...         {"role": "system", "content": "You are a poet."},
        ...         {"role": "user", "content": "Write a haiku about coding."}
        ...     ]
        ... )

        With provider options:
        >>> result = await generate_text(
        ...     model="google/gemini-2.0-flash",
        ...     prompt="Tell me a story",
        ...     provider_options={
        ...         "google": {
        ...             "safety_settings": {"HARM_CATEGORY_VIOLENCE": "BLOCK_NONE"}
        ...         }
        ...     }
        ... )
    """
    # Parse model string
    provider_name, model_name = _parse_model_string(model)

    # Get or create provider instance
    provider = get_provider(provider_name)

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

    # Generate
    return await provider.generate(
        model=model_name,
        messages=final_messages,
        provider_options=provider_options,
        **kwargs,
    )


__all__ = [
    # Main function
    "generate_text",
    # Provider management
    "register_provider",
    "get_provider",
    # Types
    "GenerateTextResult",
    "Message",
    "ProviderOptions",
    "TextPart",
    "ImagePart",
    "FilePart",
    "Usage",
    # Base class for custom providers
    "BaseProvider",
    # Built-in providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
]
