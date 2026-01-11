"""OpenRouter provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider - wraps OpenAI provider with OpenRouter's base URL."""

    name = "openrouter"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )


# Cached provider instance
_default_provider: OpenRouterProvider | None = None


def openrouter(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create an OpenRouter language model.

    OpenRouter provides access to many AI models through a single API.
    Uses the OpenAI-compatible API at https://openrouter.ai/api/v1

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet").
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, openrouter
        >>> result = await generate_text(
        ...     model=openrouter("openai/gpt-4o"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = OpenRouterProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = OpenRouterProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)
