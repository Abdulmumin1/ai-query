"""Groq provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq provider - wraps OpenAI provider with Groq's base URL."""

    name = "groq"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )


# Cached provider instance
_default_provider: GroqProvider | None = None


def groq(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create a Groq language model.

    Groq provides ultra-fast inference for open models through an OpenAI-compatible API
    at https://api.groq.com/openai/v1

    Args:
        model_id: The model identifier (e.g., "llama-3.3-70b-versatile", "mixtral-8x7b-32768").
        api_key: Groq API key. Falls back to GROQ_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, groq
        >>> result = await generate_text(
        ...     model=groq("llama-3.3-70b-versatile"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = GroqProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = GroqProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)

