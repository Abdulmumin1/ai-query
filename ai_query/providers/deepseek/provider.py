"""DeepSeek provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider - wraps OpenAI provider with DeepSeek's base URL."""

    name = "deepseek"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
            **kwargs,
        )


# Cached provider instance
_default_provider: DeepSeekProvider | None = None


def deepseek(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create a DeepSeek language model.

    DeepSeek provides advanced AI models through an OpenAI-compatible API
    at https://api.deepseek.com

    Args:
        model_id: The model identifier (e.g., "deepseek-chat", "deepseek-reasoner").
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, deepseek
        >>> result = await generate_text(
        ...     model=deepseek("deepseek-chat"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = DeepSeekProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = DeepSeekProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)
