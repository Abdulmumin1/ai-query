"""xAI provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


class XAIProvider(OpenAIProvider):
    """xAI provider - wraps OpenAI provider with xAI's base URL."""

    name = "xai"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("XAI_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://api.x.ai/v1",
            **kwargs,
        )


# Cached provider instance
_default_provider: XAIProvider | None = None


def xai(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create an xAI language model.

    Access xAI models (like Grok) through the official API at https://api.x.ai/v1

    Args:
        model_id: The model identifier (e.g., "grok-2-1212", "grok-2-vision-1212").
        api_key: xAI API key. Falls back to XAI_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, xai
        >>> result = await generate_text(
        ...     model=xai("grok-2-1212"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = XAIProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = XAIProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)
