"""Meta Llama provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


class LlamaProvider(OpenAIProvider):
    """Meta Llama provider - wraps OpenAI provider with Llama's base URL."""

    name = "llama"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("LLAMA_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://api.llama.com/compat/v1/",
            **kwargs,
        )


# Cached provider instance
_default_provider: LlamaProvider | None = None


def llama(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create a Meta Llama language model.

    Access Meta Llama models through the official API at https://api.llama.com/compat/v1/

    Args:
        model_id: The model identifier (e.g., "llama-3.1-405b-instruct", "llama-3.1-70b-instruct").
        api_key: Llama API key. Falls back to LLAMA_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, llama
        >>> result = await generate_text(
        ...     model=llama("llama-3.1-70b-instruct"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = LlamaProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = LlamaProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)
