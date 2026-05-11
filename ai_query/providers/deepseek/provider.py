"""DeepSeek provider - OpenAI-compatible API."""

from __future__ import annotations

import os

from ai_query.model import LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider
from typing import Any
from ai_query.types import Message


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider - wraps OpenAI provider with DeepSeek's base URL."""

    name = "deepseek"
    _upstream_max_tokens_param = "max_tokens"

    def __init__(self, api_key: str | None = None, **kwargs):
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
            **kwargs,
        )

    async def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for message in messages:
            converted_message = await super()._convert_messages([message])
            converted_message = [
                self._sanitize_message_content(item)
                for item in converted_message
            ]
            if message.role == "assistant":
                for item in converted_message:
                    if item.get("role") == "assistant":
                        reasoning_text = self.reasoning_text(message)
                        item["reasoning_content"] = reasoning_text
            converted.extend(converted_message)

        return converted

    def _sanitize_message_content(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, list):
            return message

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))

        return {
            **message,
            "content": "".join(text_parts),
        }

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
