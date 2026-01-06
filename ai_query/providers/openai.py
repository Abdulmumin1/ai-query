"""OpenAI provider adapter using direct HTTP API."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
import json

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage
from ai_query.model import LanguageModel


# Cached provider instance
_default_provider: OpenAIProvider | None = None


def openai(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
) -> LanguageModel:
    """Create an OpenAI language model.

    Args:
        model_id: The model identifier (e.g., "gpt-4", "gpt-4o", "gpt-4o-mini").
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        base_url: Custom base URL for API requests.
        organization: OpenAI organization ID.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, openai
        >>> result = await generate_text(
        ...     model=openai("gpt-4o"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key or base_url or organization:
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
    else:
        if _default_provider is None:
            _default_provider = OpenAIProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)


class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter using direct HTTP API."""

    name = "openai"

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            **kwargs: Additional configuration (base_url, organization, etc.).
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.organization = kwargs.get("organization")

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multimodal content
                content_parts = []
                for part in msg.content:
                    if hasattr(part, "text"):
                        content_parts.append({"type": "text", "text": part.text})
                    elif hasattr(part, "image"):
                        image_data = part.image
                        if isinstance(image_data, bytes):
                            import base64

                            image_data = base64.b64encode(image_data).decode()
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            }
                        )
                result.append({"role": msg.role, "content": content_parts})
        return result

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using OpenAI API.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o").
            messages: Conversation messages.
            provider_options: OpenAI-specific options under "openai" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult with generated text and metadata.
        """
        import aiohttp

        openai_options = self.get_provider_options(provider_options)

        # Build request parameters
        request_body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            **kwargs,
            **openai_options,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        url = f"{self.base_url}/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error ({resp.status}): {error_text}")
                response = await resp.json()

        # Extract result
        choice = response["choices"][0]
        usage = None
        if "usage" in response:
            usage = Usage(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
                total_tokens=response["usage"]["total_tokens"],
            )

        return GenerateTextResult(
            text=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            response=response,
            provider_metadata={"model": response.get("model"), "id": response.get("id")},
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text using OpenAI API.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o").
            messages: Conversation messages.
            provider_options: OpenAI-specific options under "openai" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Yields:
            Text chunks as they arrive.
        """
        import aiohttp

        openai_options = self.get_provider_options(provider_options)

        # Build request parameters with streaming enabled
        request_body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": True,
            **kwargs,
            **openai_options,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        url = f"{self.base_url}/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error ({resp.status}): {error_text}")

                # Process SSE stream
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
