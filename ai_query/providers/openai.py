"""OpenAI provider adapter."""

from __future__ import annotations

import os
from typing import Any

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage


class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter."""

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

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install it with: pip install openai"
            )

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )

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
        client = self._get_client()
        openai_options = self.get_provider_options(provider_options)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            **kwargs,
            **openai_options,
        }

        # Make API call
        response = await client.chat.completions.create(**request_params)

        # Extract result
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return GenerateTextResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            response=response.model_dump(),
            provider_metadata={"model": response.model, "id": response.id},
        )
