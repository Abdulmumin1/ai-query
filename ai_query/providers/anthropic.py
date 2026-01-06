"""Anthropic provider adapter."""

from __future__ import annotations

import os
from typing import Any

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage


class AnthropicProvider(BaseProvider):
    """Anthropic provider adapter."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            **kwargs: Additional configuration (base_url, etc.).
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = kwargs.get("base_url")

    def _get_client(self) -> Any:
        """Get or create Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install it with: pip install anthropic"
            )

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        return AsyncAnthropic(**client_kwargs)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Message objects to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages_list).
        """
        system_prompt: str | None = None
        result = []

        for msg in messages:
            # Extract system message separately (Anthropic uses system parameter)
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_prompt = msg.content
                continue

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
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": getattr(
                                        part, "media_type", "image/png"
                                    ),
                                    "data": image_data,
                                },
                            }
                        )
                result.append({"role": msg.role, "content": content_parts})

        return system_prompt, result

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using Anthropic API.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-3-opus-20240229").
            messages: Conversation messages.
            provider_options: Anthropic-specific options under "anthropic" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult with generated text and metadata.
        """
        client = self._get_client()
        anthropic_options = self.get_provider_options(provider_options)

        # Convert messages and extract system prompt
        system_prompt, converted_messages = self._convert_messages(messages)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
            **anthropic_options,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        # Make API call
        response = await client.messages.create(**request_params)

        # Extract text from response
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Build usage info
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return GenerateTextResult(
            text=text,
            finish_reason=response.stop_reason,
            usage=usage,
            response=response.model_dump(),
            provider_metadata={"model": response.model, "id": response.id},
        )
