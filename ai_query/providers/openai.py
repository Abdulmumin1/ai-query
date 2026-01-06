"""OpenAI provider adapter using direct HTTP API."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
import json
import base64

import aiohttp

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage, StreamChunk
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

    async def _convert_messages(self, messages: list[Message], session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multimodal content
                content_parts = []
                for part in msg.content:
                    # Handle dict-style parts (from user input)
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image":
                            image_data = part.get("image")
                            media_type = part.get("media_type", "image/png")

                            # Handle URL
                            if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                                image_data, media_type = await self._fetch_resource_as_base64(image_data, session)
                                image_data = f"data:{media_type};base64,{image_data}"
                            elif isinstance(image_data, bytes):
                                image_data = f"data:{media_type};base64,{base64.b64encode(image_data).decode()}"
                            elif isinstance(image_data, str) and not image_data.startswith("data:"):
                                # Assume it's base64 data
                                image_data = f"data:{media_type};base64,{image_data}"

                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            })
                        elif part.get("type") == "file":
                            file_data = part.get("data")
                            media_type = part.get("media_type")

                            # Handle URL
                            if isinstance(file_data, str) and file_data.startswith(("http://", "https://")):
                                file_data, fetched_type = await self._fetch_resource_as_base64(file_data, session)
                                if not media_type:
                                    media_type = fetched_type
                                file_data = f"data:{media_type};base64,{file_data}"
                            elif isinstance(file_data, bytes):
                                file_data = f"data:{media_type};base64,{base64.b64encode(file_data).decode()}"

                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": file_data}
                            })

                    # Handle dataclass-style parts
                    elif hasattr(part, "text"):
                        content_parts.append({"type": "text", "text": part.text})
                    elif hasattr(part, "image"):
                        image_data = part.image
                        media_type = getattr(part, "media_type", "image/png")

                        if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                            image_data, media_type = await self._fetch_resource_as_base64(image_data, session)
                            image_data = f"data:{media_type};base64,{image_data}"
                        elif isinstance(image_data, bytes):
                            image_data = f"data:{media_type};base64,{base64.b64encode(image_data).decode()}"

                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        })
                    elif hasattr(part, "data"): # FilePart
                        file_data = part.data
                        media_type = getattr(part, "media_type", None)

                        if isinstance(file_data, str) and file_data.startswith(("http://", "https://")):
                            file_data, fetched_type = await self._fetch_resource_as_base64(file_data, session)
                            if not media_type:
                                media_type = fetched_type
                            file_data = f"data:{media_type};base64,{file_data}"
                        elif isinstance(file_data, bytes):
                            file_data = f"data:{media_type};base64,{base64.b64encode(file_data).decode()}"

                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": file_data},
                        })

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
        openai_options = self.get_provider_options(provider_options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        url = f"{self.base_url}/chat/completions"

        async with aiohttp.ClientSession() as session:
            # Convert messages and fetch resources if needed
            converted_messages = await self._convert_messages(messages, session)

            # Build request parameters
            request_body: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                **kwargs,
                **openai_options,
            }

            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error ({resp.status}): {error_text}")
                response = await resp.json()

        # Extract result
        choice = response["choices"][0]
        usage = None
        if "usage" in response:
            usage_data = response["usage"]
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                cached_tokens=usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
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
    ) -> AsyncIterator[StreamChunk]:
        """Stream text using OpenAI API.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o").
            messages: Conversation messages.
            provider_options: OpenAI-specific options under "openai" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Yields:
            StreamChunk objects with text and final metadata.
        """
        openai_options = self.get_provider_options(provider_options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        url = f"{self.base_url}/chat/completions"

        finish_reason = None
        usage = None

        async with aiohttp.ClientSession() as session:
            # Convert messages and fetch resources if needed
            converted_messages = await self._convert_messages(messages, session)

            # Build request parameters with streaming enabled
            # Include stream_options to get usage in streaming response
            request_body: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                **kwargs,
                **openai_options,
            }

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

                        # Check for usage in the chunk (sent at the end)
                        if "usage" in chunk and chunk["usage"]:
                            usage_data = chunk["usage"]
                            usage = Usage(
                                input_tokens=usage_data.get("prompt_tokens", 0),
                                output_tokens=usage_data.get("completion_tokens", 0),
                                cached_tokens=usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0),
                            )

                        choices = chunk.get("choices", [])
                        if choices:
                            choice = choices[0]
                            # Check for finish reason
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]

                            delta = choice.get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield StreamChunk(text=content)
                    except json.JSONDecodeError:
                        continue

        # Yield final chunk with metadata
        yield StreamChunk(is_final=True, usage=usage, finish_reason=finish_reason)
