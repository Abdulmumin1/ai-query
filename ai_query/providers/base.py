"""Base provider interface for ai-query."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ai_query.types import GenerateTextResult, Message, ProviderOptions, StreamChunk


class BaseProvider(ABC):
    """Abstract base class for AI providers.

    To create a new provider adapter:
    1. Subclass BaseProvider
    2. Implement the `generate` method
    3. Implement the `stream` method for streaming support
    """

    # Provider identifier (e.g., "openai", "anthropic", "google")
    name: str

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize the provider.

        Args:
            api_key: API key for the provider. If None, will try to read from
                     environment variable (provider-specific).
            **kwargs: Additional provider-specific configuration.
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using the provider's API.

        Args:
            model: The model identifier (without provider prefix).
            messages: List of messages in the conversation.
            provider_options: Provider-specific options.
            **kwargs: Additional parameters (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult containing the generated text and metadata.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text using the provider's API.

        Args:
            model: The model identifier (without provider prefix).
            messages: List of messages in the conversation.
            provider_options: Provider-specific options.
            **kwargs: Additional parameters (max_tokens, temperature, etc.).

        Yields:
            StreamChunk objects containing text and metadata.
            The final chunk will have is_final=True with usage and finish_reason.
        """
        pass
        # Make this an async generator
        if False:
            yield StreamChunk()

    def get_provider_options(
        self, provider_options: ProviderOptions | None
    ) -> dict[str, Any]:
        """Extract options specific to this provider.

        Args:
            provider_options: Full provider options dict.

        Returns:
            Options for this specific provider, or empty dict.
        """
        if provider_options is None:
            return {}
        return provider_options.get(self.name, {})

    async def _fetch_resource_as_base64(self, url: str, session: Any) -> tuple[str, str]:
        """Fetch a resource from URL and return as base64 with media type."""
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch resource from {url}: {resp.status}")
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            # Extract just the mime type (remove charset etc)
            media_type = content_type.split(";")[0].strip()
            data_bytes = await resp.read()
            import base64
            return base64.b64encode(data_bytes).decode(), media_type
