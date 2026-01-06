"""Base provider interface for ai-query."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai_query.types import GenerateTextResult, Message, ProviderOptions


class BaseProvider(ABC):
    """Abstract base class for AI providers.

    To create a new provider adapter:
    1. Subclass BaseProvider
    2. Implement the `generate` method
    3. Register your provider using `register_provider`
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
