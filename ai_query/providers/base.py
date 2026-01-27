"""Base provider interface for ai-query."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, TYPE_CHECKING

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    StreamChunk,
    ToolSet,
    Usage,
    EmbedResult,
    EmbedManyResult,
    EmbeddingUsage,
)


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
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using the provider's API.

        Args:
            model: The model identifier (without provider prefix).
            messages: List of messages in the conversation.
            tools: Optional dict of tool definitions for tool calling.
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
        tools: ToolSet | None = None,
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

    async def _fetch_resource_as_base64(
        self, url: str, session: Any
    ) -> tuple[str, str]:
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

    def create_session(self) -> aiohttp.ClientSession:
        """Create an aiohttp ClientSession with environment-specific configuration.

        This handles Cloudflare Worker specific issues like disabling SSL verification
        which is required in that environment as the native 'ssl' module is not available.
        """
        import os
        import aiohttp

        connector = None
        # Check for Cloudflare/Pyodide environment by checking for ssl module
        # This is more robust than checking sys.platform or env vars
        try:
            import ssl
        except ImportError:
            connector = aiohttp.TCPConnector(ssl=False)
        else:
            # Even if ssl imports, we might be in an environment where we want to force disable it
            # if we are definitely in Cloudflare
            import sys

            if (
                sys.platform == "emscripten"
                or os.environ.get("WORKER_RUNTIME") == "cloudflare"
            ):
                connector = aiohttp.TCPConnector(ssl=False)

        return aiohttp.ClientSession(connector=connector)

    def _parse_sse_line(self, line: bytes | str) -> str | None:
        """Parse an SSE line and extract the data payload.

        Args:
            line: Raw SSE line (bytes or str).

        Returns:
            The data string if line starts with "data: ", None otherwise.
        """
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            return None
        if line.startswith("data: "):
            return line[6:]
        return None

    def _parse_sse_json(self, line: bytes | str) -> dict[str, Any] | None:
        """Parse an SSE line and decode the JSON payload.

        Args:
            line: Raw SSE line (bytes or str).

        Returns:
            Parsed JSON dict, or None if line is not valid SSE data or invalid JSON.
        """
        data = self._parse_sse_line(line)
        if data is None:
            return None
        if data == "[DONE]":
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def _accumulate_usage(self, total: Usage, delta: Usage) -> None:
        """Accumulate usage statistics in-place.

        Args:
            total: The running total Usage object (modified in-place).
            delta: The delta Usage object to add.
        """
        total.input_tokens += delta.input_tokens
        total.output_tokens += delta.output_tokens
        total.cached_tokens += delta.cached_tokens
        total.total_tokens += delta.total_tokens

    async def embed(
        self,
        *,
        model: str,
        value: str,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> EmbedResult:
        """Generate an embedding for a single value.

        Args:
            model: The embedding model identifier.
            value: The text to embed.
            provider_options: Provider-specific options.
            **kwargs: Additional parameters (dimensions, etc.).

        Returns:
            EmbedResult containing the embedding vector and metadata.

        Raises:
            NotImplementedError: If the provider doesn't support embeddings.
        """
        raise NotImplementedError(f"{self.name} provider does not support embeddings")

    async def embed_many(
        self,
        *,
        model: str,
        values: list[str],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> EmbedManyResult:
        """Generate embeddings for multiple values.

        Default implementation calls embed() for each value in parallel.
        Providers should override this for more efficient batch processing
        if their API supports it natively.

        Args:
            model: The embedding model identifier.
            values: List of texts to embed.
            provider_options: Provider-specific options.
            **kwargs: Additional parameters (dimensions, etc.).

        Returns:
            EmbedManyResult containing embedding vectors and metadata.
        """
        # Run all embed calls in parallel
        tasks = [
            self.embed(
                model=model,
                value=value,
                provider_options=provider_options,
                **kwargs,
            )
            for value in values
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        embeddings = [r.embedding for r in results]
        total_tokens = sum(r.usage.tokens for r in results)

        return EmbedManyResult(
            values=values,
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=total_tokens),
        )
