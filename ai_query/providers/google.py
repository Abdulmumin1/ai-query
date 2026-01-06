"""Google (Gemini) provider adapter using direct HTTP API."""

from __future__ import annotations

import os
import base64
from typing import Any, AsyncIterator
import json

import aiohttp

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage, StreamChunk
from ai_query.model import LanguageModel


# Cached provider instance
_default_provider: GoogleProvider | None = None


def google(
    model_id: str,
    *,
    api_key: str | None = None,
) -> LanguageModel:
    """Create a Google (Gemini) language model.

    Args:
        model_id: The model identifier (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, google
        >>> result = await generate_text(
        ...     model=google("gemini-2.0-flash"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key:
        provider = GoogleProvider(api_key=api_key)
    else:
        if _default_provider is None:
            _default_provider = GoogleProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)


class GoogleProvider(BaseProvider):
    """Google (Gemini) provider adapter using direct HTTP API."""

    name = "google"

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize Google provider.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            **kwargs: Additional configuration.
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def _convert_messages(
        self, messages: list[Message], session: aiohttp.ClientSession
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Message objects to Google format.

        Returns:
            Tuple of (system_instruction, contents_list).
        """
        system_instruction: str | None = None
        contents = []

        for msg in messages:
            # Extract system message for system_instruction
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_instruction = msg.content
                continue

            # Map roles to Google format
            role = "user" if msg.role == "user" else "model"

            if isinstance(msg.content, str):
                contents.append({"role": role, "parts": [{"text": msg.content}]})
            else:
                # Handle multimodal content
                parts = []
                for part in msg.content:
                    # Handle dict-style parts (from user input)
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image":
                            image_data = part.get("image")
                            media_type = part.get("media_type", "image/png")

                            # Handle URL
                            if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                                image_data, media_type = await self._fetch_resource_as_base64(image_data, session)
                            elif isinstance(image_data, bytes):
                                import base64
                                image_data = base64.b64encode(image_data).decode()

                            parts.append({
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": image_data,
                                }
                            })
                        elif part.get("type") == "file":
                            file_data = part.get("data")
                            media_type = part.get("media_type")

                            # Handle URL
                            if isinstance(file_data, str) and file_data.startswith(("http://", "https://")):
                                file_data, fetched_type = await self._fetch_resource_as_base64(file_data, session)
                                if not media_type:
                                    media_type = fetched_type
                            elif isinstance(file_data, bytes):
                                import base64
                                file_data = base64.b64encode(file_data).decode()

                            parts.append({
                                "inline_data": {
                                    "mime_type": media_type or "application/octet-stream",
                                    "data": file_data,
                                }
                            })
                    # Handle dataclass-style parts
                    elif hasattr(part, "text"):
                        parts.append({"text": part.text})
                    elif hasattr(part, "image"):
                        image_data = part.image
                        media_type = getattr(part, "media_type", "image/png")

                        # Handle URL
                        if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                            image_data, media_type = await self._fetch_resource_as_base64(image_data, session)
                        elif isinstance(image_data, bytes):
                            import base64
                            image_data = base64.b64encode(image_data).decode()

                        parts.append({
                            "inline_data": {
                                "mime_type": media_type,
                                "data": image_data,
                            }
                        })
                    elif hasattr(part, "data"):  # FilePart
                        file_data = part.data
                        media_type = getattr(part, "media_type", None)

                        # Handle URL
                        if isinstance(file_data, str) and file_data.startswith(("http://", "https://")):
                            file_data, fetched_type = await self._fetch_resource_as_base64(file_data, session)
                            if not media_type:
                                media_type = fetched_type
                        elif isinstance(file_data, bytes):
                            import base64
                            file_data = base64.b64encode(file_data).decode()

                        parts.append({
                            "inline_data": {
                                "mime_type": media_type or "application/octet-stream",
                                "data": file_data,
                            }
                        })
                contents.append({"role": role, "parts": parts})

        return system_instruction, contents

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using Google Gemini API.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            messages: Conversation messages.
            provider_options: Google-specific options under "google" key.
                Supports: safety_settings, generation_config, etc.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult with generated text and metadata.
        """
        import aiohttp

        google_options = self.get_provider_options(provider_options)

        async with aiohttp.ClientSession() as session:
            # Convert messages
            system_instruction, contents = await self._convert_messages(messages, session)

            # Build generation config from kwargs
            generation_config: dict[str, Any] = {}
            if "max_tokens" in kwargs:
                generation_config["maxOutputTokens"] = kwargs.pop("max_tokens")
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "top_p" in kwargs:
                generation_config["topP"] = kwargs.pop("top_p")
            if "top_k" in kwargs:
                generation_config["topK"] = kwargs.pop("top_k")
            if "stop_sequences" in kwargs:
                generation_config["stopSequences"] = kwargs.pop("stop_sequences")

            # Merge with any generation_config from provider options
            if "generation_config" in google_options:
                generation_config.update(google_options.pop("generation_config"))

            # Build request body
            request_body: dict[str, Any] = {
                "contents": contents,
            }

            if system_instruction:
                request_body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            if generation_config:
                request_body["generationConfig"] = generation_config

            # Add remaining google options (safety_settings, etc.)
            if "safety_settings" in google_options:
                # Convert safety settings to proper format
                safety_settings = google_options.pop("safety_settings")
                if isinstance(safety_settings, dict):
                    # Convert dict format to list format expected by API
                    request_body["safetySettings"] = [
                        {"category": k, "threshold": v}
                        for k, v in safety_settings.items()
                    ]
                else:
                    request_body["safetySettings"] = safety_settings

            # Only pass through known Google API fields, ignore others
            # (prevents errors from unknown fields like "thinkingConfig")

            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Google API error ({resp.status}): {error_text}")
                response = await resp.json()

        # Extract text from response
        text = ""
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    text += part["text"]

        # Build usage info if available
        usage = None
        usage_metadata = response.get("usageMetadata", {})
        if usage_metadata:
            usage = Usage(
                input_tokens=usage_metadata.get("promptTokenCount", 0),
                output_tokens=usage_metadata.get("candidatesTokenCount", 0),
                cached_tokens=usage_metadata.get("cachedContentTokenCount", 0),
                total_tokens=usage_metadata.get("totalTokenCount", 0),
            )

        # Determine finish reason
        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")

        return GenerateTextResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            response=response,
            provider_metadata={"model": model},
        )

    def _build_request_body(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None,
        google_options: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Google API."""
        # Build generation config from kwargs
        generation_config: dict[str, Any] = {}
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config["topK"] = kwargs.pop("top_k")
        if "stop_sequences" in kwargs:
            generation_config["stopSequences"] = kwargs.pop("stop_sequences")

        # Merge with any generation_config from provider options
        if "generation_config" in google_options:
            generation_config.update(google_options.pop("generation_config"))

        # Build request body
        request_body: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            request_body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        if generation_config:
            request_body["generationConfig"] = generation_config

        # Add safety_settings if provided
        if "safety_settings" in google_options:
            safety_settings = google_options.pop("safety_settings")
            if isinstance(safety_settings, dict):
                request_body["safetySettings"] = [
                    {"category": k, "threshold": v}
                    for k, v in safety_settings.items()
                ]
            else:
                request_body["safetySettings"] = safety_settings

        return request_body

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text using Google Gemini API.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            messages: Conversation messages.
            provider_options: Google-specific options under "google" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Yields:
            StreamChunk objects with text and final metadata.
        """
        import aiohttp

        google_options = self.get_provider_options(provider_options)

        async with aiohttp.ClientSession() as session:
            # Convert messages
            system_instruction, contents = await self._convert_messages(messages, session)

            # Build request body
            request_body = self._build_request_body(
                contents, system_instruction, google_options, **kwargs
            )

            # Use streamGenerateContent endpoint
            url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"

            finish_reason = None
            usage = None

            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Google API error ({resp.status}): {error_text}")

                # Process SSE stream
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]
                    try:
                        chunk = json.loads(data)

                        # Check for usage metadata
                        usage_metadata = chunk.get("usageMetadata", {})
                        if usage_metadata:
                            usage = Usage(
                                input_tokens=usage_metadata.get("promptTokenCount", 0),
                                output_tokens=usage_metadata.get("candidatesTokenCount", 0),
                                cached_tokens=usage_metadata.get("cachedContentTokenCount", 0),
                                total_tokens=usage_metadata.get("totalTokenCount", 0),
                            )

                        candidates = chunk.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            # Check for finish reason
                            if candidate.get("finishReason"):
                                finish_reason = candidate["finishReason"]

                            content = candidate.get("content", {})
                            for part in content.get("parts", []):
                                if "text" in part:
                                    yield StreamChunk(text=part["text"])
                    except json.JSONDecodeError:
                        continue

        # Yield final chunk with metadata
        yield StreamChunk(is_final=True, usage=usage, finish_reason=finish_reason)
