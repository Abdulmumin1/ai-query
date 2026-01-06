"""Google (Gemini) provider adapter."""

from __future__ import annotations

import os
from typing import Any

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage


class GoogleProvider(BaseProvider):
    """Google (Gemini) provider adapter."""

    name = "google"

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize Google provider.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            **kwargs: Additional configuration.
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    def _get_client(self) -> Any:
        """Get or create Google Generative AI client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required for Google provider. "
                "Install it with: pip install google-genai"
            )

        return genai.Client(api_key=self.api_key)

    def _convert_messages(
        self, messages: list[Message]
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
                    if hasattr(part, "text"):
                        parts.append({"text": part.text})
                    elif hasattr(part, "image"):
                        image_data = part.image
                        if isinstance(image_data, bytes):
                            import base64

                            image_data = base64.b64encode(image_data).decode()
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": getattr(
                                        part, "media_type", "image/png"
                                    ),
                                    "data": image_data,
                                }
                            }
                        )
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
        client = self._get_client()
        google_options = self.get_provider_options(provider_options)

        # Convert messages
        system_instruction, contents = self._convert_messages(messages)

        # Build generation config from kwargs
        from google.genai import types

        generation_config_params: dict[str, Any] = {}
        if "max_tokens" in kwargs:
            generation_config_params["max_output_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            generation_config_params["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            generation_config_params["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config_params["top_k"] = kwargs.pop("top_k")
        if "stop_sequences" in kwargs:
            generation_config_params["stop_sequences"] = kwargs.pop("stop_sequences")

        # Merge with any generation_config from provider options
        if "generation_config" in google_options:
            generation_config_params.update(google_options.pop("generation_config"))

        # Build request config
        config: dict[str, Any] = {}
        if generation_config_params:
            config.update(generation_config_params)
        if system_instruction:
            config["system_instruction"] = system_instruction

        # Add remaining google options (safety_settings, etc.)
        config.update(google_options)

        # Make API call
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config) if config else None,
        )

        # Extract text
        text = response.text or ""

        # Build usage info if available
        usage = None
        if response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        # Determine finish reason
        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

        return GenerateTextResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            response=response.model_dump() if hasattr(response, "model_dump") else {},
            provider_metadata={"model": model},
        )
