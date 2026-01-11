"""Google (Gemini) provider adapter using direct HTTP API."""

from __future__ import annotations

import os
import base64
from typing import Any, AsyncIterator
import json

import aiohttp

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage, StreamChunk, ToolSet, ToolCall, ToolCallPart, ToolResultPart
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
            if msg.role == "user" or msg.role == "tool":
                role = "user"
            else:
                role = "model"

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
                    elif hasattr(part, "type") and part.type == "tool_call":
                        # ToolCallPart - convert to functionCall
                        tc = part.tool_call
                        if tc:
                            part_data = {
                                "functionCall": {
                                    "name": tc.name,
                                    "args": tc.arguments,
                                }
                            }
                            # Include thoughtSignature if present (required for Gemini 3)
                            if tc.metadata and tc.metadata.get("thought_signature"):
                                part_data["thoughtSignature"] = tc.metadata["thought_signature"]
                            parts.append(part_data)
                    elif hasattr(part, "type") and part.type == "tool_result":
                        # ToolResultPart - convert to functionResponse
                        tr = part.tool_result
                        if tr:
                            parts.append({
                                "functionResponse": {
                                    "name": tr.tool_name,
                                    "response": tr.result if isinstance(tr.result, dict) else {"result": tr.result},
                                }
                            })
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

    def _convert_tools(self, tools: ToolSet) -> list[dict[str, Any]]:
        """Convert ToolSet to Google function calling format."""
        function_declarations = []
        for name, tool in tools.items():
            function_declarations.append({
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return [{"functionDeclarations": function_declarations}]

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
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

            # Build generation config from kwargs (common params)
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
            if "presence_penalty" in kwargs:
                generation_config["presencePenalty"] = kwargs.pop("presence_penalty")
            if "frequency_penalty" in kwargs:
                generation_config["frequencyPenalty"] = kwargs.pop("frequency_penalty")
            if "seed" in kwargs:
                generation_config["seed"] = kwargs.pop("seed")

            # Key mapping for snake_case to camelCase conversion
            key_mapping = {
                "max_output_tokens": "maxOutputTokens",
                "top_p": "topP",
                "top_k": "topK",
                "stop_sequences": "stopSequences",
                "candidate_count": "candidateCount",
                "presence_penalty": "presencePenalty",
                "frequency_penalty": "frequencyPenalty",
                "response_mime_type": "responseMimeType",
                "response_schema": "responseSchema",
                "thinking_config": "thinkingConfig",
                "speech_config": "speechConfig",
            }

            # Extract generation config options directly from google_options
            # (users can pass them flat, without nesting in "generation_config")
            for snake_key, camel_key in key_mapping.items():
                if snake_key in google_options:
                    generation_config[camel_key] = google_options.pop(snake_key)
                elif camel_key in google_options:
                    generation_config[camel_key] = google_options.pop(camel_key)
            
            # Also check for direct camelCase keys that aren't in the mapping
            direct_keys = ["temperature", "seed"]
            for key in direct_keys:
                if key in google_options:
                    generation_config[key] = google_options.pop(key)

            # Handle nested thinking_config conversion
            if "thinkingConfig" in generation_config:
                tc = generation_config["thinkingConfig"]
                if isinstance(tc, dict):
                    if "include_thoughts" in tc:
                        tc["includeThoughts"] = tc.pop("include_thoughts")
                    if "thinking_budget" in tc:
                        tc["thinkingBudget"] = tc.pop("thinking_budget")

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
                    # Convert dict format {category: threshold} to list format
                    request_body["safetySettings"] = [
                        {"category": k, "threshold": v}
                        for k, v in safety_settings.items()
                    ]
                elif isinstance(safety_settings, list):
                    # Already in list format, use as-is
                    request_body["safetySettings"] = safety_settings

            # Add tools if provided
            if tools:
                request_body["tools"] = self._convert_tools(tools)

            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Google API error ({resp.status}): {error_text}")
                response = await resp.json()

        # Extract text and function calls from response
        text = ""
        tool_calls: list[ToolCall] = []
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            for i, part in enumerate(content.get("parts", [])):
                if "text" in part:
                    text += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Capture thoughtSignature for Gemini 3 models
                    metadata = {}
                    if "thoughtSignature" in part:
                        metadata["thought_signature"] = part["thoughtSignature"]
                    tool_calls.append(ToolCall(
                        id=f"call_{i}",  # Google doesn't provide IDs, generate one
                        name=fc.get("name"),
                        arguments=fc.get("args", {}),
                        metadata=metadata,
                    ))

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

        # Build response dict with tool_calls for the execution loop
        response_with_tools = dict(response)
        response_with_tools["tool_calls"] = tool_calls

        return GenerateTextResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            response=response_with_tools,
            provider_metadata={"model": model},
        )

    def _build_request_body(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None,
        google_options: dict[str, Any],
        tools: ToolSet | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Google API."""
        # Build generation config from kwargs (common params)
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
        if "presence_penalty" in kwargs:
            generation_config["presencePenalty"] = kwargs.pop("presence_penalty")
        if "frequency_penalty" in kwargs:
            generation_config["frequencyPenalty"] = kwargs.pop("frequency_penalty")
        if "seed" in kwargs:
            generation_config["seed"] = kwargs.pop("seed")

        # Key mapping for snake_case to camelCase conversion
        key_mapping = {
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "stop_sequences": "stopSequences",
            "candidate_count": "candidateCount",
            "presence_penalty": "presencePenalty",
            "frequency_penalty": "frequencyPenalty",
            "response_mime_type": "responseMimeType",
            "response_schema": "responseSchema",
            "thinking_config": "thinkingConfig",
            "speech_config": "speechConfig",
        }

        # Extract generation config options directly from google_options
        # (users can pass them flat, without nesting in "generation_config")
        for snake_key, camel_key in key_mapping.items():
            if snake_key in google_options:
                generation_config[camel_key] = google_options.pop(snake_key)
            elif camel_key in google_options:
                generation_config[camel_key] = google_options.pop(camel_key)
        
        # Also check for direct camelCase keys that aren't in the mapping
        direct_keys = ["temperature", "seed"]
        for key in direct_keys:
            if key in google_options:
                generation_config[key] = google_options.pop(key)

        # Handle nested thinking_config conversion
        if "thinkingConfig" in generation_config:
            tc = generation_config["thinkingConfig"]
            if isinstance(tc, dict):
                if "include_thoughts" in tc:
                    tc["includeThoughts"] = tc.pop("include_thoughts")
                if "thinking_budget" in tc:
                    tc["thinkingBudget"] = tc.pop("thinking_budget")

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
                # Convert dict format {category: threshold} to list format
                request_body["safetySettings"] = [
                    {"category": k, "threshold": v}
                    for k, v in safety_settings.items()
                ]
            elif isinstance(safety_settings, list):
                # Already in list format, use as-is
                request_body["safetySettings"] = safety_settings

        # Add tools if provided
        if tools:
            request_body["tools"] = self._convert_tools(tools)

        return request_body

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text using Google Gemini API.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            messages: Conversation messages.
            tools: Optional tool definitions for tool calling.
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
                contents, system_instruction, google_options, tools=tools, **kwargs
            )

            # Use streamGenerateContent endpoint
            url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"

            finish_reason = None
            usage = None
            tool_calls: list[ToolCall] = []

            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Google API error ({resp.status}): {error_text}")

                # Process SSE stream
                async for line in resp.content:
                    chunk = self._parse_sse_json(line)
                    if chunk is None:
                        continue

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
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                # Capture thoughtSignature for Gemini 3 models
                                metadata = {}
                                if "thoughtSignature" in part:
                                    metadata["thought_signature"] = part["thoughtSignature"]
                                tool_calls.append(ToolCall(
                                    id=f"call_{len(tool_calls)}",  # Google doesn't provide IDs
                                    name=fc.get("name"),
                                    arguments=fc.get("args", {}),
                                    metadata=metadata,
                                ))

        # Yield final chunk with metadata
        yield StreamChunk(
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None
        )
