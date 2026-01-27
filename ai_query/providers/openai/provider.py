"""OpenAI provider adapter using direct HTTP API."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, AsyncIterator

from ai_query.providers.base import BaseProvider
from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    Usage,
    StreamChunk,
    ToolSet,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
    EmbedResult,
    EmbedManyResult,
    EmbeddingUsage,
    TextPart,
    ImagePart,
    FilePart,
)
from ai_query.model import LanguageModel, EmbeddingModel


# Cached provider instance
_default_provider: OpenAIProvider | None = None


# Cached embedding provider instance
_default_embedding_provider: OpenAIProvider | None = None


class _OpenAINamespace:
    """Namespace for OpenAI provider functions.

    Provides both language model and embedding model factory functions.

    Example:
        >>> from ai_query import openai
        >>> # Language model
        >>> model = openai("gpt-4o")
        >>> # Embedding model
        >>> embedding_model = openai.embedding("text-embedding-3-small")
    """

    def __call__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> LanguageModel:
        """Create an OpenAI language model."""
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

    def embedding(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> EmbeddingModel:
        """Create an OpenAI embedding model."""
        global _default_embedding_provider

        if api_key or base_url or organization:
            provider = OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
            )
        else:
            if _default_embedding_provider is None:
                _default_embedding_provider = OpenAIProvider()
            provider = _default_embedding_provider

        return EmbeddingModel(provider=provider, model_id=model_id)


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
        env_var = f"{self.name.upper()}_API_KEY"
        self.api_key = api_key or os.environ.get(env_var)
        if not self.api_key:
            raise ValueError(
                f"Error: {self.name.upper()} API key is missing. Pass it using the 'api_key' parameter "
                f"or the {env_var} environment variable."
            )
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.organization = kwargs.get("organization")

    async def _convert_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
                continue

            # Handle list content
            # Special handling for tool messages (OpenAI requires one message per tool result)
            if msg.role == "tool":
                for part in msg.content:
                    if hasattr(part, "type") and part.type == "tool_result":
                        tr = part.tool_result
                        if tr:
                            result.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tr.tool_call_id,
                                    "content": str(tr.result),  # Ensure string content
                                }
                            )
                    elif isinstance(part, dict) and part.get("type") == "tool_result":
                        # Handle dict format if passed directly
                        tr = part.get("tool_result")
                        if tr:
                            result.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tr.tool_call_id,
                                    "content": str(tr.result),
                                }
                            )
                continue

            # Special handling for assistant messages with tool calls
            if msg.role == "assistant":
                content_text = ""
                tool_calls = []

                for part in msg.content:
                    if isinstance(part, ToolCallPart):
                        tc = part.tool_call
                        if tc:
                            tool_calls.append(
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments),
                                    },
                                }
                            )
                    elif isinstance(part, TextPart):
                        content_text += part.text
                    elif isinstance(part, dict):
                        if part.get("type") == "tool_call":
                            tc = part.get("tool_call")
                            if tc:
                                tool_calls.append(
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": json.dumps(tc.arguments),
                                        },
                                    }
                                )
                        elif part.get("type") == "text":
                            content_text += part.get("text", "")

                message_obj: dict[str, Any] = {"role": "assistant"}
                if content_text:
                    message_obj["content"] = content_text
                else:
                    message_obj["content"] = None

                if tool_calls:
                    message_obj["tool_calls"] = tool_calls

                result.append(message_obj)
                continue

            # Handle standard multimodal content (user role usually)
            content_parts: list[dict[str, Any]] = []
            for part in msg.content:
                # Handle dict-style parts (from user input)
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append(
                            {"type": "text", "text": part.get("text", "")}
                        )
                    elif part.get("type") == "image":
                        image_data = part.get("image")
                        media_type = part.get("media_type", "image/png")

                        # Handle URL
                        if isinstance(image_data, str) and image_data.startswith(
                            ("http://", "https://")
                        ):
                            (
                                image_data,
                                media_type,
                            ) = await self._fetch_resource_as_base64(image_data)
                            image_data = f"data:{media_type};base64,{image_data}"
                        elif isinstance(image_data, bytes):
                            image_data = f"data:{media_type};base64,{base64.b64encode(image_data).decode()}"
                        elif isinstance(image_data, str) and not image_data.startswith(
                            "data:"
                        ):
                            # Assume it's base64 data
                            image_data = f"data:{media_type};base64,{image_data}"

                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            }
                        )
                    elif part.get("type") == "file":
                        file_data = part.get("data")
                        media_type = part.get("media_type")

                        # Handle URL
                        if isinstance(file_data, str) and file_data.startswith(
                            ("http://", "https://")
                        ):
                            (
                                file_data,
                                fetched_type,
                            ) = await self._fetch_resource_as_base64(file_data)
                            if not media_type:
                                media_type = fetched_type
                            file_data = f"data:{media_type};base64,{file_data}"
                        elif isinstance(file_data, bytes):
                            file_data = f"data:{media_type};base64,{base64.b64encode(file_data).decode()}"

                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": file_data}}
                        )

                # Handle dataclass-style parts
                elif isinstance(part, TextPart):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    image_data = part.image
                    media_type = getattr(part, "media_type", "image/png")

                    if isinstance(image_data, str) and image_data.startswith(
                        ("http://", "https://")
                    ):
                        image_data, media_type = await self._fetch_resource_as_base64(
                            image_data
                        )
                        image_data = f"data:{media_type};base64,{image_data}"
                    elif isinstance(image_data, bytes):
                        image_data = f"data:{media_type};base64,{base64.b64encode(image_data).decode()}"

                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        }
                    )
                elif isinstance(part, FilePart):
                    file_data = part.data
                    media_type = getattr(part, "media_type", None)

                    if isinstance(file_data, str) and file_data.startswith(
                        ("http://", "https://")
                    ):
                        file_data, fetched_type = await self._fetch_resource_as_base64(
                            file_data
                        )
                        if not media_type:
                            media_type = fetched_type
                        file_data = f"data:{media_type};base64,{file_data}"
                    elif isinstance(file_data, bytes):
                        file_data = f"data:{media_type};base64,{base64.b64encode(file_data).decode()}"

                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": file_data},
                        }
                    )

            result.append({"role": msg.role, "content": content_parts})
        return result

    def _convert_tools(self, tools: ToolSet) -> list[dict[str, Any]]:
        """Convert ToolSet to OpenAI function calling format."""
        result = []
        for name, tool in tools.items():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return result

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using OpenAI API."""
        openai_options = self.get_provider_options(provider_options)

        url = f"{self.base_url}/chat/completions"

        # Convert messages
        converted_messages = await self._convert_messages(messages)

        # Build request parameters
        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            **kwargs,
            **openai_options,
        }

        # Add tools if provided
        if tools:
            request_body["tools"] = self._convert_tools(tools)

        # Use transport for HTTP request
        response = await self.transport.post(url, request_body, headers=self._get_headers())

        # Extract result
        choice = response["choices"][0]
        usage = None
        if "usage" in response:
            usage_data = response["usage"]
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                cached_tokens=usage_data.get("prompt_tokens_details", {}).get(
                    "cached_tokens", 0
                ),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
            for tc in choice["message"]["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                )

        # Build response dict with tool_calls for the execution loop
        response_with_tools = dict(response)
        response_with_tools["tool_calls"] = tool_calls

        return GenerateTextResult(
            text=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            response=response_with_tools,
            provider_metadata={
                "model": response.get("model"),
                "id": response.get("id"),
            },
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text using OpenAI API."""
        openai_options = self.get_provider_options(provider_options)

        url = f"{self.base_url}/chat/completions"

        finish_reason = None
        usage = None
        current_tool_calls: dict[int, dict[str, Any]] = {}

        # Convert messages
        converted_messages = await self._convert_messages(messages)

        # Build request parameters with streaming enabled
        request_body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
            **openai_options,
        }

        # Add tools if provided
        if tools:
            request_body["tools"] = self._convert_tools(tools)

        # Buffer for accumulating partial SSE data
        buffer = b""

        # Use transport for streaming
        async for chunk_bytes in self.transport.stream(url, request_body, headers=self._get_headers()):
            buffer += chunk_bytes

            # Process complete lines
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                chunk = self._parse_sse_json(line)
                if chunk is None:
                    continue

                # Check for usage in the chunk (sent at the end)
                if "usage" in chunk and chunk["usage"]:
                    usage_data = chunk["usage"]
                    usage = Usage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                        cached_tokens=usage_data.get(
                            "prompt_tokens_details", {}
                        ).get("cached_tokens", 0),
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

                    # Handle tool calls
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc["index"]
                            if idx not in current_tool_calls:
                                current_tool_calls[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }

                            if "id" in tc:
                                current_tool_calls[idx]["id"] += tc["id"]

                            if "function" in tc:
                                fn = tc["function"]
                                if "name" in fn:
                                    current_tool_calls[idx]["name"] += fn["name"]
                                if "arguments" in fn:
                                    current_tool_calls[idx]["arguments"] += fn[
                                        "arguments"
                                    ]

        # Process accumulated tool calls
        final_tool_calls = []
        if current_tool_calls:
            # Sort by index to maintain order
            sorted_calls = sorted(current_tool_calls.items(), key=lambda x: x[0])
            for _, call_data in sorted_calls:
                try:
                    arguments = json.loads(call_data["arguments"])
                    final_tool_calls.append(
                        ToolCall(
                            id=call_data["id"],
                            name=call_data["name"],
                            arguments=arguments,
                        )
                    )
                except json.JSONDecodeError:
                    # Incomplete JSON or other error
                    pass

        # Yield final chunk with metadata
        yield StreamChunk(
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=final_tool_calls if final_tool_calls else None,
        )

    async def embed(
        self,
        *,
        model: str,
        value: str,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> EmbedResult:
        """Generate an embedding for a single value using OpenAI API."""
        openai_options = self.get_provider_options(provider_options)

        url = f"{self.base_url}/embeddings"

        request_body: dict[str, Any] = {
            "model": model,
            "input": value,
            **kwargs,
            **openai_options,
        }

        # Use transport for HTTP request
        response = await self.transport.post(url, request_body, headers=self._get_headers())

        # Extract result
        embedding = response["data"][0]["embedding"]
        usage = EmbeddingUsage(tokens=response.get("usage", {}).get("prompt_tokens", 0))

        return EmbedResult(
            value=value,
            embedding=embedding,
            usage=usage,
            provider_metadata={"model": response.get("model")},
        )

    async def embed_many(
        self,
        *,
        model: str,
        values: list[str],
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> EmbedManyResult:
        """Generate embeddings for multiple values using OpenAI API."""
        openai_options = self.get_provider_options(provider_options)

        url = f"{self.base_url}/embeddings"

        request_body: dict[str, Any] = {
            "model": model,
            "input": values,
            **kwargs,
            **openai_options,
        }

        # Use transport for HTTP request
        response = await self.transport.post(url, request_body, headers=self._get_headers())

        # Extract embeddings in order (OpenAI returns them sorted by index)
        data = sorted(response["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data]

        usage = EmbeddingUsage(tokens=response.get("usage", {}).get("prompt_tokens", 0))

        return EmbedManyResult(
            values=values,
            embeddings=embeddings,
            usage=usage,
            provider_metadata={"model": response.get("model")},
        )


# Create the openai namespace instance
openai = _OpenAINamespace()
