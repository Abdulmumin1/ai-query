"""OpenAI provider adapter using direct HTTP API."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
import json
import base64

import aiohttp

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage, StreamChunk, ToolSet, ToolCall, ToolCallPart, ToolResultPart
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
                continue

            # Handle list content
            # Special handling for tool messages (OpenAI requires one message per tool result)
            if msg.role == "tool":
                for part in msg.content:
                    if hasattr(part, "type") and part.type == "tool_result":
                        tr = part.tool_result
                        if tr:
                            result.append({
                                "role": "tool",
                                "tool_call_id": tr.tool_call_id,
                                "content": str(tr.result), # Ensure string content
                            })
                    elif isinstance(part, dict) and part.get("type") == "tool_result":
                         # Handle dict format if passed directly
                         tr = part.get("tool_result")
                         if tr:
                             result.append({
                                "role": "tool",
                                "tool_call_id": tr.tool_call_id,
                                "content": str(tr.result),
                             })
                continue

            # Special handling for assistant messages with tool calls
            if msg.role == "assistant":
                content_text = ""
                tool_calls = []

                for part in msg.content:
                    if hasattr(part, "type") and part.type == "tool_call":
                        tc = part.tool_call
                        if tc:
                            tool_calls.append({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                }
                            })
                    elif hasattr(part, "text"):
                        content_text += part.text
                    elif isinstance(part, dict):
                        if part.get("type") == "tool_call":
                            tc = part.get("tool_call")
                            if tc:
                                tool_calls.append({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments),
                                    }
                                })
                        elif part.get("type") == "text":
                            content_text += part.get("text", "")

                message_obj = {"role": "assistant"}
                if content_text:
                    message_obj["content"] = content_text
                else:
                    message_obj["content"] = None

                if tool_calls:
                    message_obj["tool_calls"] = tool_calls

                result.append(message_obj)
                continue

            # Handle standard multimodal content (user role usually)
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

    def _convert_tools(self, tools: ToolSet) -> list[dict[str, Any]]:
        """Convert ToolSet to OpenAI function calling format."""
        result = []
        for name, tool in tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
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

            # Add tools if provided
            if tools:
                request_body["tools"] = self._convert_tools(tools)

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

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
            for tc in choice["message"]["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                ))

        # Build response dict with tool_calls for the execution loop
        response_with_tools = dict(response)
        response_with_tools["tool_calls"] = tool_calls

        return GenerateTextResult(
            text=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            response=response_with_tools,
            provider_metadata={"model": response.get("model"), "id": response.get("id")},
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
        """Stream text using OpenAI API.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o").
            messages: Conversation messages.
            tools: Optional tool definitions for tool calling.
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
        current_tool_calls: dict[int, dict[str, Any]] = {}

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

            # Add tools if provided
            if tools:
                request_body["tools"] = self._convert_tools(tools)

            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error ({resp.status}): {error_text}")

                # Process SSE stream
                async for line in resp.content:
                    chunk = self._parse_sse_json(line)
                    if chunk is None:
                        continue

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

                        # Handle tool calls
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc in delta["tool_calls"]:
                                idx = tc["index"]
                                if idx not in current_tool_calls:
                                    current_tool_calls[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": ""
                                    }

                                if "id" in tc:
                                    current_tool_calls[idx]["id"] += tc["id"]

                                if "function" in tc:
                                    fn = tc["function"]
                                    if "name" in fn:
                                        current_tool_calls[idx]["name"] += fn["name"]
                                    if "arguments" in fn:
                                        current_tool_calls[idx]["arguments"] += fn["arguments"]

        # Process accumulated tool calls
        final_tool_calls = []
        if current_tool_calls:
            # Sort by index to maintain order
            sorted_calls = sorted(current_tool_calls.items(), key=lambda x: x[0])
            for _, call_data in sorted_calls:
                try:
                    arguments = json.loads(call_data["arguments"])
                    final_tool_calls.append(ToolCall(
                        id=call_data["id"],
                        name=call_data["name"],
                        arguments=arguments
                    ))
                except json.JSONDecodeError:
                    # Incomplete JSON or other error
                    pass

        # Yield final chunk with metadata
        yield StreamChunk(
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=final_tool_calls if final_tool_calls else None
        )
