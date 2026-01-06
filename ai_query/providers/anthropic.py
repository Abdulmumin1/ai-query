"""Anthropic provider adapter using direct HTTP API."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
import json

from ai_query.providers.base import BaseProvider
from ai_query.types import GenerateTextResult, Message, ProviderOptions, Usage, StreamChunk, ToolSet, ToolCall
from ai_query.model import LanguageModel

import aiohttp


# Cached provider instance
_default_provider: AnthropicProvider | None = None


def anthropic(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LanguageModel:
    """Create an Anthropic language model.

    Args:
        model_id: The model identifier (e.g., "claude-sonnet-4-20250514", "claude-3-opus-20240229").
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        base_url: Custom base URL for API requests.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> from ai_query import generate_text, anthropic
        >>> result = await generate_text(
        ...     model=anthropic("claude-sonnet-4-20250514"),
        ...     prompt="Hello!"
        ... )
    """
    global _default_provider

    # Create provider with custom settings, or reuse default
    if api_key or base_url:
        provider = AnthropicProvider(api_key=api_key, base_url=base_url)
    else:
        if _default_provider is None:
            _default_provider = AnthropicProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)


class AnthropicProvider(BaseProvider):
    """Anthropic provider adapter using direct HTTP API."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            **kwargs: Additional configuration (base_url, etc.).
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = kwargs.get("base_url", "https://api.anthropic.com")

    async def _convert_messages(
        self, messages: list[Message], session: aiohttp.ClientSession
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

            # Map 'tool' role to 'user' for Anthropic as they handle tool results in user messages
            role = "user" if msg.role == "tool" else msg.role

            content_parts = []
            if isinstance(msg.content, str):
                content_parts.append({"type": "text", "text": msg.content})
            else:
                # Handle multimodal and tool content
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
                            elif isinstance(image_data, bytes):
                                import base64
                                image_data = base64.b64encode(image_data).decode()

                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
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

                            content_parts.append({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type or "application/pdf",
                                    "data": file_data,
                                },
                            })
                        elif part.get("type") == "tool_call":
                            tc = part.get("tool_call")
                            if tc:
                                content_parts.append({
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.name,
                                    "input": tc.arguments,
                                })
                        elif part.get("type") == "tool_result":
                            tr = part.get("tool_result")
                            if tr:
                                content_parts.append({
                                    "type": "tool_result",
                                    "tool_use_id": tr.tool_call_id,
                                    "content": str(tr.result),
                                    "is_error": tr.is_error,
                                })

                    # Handle dataclass-style parts
                    elif hasattr(part, "text"):
                        content_parts.append({"type": "text", "text": part.text})
                    elif hasattr(part, "image"):
                        image_data = part.image
                        media_type = getattr(part, "media_type", "image/png")

                        # Handle URL
                        if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                            image_data, media_type = await self._fetch_resource_as_base64(image_data, session)
                        elif isinstance(image_data, bytes):
                            import base64
                            image_data = base64.b64encode(image_data).decode()

                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        })
                    elif hasattr(part, "data"): # FilePart
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

                        content_parts.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type or "application/pdf",
                                "data": file_data,
                            },
                        })
                    elif hasattr(part, "type") and part.type == "tool_call":
                        tc = part.tool_call
                        if tc:
                            content_parts.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            })
                    elif hasattr(part, "type") and part.type == "tool_result":
                        tr = part.tool_result
                        if tr:
                            content_parts.append({
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": str(tr.result),
                                "is_error": tr.is_error,
                            })

            result.append({"role": role, "content": content_parts})

        return system_prompt, result

    def _convert_tools(self, tools: ToolSet) -> list[dict[str, Any]]:
        """Convert ToolSet to Anthropic tool format."""
        result = []
        for name, tool in tools.items():
            result.append({
                "name": name,
                "description": tool.description,
                "input_schema": tool.parameters,
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
        """Generate text using Anthropic API.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-3-opus-20240229").
            messages: Conversation messages.
            provider_options: Anthropic-specific options under "anthropic" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult with generated text and metadata.
        """
        anthropic_options = self.get_provider_options(provider_options)

        async with aiohttp.ClientSession() as session:
            # Convert messages and extract system prompt
            system_prompt, converted_messages = await self._convert_messages(messages, session)

            # Build request body
            request_body: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs,
                **anthropic_options,
            }

            if system_prompt:
                request_body["system"] = system_prompt

            # Add tools if provided
            if tools:
                request_body["tools"] = self._convert_tools(tools)

            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            url = f"{self.base_url}/v1/messages"

            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Anthropic API error ({resp.status}): {error_text}")
                response = await resp.json()

        # Extract text and tool calls from response
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id"),
                    name=block.get("name"),
                    arguments=block.get("input", {}),
                ))

        # Build usage info
        usage_data = response.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cached_tokens = usage_data.get("cache_read_input_tokens", 0)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        # Build response dict with tool_calls for the execution loop
        response_with_tools = dict(response)
        response_with_tools["tool_calls"] = tool_calls

        return GenerateTextResult(
            text=text,
            finish_reason=response.get("stop_reason"),
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
        """Stream text using Anthropic API.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-3-opus-20240229").
            messages: Conversation messages.
            tools: Optional tool definitions for tool calling.
            provider_options: Anthropic-specific options under "anthropic" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Yields:
            StreamChunk objects with text and final metadata.
        """
        anthropic_options = self.get_provider_options(provider_options)

        async with aiohttp.ClientSession() as session:
            # Convert messages and extract system prompt
            system_prompt, converted_messages = await self._convert_messages(messages, session)

            # Build request body with streaming enabled
            request_body: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": True,
                **kwargs,
                **anthropic_options,
            }

            if system_prompt:
                request_body["system"] = system_prompt

            # Add tools if provided
            if tools:
                request_body["tools"] = self._convert_tools(tools)

            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            url = f"{self.base_url}/v1/messages"

            finish_reason = None
            usage = None
            tool_calls: list[ToolCall] = []
            current_tool_call: dict[str, Any] | None = None

            async with session.post(url, headers=headers, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Anthropic API error ({resp.status}): {error_text}")

                # Process SSE stream
                async for line in resp.content:
                    # Anthropic uses "event:" and "data:" lines
                    chunk = self._parse_sse_json(line)
                    if chunk is None:
                        continue

                    event_type = chunk.get("type")

                    # content_block_start: start of a block (text or tool_use)
                    if event_type == "content_block_start":
                        index = chunk.get("index")
                        content_block = chunk.get("content_block", {})
                        if content_block.get("type") == "tool_use":
                            current_tool_call = {
                                "index": index,
                                "id": content_block.get("id"),
                                "name": content_block.get("name"),
                                "arguments_json": ""
                            }

                    # content_block_delta contains the text chunks or json fragments
                    elif event_type == "content_block_delta":
                        index = chunk.get("index")
                        delta = chunk.get("delta", {})

                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield StreamChunk(text=text)

                        elif delta.get("type") == "input_json_delta":
                            if current_tool_call and current_tool_call["index"] == index:
                                current_tool_call["arguments_json"] += delta.get("partial_json", "")

                    # content_block_stop: end of a block
                    elif event_type == "content_block_stop":
                        index = chunk.get("index")
                        if current_tool_call and current_tool_call["index"] == index:
                            # Parse collected JSON and add to tool_calls
                            try:
                                arguments = json.loads(current_tool_call["arguments_json"])
                                tool_calls.append(ToolCall(
                                    id=current_tool_call["id"],
                                    name=current_tool_call["name"],
                                    arguments=arguments
                                ))
                            except json.JSONDecodeError:
                                # Handle parsing error gracefully
                                pass
                            current_tool_call = None

                    # message_delta contains stop_reason
                    elif event_type == "message_delta":
                        delta = chunk.get("delta", {})
                        if delta.get("stop_reason"):
                            finish_reason = delta["stop_reason"]
                        # Usage is in the message_delta
                        usage_data = chunk.get("usage", {})
                        if usage_data:
                            output_tokens = usage_data.get("output_tokens", 0)
                            # We need input tokens from message_start
                            if usage:
                                usage.output_tokens = output_tokens
                                usage.total_tokens = usage.input_tokens + output_tokens

                    # message_start contains input token count
                    elif event_type == "message_start":
                        message = chunk.get("message", {})
                        usage_data = message.get("usage", {})
                        if usage_data:
                            usage = Usage(
                                input_tokens=usage_data.get("input_tokens", 0),
                                output_tokens=0,
                                cached_tokens=usage_data.get("cache_read_input_tokens", 0),
                                total_tokens=usage_data.get("input_tokens", 0),
                            )

        # Yield final chunk with metadata
        yield StreamChunk(
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None
        )
