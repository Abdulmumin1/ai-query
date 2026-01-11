"""AWS Bedrock provider using boto3 and the Converse API."""

from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, Callable
import asyncio

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None

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
)
from ai_query.model import LanguageModel


# Cached provider instance
_default_provider: "BedrockProvider | None" = None


def bedrock(
    model_id: str,
    *,
    region: str | None = None,
    session: Any | None = None,
) -> LanguageModel:
    """Create an AWS Bedrock language model.

    Requires boto3 to be installed: pip install ai-query[bedrock]

    Args:
        model_id: Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0").
        region: AWS region. To use a custom region.
        session: Optional boto3.Session object. If provided, this session will be used
            to create the client. This allows handling credentials (SSO, AssumeRole, etc.)
            entirely outside the library.

    Returns:
        A LanguageModel instance for use with generate_text().

    Example:
        >>> import boto3
        >>> from ai_query.providers.bedrock import bedrock

        # Default auth
        >>> model = bedrock("anthropic.claude-3-5-sonnet-20241022-v2:0")
        
        # Custom session (e.g. SSO profile)
        >>> session = boto3.Session(profile_name="my-sso-profile")
        >>> model = bedrock("anthropic.claude-3-5-sonnet", session=session)
    """
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required for the Bedrock provider. "
            "Install with: pip install ai-query[bedrock]"
        )

    global _default_provider

    # Create provider with custom settings, or reuse default
    if region or session:
        provider = BedrockProvider(region=region, session=session)
    else:
        if _default_provider is None:
            _default_provider = BedrockProvider()
        provider = _default_provider

    return LanguageModel(provider=provider, model_id=model_id)


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider using boto3 and the Converse API."""

    name = "bedrock"

    def __init__(
        self,
        region: str | None = None,
        session: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize Bedrock provider.

        Args:
            region: AWS region. Falls back to session region or env vars.
            session: Optional boto3.Session object.
            **kwargs: Additional configuration.
        """
        super().__init__(None, **kwargs)

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for the Bedrock provider. "
                "Install with: pip install ai-query[bedrock]"
            )

        self.region = region
        self.session = session
        self._client = None

    def _get_client(self):
        """Get or create the boto3 bedrock-runtime client."""
        if self._client is None:
            config = Config(retries={"max_attempts": 3, "mode": "adaptive"})
            
            # Use provided session or create default
            if self.session:
                # If region provided, override session region
                self._client = self.session.client(
                    "bedrock-runtime",
                    region_name=self.region,  # If None, session uses its own default
                    config=config,
                )
            else:
                # Default session
                region = self.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=region,
                    config=config,
                )

        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Message objects to Bedrock Converse format.

        Returns:
            Tuple of (system_prompt, messages_list).
        """
        system_prompt: str | None = None
        bedrock_messages = []

        for msg in messages:
            # Extract system message
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_prompt = msg.content
                continue

            # Map roles
            if msg.role == "assistant":
                role = "assistant"
            else:
                role = "user"

            if isinstance(msg.content, str):
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": msg.content}]
                })
            else:
                # Handle multimodal/tool content
                content_blocks = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_blocks.append({"text": part.get("text", "")})
                        elif part.get("type") == "image":
                            image_data = part.get("image")
                            media_type = part.get("media_type", "image/png")
                            if isinstance(image_data, bytes):
                                content_blocks.append({
                                    "image": {
                                        "format": media_type.split("/")[-1],
                                        "source": {"bytes": image_data}
                                    }
                                })
                    elif hasattr(part, "type"):
                        if part.type == "tool_call":
                            tc = part.tool_call
                            if tc:
                                content_blocks.append({
                                    "toolUse": {
                                        "toolUseId": tc.id,
                                        "name": tc.name,
                                        "input": tc.arguments,
                                    }
                                })
                        elif part.type == "tool_result":
                            tr = part.tool_result
                            if tr:
                                content_blocks.append({
                                    "toolResult": {
                                        "toolUseId": tr.tool_call_id,
                                        "content": [{"text": str(tr.result)}],
                                        "status": "error" if tr.is_error else "success",
                                    }
                                })
                    elif hasattr(part, "text"):
                        content_blocks.append({"text": part.text})

                if content_blocks:
                    bedrock_messages.append({"role": role, "content": content_blocks})

        return system_prompt, bedrock_messages

    def _convert_tools(self, tools: ToolSet) -> dict[str, Any]:
        """Convert ToolSet to Bedrock tool configuration format."""
        tool_list = []
        for name, tool in tools.items():
            tool_list.append({
                "toolSpec": {
                    "name": name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters},
                }
            })
        return {"tools": tool_list}

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Generate text using AWS Bedrock Converse API.

        Args:
            model: Model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0").
            messages: Conversation messages.
            tools: Optional tool definitions for tool calling.
            provider_options: Bedrock-specific options under "bedrock" key.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            GenerateTextResult with generated text and metadata.
        """
        bedrock_options = self.get_provider_options(provider_options)

        # Convert messages
        system_prompt, bedrock_messages = self._convert_messages(messages)

        # Build inference config from kwargs and bedrock_options
        inference_config: dict[str, Any] = {}
        
        # From kwargs
        if "max_tokens" in kwargs:
            inference_config["maxTokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            inference_config["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs.pop("top_p")
        if "stop_sequences" in kwargs:
            inference_config["stopSequences"] = kwargs.pop("stop_sequences")

        # From bedrock_options (flat structure)
        key_mapping = {
            "max_tokens": "maxTokens",
            "top_p": "topP",
            "stop_sequences": "stopSequences",
        }
        for snake_key, camel_key in key_mapping.items():
            if snake_key in bedrock_options:
                inference_config[camel_key] = bedrock_options.pop(snake_key)
            elif camel_key in bedrock_options:
                inference_config[camel_key] = bedrock_options.pop(camel_key)
        
        # Direct keys
        for key in ["temperature", "maxTokens", "topP", "stopSequences"]:
            if key in bedrock_options:
                inference_config[key] = bedrock_options.pop(key)

        # Build request
        request: dict[str, Any] = {
            "modelId": model,
            "messages": bedrock_messages,
        }

        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        if inference_config:
            request["inferenceConfig"] = inference_config

        if tools:
            request["toolConfig"] = self._convert_tools(tools)

        # Add any additional config from bedrock_options
        if "additionalModelRequestFields" in bedrock_options:
            request["additionalModelRequestFields"] = bedrock_options["additionalModelRequestFields"]

        # Execute in thread pool since boto3 is synchronous
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.converse(**request))

        # Extract result
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text = ""
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=tu.get("input", {}),
                ))

        # Build usage info
        usage = None
        usage_data = response.get("usage", {})
        if usage_data:
            usage = Usage(
                input_tokens=usage_data.get("inputTokens", 0),
                output_tokens=usage_data.get("outputTokens", 0),
                total_tokens=usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0),
            )

        # Build response with tool_calls
        response_with_tools = dict(response)
        response_with_tools["tool_calls"] = tool_calls

        return GenerateTextResult(
            text=text,
            finish_reason=response.get("stopReason"),
            usage=usage,
            response=response_with_tools,
            provider_metadata={"model": model},
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
        """Stream text using AWS Bedrock Converse Stream API.

        Args:
            model: Model ID.
            messages: Conversation messages.
            tools: Optional tool definitions.
            provider_options: Bedrock-specific options.
            **kwargs: Additional params.

        Yields:
            StreamChunk objects with text and final metadata.
        """
        bedrock_options = self.get_provider_options(provider_options)

        # Convert messages
        system_prompt, bedrock_messages = self._convert_messages(messages)

        # Build inference config
        inference_config: dict[str, Any] = {}
        
        if "max_tokens" in kwargs:
            inference_config["maxTokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            inference_config["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs.pop("top_p")
        if "stop_sequences" in kwargs:
            inference_config["stopSequences"] = kwargs.pop("stop_sequences")

        # From bedrock_options
        key_mapping = {
            "max_tokens": "maxTokens",
            "top_p": "topP",
            "stop_sequences": "stopSequences",
        }
        for snake_key, camel_key in key_mapping.items():
            if snake_key in bedrock_options:
                inference_config[camel_key] = bedrock_options.pop(snake_key)
            elif camel_key in bedrock_options:
                inference_config[camel_key] = bedrock_options.pop(camel_key)
        
        for key in ["temperature", "maxTokens", "topP", "stopSequences"]:
            if key in bedrock_options:
                inference_config[key] = bedrock_options.pop(key)

        # Build request
        request: dict[str, Any] = {
            "modelId": model,
            "messages": bedrock_messages,
        }

        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        if inference_config:
            request["inferenceConfig"] = inference_config

        if tools:
            request["toolConfig"] = self._convert_tools(tools)

        if "additionalModelRequestFields" in bedrock_options:
            request["additionalModelRequestFields"] = bedrock_options["additionalModelRequestFields"]

        # Execute streaming call
        client = self._get_client()
        loop = asyncio.get_event_loop()

        # Call converse_stream in executor
        response = await loop.run_in_executor(
            None, lambda: client.converse_stream(**request)
        )

        stream = response.get("stream")
        if not stream:
            yield StreamChunk(is_final=True)
            return

        finish_reason = None
        usage = None
        tool_calls: list[ToolCall] = []
        current_tool_use: dict[str, Any] | None = None

        # Process stream events
        def iterate_stream():
            for event in stream:
                yield event

        for event in iterate_stream():
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield StreamChunk(text=delta["text"])
                elif "toolUse" in delta:
                    # Tool use delta (partial input)
                    if current_tool_use is not None:
                        current_tool_use["input_json"] += delta["toolUse"].get("input", "")

            elif "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    tu = start["toolUse"]
                    current_tool_use = {
                        "id": tu["toolUseId"],
                        "name": tu["name"],
                        "input_json": "",
                    }

            elif "contentBlockStop" in event:
                if current_tool_use is not None:
                    # Parse the accumulated JSON input
                    try:
                        arguments = json.loads(current_tool_use["input_json"]) if current_tool_use["input_json"] else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    tool_calls.append(ToolCall(
                        id=current_tool_use["id"],
                        name=current_tool_use["name"],
                        arguments=arguments,
                    ))
                    current_tool_use = None

            elif "messageStop" in event:
                finish_reason = event["messageStop"].get("stopReason")

            elif "metadata" in event:
                usage_data = event["metadata"].get("usage", {})
                if usage_data:
                    usage = Usage(
                        input_tokens=usage_data.get("inputTokens", 0),
                        output_tokens=usage_data.get("outputTokens", 0),
                        total_tokens=usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0),
                    )

        # Yield final chunk
        yield StreamChunk(
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )
