"""Main entry point for ai-query library."""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    TextPart,
    ImagePart,
    FilePart,
    Usage,
    TextStreamResult,
    StreamChunk,
    # Tool types
    Tool,
    tool,
    Field,
    ToolSet,
    ToolCall,
    ToolResult,
    ToolCallPart,
    ToolResultPart,
    # Stop conditions
    StepResult,
    StopCondition,
    step_count_is,
    has_tool_call,
    # Step callbacks
    StepStartEvent,
    StepFinishEvent,
    OnStepStart,
    OnStepFinish,
)
from ai_query.model import LanguageModel
from ai_query.providers.base import BaseProvider
from ai_query.providers.openai import OpenAIProvider, openai
from ai_query.providers.anthropic import AnthropicProvider, anthropic
from ai_query.providers.google import GoogleProvider, google
from ai_query.providers.openrouter import openrouter
from ai_query.providers.deepseek import deepseek
from ai_query.providers.groq import groq
from ai_query.mcp import (
    mcp,
    mcp_sse,
    mcp_http,
    connect_mcp,
    connect_mcp_sse,
    connect_mcp_http,
    merge_tools,
    MCPServer,
    MCPClient,
)


async def generate_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
    tools: ToolSet | None = None,
    stop_when: StopCondition | list[StopCondition] | None = None,
    on_step_start: OnStepStart | None = None,
    on_step_finish: OnStepFinish | None = None,
    provider_options: ProviderOptions | None = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate text using an AI model.

    This is the main function for text generation. It supports three input modes:
    1. Simple prompt: Just pass `prompt` for quick queries
    2. System + prompt: Pass `system` and `prompt` for guided generation
    3. Full messages: Pass `messages` for complete conversation control

    Args:
        model: A LanguageModel instance created by a provider function
            (e.g., openai("gpt-4"), anthropic("claude-sonnet-4-20250514"), google("gemini-2.0-flash")).
        prompt: Simple text prompt (mutually exclusive with messages).
        system: System prompt to guide model behavior.
        messages: Full conversation history as Message objects or dicts.
        provider_options: Provider-specific options.
            Example: {"google": {"safety_settings": {...}}}
        **kwargs: Additional parameters (max_tokens, temperature, etc.).

    Returns:
        GenerateTextResult containing:
            - text: The generated text
            - finish_reason: Why generation stopped
            - usage: Token usage statistics
            - response: Raw response data
            - provider_metadata: Provider-specific metadata

    Examples:
        Simple prompt:
        >>> from ai_query import generate_text, openai
        >>> result = await generate_text(
        ...     model=openai("gpt-4"),
        ...     prompt="What is the capital of France?"
        ... )
        >>> print(result.text)

        With system prompt:
        >>> from ai_query import generate_text, anthropic
        >>> result = await generate_text(
        ...     model=anthropic("claude-sonnet-4-20250514"),
        ...     system="You are a helpful assistant.",
        ...     prompt="Explain quantum computing simply."
        ... )

        Full conversation:
        >>> from ai_query import generate_text, google
        >>> result = await generate_text(
        ...     model=google("gemini-2.0-flash"),
        ...     messages=[
        ...         {"role": "system", "content": "You are a poet."},
        ...         {"role": "user", "content": "Write a haiku about coding."}
        ...     ]
        ... )

        With provider options:
        >>> result = await generate_text(
        ...     model=google("gemini-2.0-flash"),
        ...     prompt="Tell me a story",
        ...     provider_options={
        ...         "google": {
        ...             "safety_settings": {"HARM_CATEGORY_VIOLENCE": "BLOCK_NONE"}
        ...         }
        ...     }
        ... )
    """
    # Build messages list
    final_messages: list[Message] = []

    if messages is not None:
        # Convert dict messages to Message objects if needed
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        # Build from prompt and system
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")



    # Normalize stop_when to a list
    stop_conditions: list[StopCondition] = []
    if stop_when is not None:
        if isinstance(stop_when, list):
            stop_conditions = stop_when
        else:
            stop_conditions = [stop_when]

    # If no stop condition provided, default to step_count_is(1)
    if not stop_conditions:
        stop_conditions = [step_count_is(1)]

    # Execution loop
    steps: list[StepResult] = []
    current_messages = list(final_messages)
    total_usage = Usage()
    accumulated_text = ""
    final_result: GenerateTextResult | None = None
    step_number = 0

    while True:
        step_number += 1

        # Call on_step_start callback
        if on_step_start:
            start_event = StepStartEvent(
                step_number=step_number,
                messages=current_messages,
                tools=tools,
            )
            callback_result = on_step_start(start_event)
            if inspect.isawaitable(callback_result):
                await callback_result

        # Generate with tools
        result = await model.provider.generate(
            model=model.model_id,
            messages=current_messages,
            tools=tools,
            provider_options=provider_options,
            **kwargs,
        )

        # Accumulate usage
        if result.usage:
            total_usage.input_tokens += result.usage.input_tokens
            total_usage.output_tokens += result.usage.output_tokens
            total_usage.cached_tokens += result.usage.cached_tokens
            total_usage.total_tokens += result.usage.total_tokens

        # Extract tool calls from the response
        tool_calls: list[ToolCall] = result.response.get("tool_calls", [])

        # Create step result
        step = StepResult(
            text=result.text,
            tool_calls=tool_calls,
            tool_results=[],
            finish_reason=result.finish_reason,
        )

        # Accumulate text
        if result.text:
            accumulated_text += result.text

        # If no tool calls, we're done
        if not tool_calls:
            final_result = result
            final_result.usage = total_usage
            steps.append(step)

            # Call on_step_finish callback
            if on_step_finish:
                finish_event = StepFinishEvent(
                    step_number=step_number,
                    step=step,
                    text=accumulated_text,
                    usage=total_usage,
                    steps=steps,
                )
                callback_result = on_step_finish(finish_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

            break

        # Execute tools
        tool_results: list[ToolResult] = []
        for tc in tool_calls:
            tool_def = tools.get(tc.name)
            if tool_def is None:
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: Unknown tool '{tc.name}'",
                    is_error=True,
                ))
                continue

            try:
                output = await tool_def.run(**tc.arguments)
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=output,
                ))
            except Exception as e:
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: {e}",
                    is_error=True,
                ))

        step.tool_results = tool_results
        steps.append(step)

        # Call on_step_finish callback
        if on_step_finish:
            finish_event = StepFinishEvent(
                step_number=step_number,
                step=step,
                text=accumulated_text,
                usage=total_usage,
                steps=steps,
            )
            callback_result = on_step_finish(finish_event)
            if inspect.isawaitable(callback_result):
                await callback_result

        # Check stop conditions
        should_stop = False
        for condition in stop_conditions:
            cond_result = condition(steps)
            if inspect.isawaitable(cond_result):
                cond_result = await cond_result
            if cond_result:
                should_stop = True
                break

        if should_stop:
            # Return the last result with accumulated usage
            final_result = result
            final_result.usage = total_usage
            break

        # Build messages for next iteration
        # Add assistant message with tool calls (include both text and tool call parts)
        assistant_content: list = []
        if result.text:
            assistant_content.append(TextPart(text=result.text))
        for tc in tool_calls:
            assistant_content.append(ToolCallPart(tool_call=tc))

        current_messages.append(Message(
            role="assistant",
            content=assistant_content if assistant_content else "",
        ))

        # Add tool results
        tool_result_parts = [ToolResultPart(tool_result=tr) for tr in tool_results]
        current_messages.append(Message(
            role="tool",
            content=tool_result_parts,
        ))

    # Store steps in the result for access
    if final_result:
        final_result.steps = steps
        return final_result

    # Fallback (shouldn't reach here)
    raise RuntimeError("Generation loop ended without a result")


def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
    tools: ToolSet | None = None,
    stop_when: StopCondition | list[StopCondition] | None = None,
    on_step_start: OnStepStart | None = None,
    on_step_finish: OnStepFinish | None = None,
    provider_options: ProviderOptions | None = None,
    **kwargs: Any,
) -> TextStreamResult:
    """Stream text from an AI model.

    Returns a TextStreamResult object that provides both the text stream
    and metadata (usage, finish_reason) after streaming completes.

    Args:
        model: A LanguageModel instance created by a provider function
            (e.g., openai("gpt-4"), anthropic("claude-sonnet-4-20250514"), google("gemini-2.0-flash")).
        prompt: Simple text prompt (mutually exclusive with messages).
        system: System prompt to guide model behavior.
        messages: Full conversation history as Message objects or dicts.
        tools: Optional tool definitions for tool calling.
        stop_when: Condition(s) to stop the generation loop.
        provider_options: Provider-specific options.
        **kwargs: Additional parameters (max_tokens, temperature, etc.).

    Returns:
        TextStreamResult with:
            - text_stream: AsyncIterator yielding text chunks
            - text: Awaitable for full text after completion
            - usage: Awaitable for Usage stats after completion
            - finish_reason: Awaitable for finish reason after completion

    Examples:
        Simple streaming (direct iteration):
        >>> async for chunk in stream_text(model=openai("gpt-4"), prompt="Hi"):
        ...     print(chunk, end="", flush=True)

        With usage access:
        >>> result = stream_text(model=google("gemini-2.0-flash"), prompt="Hi")
        >>> async for chunk in result.text_stream:
        ...     print(chunk, end="", flush=True)
        >>> usage = await result.usage
        >>> print(f"Tokens: {usage.total_tokens}")
    """
    # Build messages list
    final_messages: list[Message] = []

    if messages is not None:
        # Convert dict messages to Message objects if needed
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        # Build from prompt and system
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")



    # Normalize stop_when to a list
    stop_conditions: list[StopCondition] = []
    if stop_when is not None:
        if isinstance(stop_when, list):
            stop_conditions = stop_when
        else:
            stop_conditions = [stop_when]

    # If no stop condition provided, default to step_count_is(1)
    if not stop_conditions:
        stop_conditions = [step_count_is(1)]
    # Shared steps list - populated by generator, accessible from result
    shared_steps: list[StepResult] = []

    async def _stream_generator() -> AsyncIterator[StreamChunk]:
        """Generator that handles the tool execution loop."""
        nonlocal shared_steps
        steps: list[StepResult] = shared_steps  # Use the shared list
        current_messages = list(final_messages)
        total_usage = Usage()
        accumulated_text = ""
        step_number = 0

        while True:
            step_number += 1

            # Call on_step_start callback
            if on_step_start:
                start_event = StepStartEvent(
                    step_number=step_number,
                    messages=current_messages,
                    tools=tools,
                )
                callback_result = on_step_start(start_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

            # Stream from provider
            stream = model.provider.stream(
                model=model.model_id,
                messages=current_messages,
                tools=tools,
                provider_options=provider_options,
                **kwargs,
            )

            step_text = ""
            step_tool_calls: list[ToolCall] = []
            step_finish_reason = None

            # Consume provider stream
            async for chunk in stream:
                if chunk.is_final:
                    # Accumulate usage
                    if chunk.usage:
                        total_usage.input_tokens += chunk.usage.input_tokens
                        total_usage.output_tokens += chunk.usage.output_tokens
                        total_usage.cached_tokens += chunk.usage.cached_tokens
                        total_usage.total_tokens += chunk.usage.total_tokens
                    step_finish_reason = chunk.finish_reason
                    if chunk.tool_calls:
                        step_tool_calls = chunk.tool_calls
                else:
                    # Yield text chunk
                    if chunk.text:
                        step_text += chunk.text
                        accumulated_text += chunk.text
                        yield StreamChunk(text=chunk.text)

            # Create step result
            step = StepResult(
                text=step_text,
                tool_calls=step_tool_calls,
                tool_results=[],
                finish_reason=step_finish_reason,
            )

            # If no tool calls, we're done
            if not step_tool_calls:
                steps.append(step)

                # Call on_step_finish callback
                if on_step_finish:
                    finish_event = StepFinishEvent(
                        step_number=step_number,
                        step=step,
                        text=accumulated_text,
                        usage=total_usage,
                        steps=steps,
                    )
                    callback_result = on_step_finish(finish_event)
                    if inspect.isawaitable(callback_result):
                        await callback_result

                # Yield final chunk with usage
                yield StreamChunk(
                    is_final=True,
                    usage=total_usage,
                    finish_reason=step_finish_reason,
                )
                break

            # Execute tools
            tool_results: list[ToolResult] = []
            for tc in step_tool_calls:
                tool_def = tools.get(tc.name) if tools else None
                if tool_def is None:
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=f"Error: Unknown tool '{tc.name}'",
                        is_error=True,
                    ))
                    continue

                try:
                    output = await tool_def.run(**tc.arguments)
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=output,
                    ))
                except Exception as e:
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=f"Error: {str(e)}",
                        is_error=True,
                    ))

            # Update step with tool results
            step.tool_results = tool_results
            steps.append(step)

            # Call on_step_finish callback
            if on_step_finish:
                finish_event = StepFinishEvent(
                    step_number=step_number,
                    step=step,
                    text=accumulated_text,
                    usage=total_usage,
                    steps=steps,
                )
                callback_result = on_step_finish(finish_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

            # Check stop conditions
            should_stop = False
            for condition in stop_conditions:
                cond_result = condition(steps)
                if inspect.isawaitable(cond_result):
                    cond_result = await cond_result
                if cond_result:
                    should_stop = True
                    break

            if should_stop:
                yield StreamChunk(
                    is_final=True,
                    usage=total_usage,
                    finish_reason=step_finish_reason,
                )
                break

            # Update messages for next iteration
            assistant_content: list = []
            if step_text:
                assistant_content.append(TextPart(text=step_text))
            for tc in step_tool_calls:
                assistant_content.append(ToolCallPart(tool_call=tc))

            current_messages.append(Message(
                role="assistant",
                content=assistant_content if assistant_content else "",
            ))

            tool_result_parts = [ToolResultPart(tool_result=tr) for tr in tool_results]
            current_messages.append(Message(
                role="tool",
                content=tool_result_parts,
            ))

    # Return TextStreamResult wrapping the generator with shared steps
    return TextStreamResult(_stream_generator(), steps=shared_steps)


__all__ = [
    # Main functions
    "generate_text",
    "stream_text",
    # Provider factory functions
    "openai",
    "anthropic",
    "google",
    "openrouter",
    "deepseek",
    "llama",
    "xai",
    "groq",
    # Types
    "LanguageModel",
    "GenerateTextResult",
    "TextStreamResult",
    "Message",
    "ProviderOptions",
    "TextPart",
    "ImagePart",
    "FilePart",
    "Usage",
    # Tool types
    "Tool",
    "tool",
    "Field",
    "ToolSet",
    "ToolCall",
    "ToolResult",
    # Stop conditions
    "StepResult",
    "StopCondition",
    "step_count_is",
    "has_tool_call",
    # Step callbacks
    "StepStartEvent",
    "StepFinishEvent",
    "OnStepStart",
    "OnStepFinish",
    # Base class for custom providers
    "BaseProvider",
    # Built-in provider classes (for advanced usage)
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    # MCP support
    "mcp",
    "mcp_sse",
    "mcp_http",
    "connect_mcp",
    "connect_mcp_sse",
    "connect_mcp_http",
    "merge_tools",
    "MCPServer",
    "MCPClient",
]

