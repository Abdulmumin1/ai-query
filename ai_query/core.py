"""Core generation functions for ai-query."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterator

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
    TextStreamResult,
    StreamChunk,
    ToolSet,
    ToolCall,
    ToolResult,
    StepResult,
    StopCondition,
    step_count_is,
    StepStartEvent,
    StepFinishEvent,
    OnStepStart,
    OnStepFinish,
    EmbedResult,
    EmbedManyResult,
    EmbeddingUsage,
    AbortSignal,
    AbortError,
)
from ai_query.model import LanguageModel, EmbeddingModel


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
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate text using an AI model."""
    # Build messages list
    final_messages: list[Message] = []

    if messages is not None:
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    stop_conditions: list[StopCondition] = []
    if stop_when is not None:
        if isinstance(stop_when, list):
            stop_conditions = stop_when
        else:
            stop_conditions = [stop_when]

    if not stop_conditions:
        stop_conditions = [step_count_is(1)]

    steps: list[StepResult] = []
    current_messages = list(final_messages)
    total_usage = Usage()
    accumulated_text = ""
    final_result: GenerateTextResult | None = None
    step_number = 0

    if signal:
        signal.throw_if_aborted()

    while True:
        step_number += 1

        if signal and signal.aborted:
            raise AbortError(signal.reason)

        if on_step_start:
            start_event = StepStartEvent(
                step_number=step_number,
                messages=current_messages,
                tools=tools,
            )
            callback_result = on_step_start(start_event)
            if inspect.isawaitable(callback_result):
                await callback_result

        result = await model.provider.generate(
            model=model.model_id,
            messages=current_messages,
            tools=tools,
            provider_options=provider_options,
            **kwargs,
        )

        if result.usage:
            total_usage.input_tokens += result.usage.input_tokens
            total_usage.output_tokens += result.usage.output_tokens
            total_usage.cached_tokens += result.usage.cached_tokens
            total_usage.total_tokens += result.usage.total_tokens

        tool_calls: list[ToolCall] = result.response.get("tool_calls", [])

        step = StepResult(
            text=result.text,
            tool_calls=tool_calls,
            tool_results=[],
            finish_reason=result.finish_reason,
        )

        if result.text:
            accumulated_text += result.text

        if not tool_calls:
            final_result = result
            final_result.usage = total_usage
            steps.append(step)

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

        tool_results: list[ToolResult] = []
        for tc in tool_calls:
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
                    result=f"Error: {e}",
                    is_error=True,
                ))

        step.tool_results = tool_results
        steps.append(step)

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

        should_stop = False
        for condition in stop_conditions:
            cond_result = condition(steps)
            if inspect.isawaitable(cond_result):
                cond_result = await cond_result
            if cond_result:
                should_stop = True
                break

        if should_stop:
            final_result = result
            final_result.usage = total_usage
            break

        assistant_content: list = []
        if result.text:
            assistant_content.append(TextPart(text=result.text))
        for tc in tool_calls:
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

    if final_result:
        final_result.steps = steps
        return final_result

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
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> TextStreamResult:
    """Stream text from an AI model."""
    final_messages: list[Message] = []

    if messages is not None:
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message(role=msg["role"], content=msg["content"]))
    else:
        if system:
            final_messages.append(Message(role="system", content=system))
        if prompt:
            final_messages.append(Message(role="user", content=prompt))

    if not final_messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    stop_conditions: list[StopCondition] = []
    if stop_when is not None:
        if isinstance(stop_when, list):
            stop_conditions = stop_when
        else:
            stop_conditions = [stop_when]

    if not stop_conditions:
        stop_conditions = [step_count_is(1)]

    shared_steps: list[StepResult] = []

    async def _stream_generator() -> AsyncIterator[StreamChunk]:
        nonlocal shared_steps
        steps: list[StepResult] = shared_steps
        current_messages = list(final_messages)
        total_usage = Usage()
        accumulated_text = ""
        step_number = 0

        if signal:
            signal.throw_if_aborted()

        while True:
            step_number += 1

            if signal and signal.aborted:
                raise AbortError(signal.reason)

            if on_step_start:
                start_event = StepStartEvent(
                    step_number=step_number,
                    messages=current_messages,
                    tools=tools,
                )
                callback_result = on_step_start(start_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

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

            async for chunk in stream:
                if chunk.is_final:
                    if chunk.usage:
                        total_usage.input_tokens += chunk.usage.input_tokens
                        total_usage.output_tokens += chunk.usage.output_tokens
                        total_usage.cached_tokens += chunk.usage.cached_tokens
                        total_usage.total_tokens += chunk.usage.total_tokens
                    step_finish_reason = chunk.finish_reason
                    if chunk.tool_calls:
                        step_tool_calls = chunk.tool_calls
                else:
                    if chunk.text:
                        if signal and signal.aborted:
                            raise AbortError(signal.reason)
                        step_text += chunk.text
                        accumulated_text += chunk.text
                        yield StreamChunk(text=chunk.text)

            step = StepResult(
                text=step_text,
                tool_calls=step_tool_calls,
                tool_results=[],
                finish_reason=step_finish_reason,
            )

            if not step_tool_calls:
                steps.append(step)

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

                yield StreamChunk(
                    is_final=True,
                    usage=total_usage,
                    finish_reason=step_finish_reason,
                )
                break

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

            step.tool_results = tool_results
            steps.append(step)

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

    return TextStreamResult(_stream_generator(), steps=shared_steps)


async def embed(
    *,
    model: EmbeddingModel,
    value: str,
    provider_options: ProviderOptions | None = None,
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> EmbedResult:
    """Generate an embedding for a single value."""
    if signal:
        signal.throw_if_aborted()

    if signal:
        embed_task = asyncio.create_task(
            model.provider.embed(
                model=model.model_id,
                value=value,
                provider_options=provider_options,
                **kwargs,
            )
        )
        abort_task = asyncio.create_task(signal.wait())

        done, pending = await asyncio.wait(
            [embed_task, abort_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if abort_task in done:
            raise AbortError(signal.reason)

        return embed_task.result()
    else:
        return await model.provider.embed(
            model=model.model_id,
            value=value,
            provider_options=provider_options,
            **kwargs,
        )


async def embed_many(
    *,
    model: EmbeddingModel,
    values: list[str],
    provider_options: ProviderOptions | None = None,
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> EmbedManyResult:
    """Generate embeddings for multiple values."""
    if signal:
        signal.throw_if_aborted()

    tasks = [
        embed(model=model, value=v, provider_options=provider_options, signal=signal, **kwargs)
        for v in values
    ]
    results = await asyncio.gather(*tasks)

    total_tokens = sum(r.usage.tokens for r in results)
    return EmbedManyResult(
        values=values,
        embeddings=[r.embedding for r in results],
        usage=EmbeddingUsage(tokens=total_tokens),
    )
