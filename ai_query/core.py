"""Core generation functions for ai-query."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterator

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    ReasoningEvent,
    ReasoningPart,
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
    BeforeToolCallEvent,
    AfterToolCallEvent,
    StepStartEvent,
    StepFinishEvent,
    OnStepStart,
    OnStepFinish,
    OnBeforeToolCall,
    OnAfterToolCall,
    OnReasoningEvent,
    EmbedResult,
    EmbedManyResult,
    EmbeddingUsage,
    AbortSignal,
    AbortError,
    ReasoningConfig,
)
from ai_query.model import LanguageModel, EmbeddingModel


def _apply_reasoning(
    model: LanguageModel,
    provider_options: ProviderOptions | None,
    reasoning: ReasoningConfig | None,
) -> ProviderOptions | None:
    if not reasoning:
        return provider_options
    return model.provider.apply_reasoning(
        provider_options,
        reasoning,
        model=model.model_id,
    )


def _normalize_injected_messages(
    messages: list[Message] | list[dict[str, Any]],
) -> list[Message]:
    normalized: list[Message] = []
    for msg in messages:
        if isinstance(msg, Message):
            normalized.append(msg)
        elif isinstance(msg, dict):
            normalized.append(Message.from_dict(msg))
        else:
            raise TypeError(
                "StepControl.inject_messages items must be Message or dict, "
                f"got {type(msg).__name__}"
            )
    return normalized


def _build_initial_messages(
    *,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[Message] | list[dict[str, Any]] | None = None,
) -> list[Message]:
    final_messages: list[Message] = []

    if messages is not None:
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            else:
                final_messages.append(Message.from_dict(msg))

        if system and (not final_messages or final_messages[0].role != "system"):
            final_messages.insert(0, Message(role="system", content=system))
        return final_messages

    if system:
        final_messages.append(Message(role="system", content=system))
    if prompt:
        final_messages.append(Message(role="user", content=prompt))
    return final_messages


def _reasoning_event_payload(event: ReasoningEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": event.provider,
        "kind": event.kind,
    }
    payload.update(event.data)
    return payload


def _append_reasoning_event(
    reasoning_parts: list[ReasoningPart],
    event: ReasoningEvent,
) -> None:
    event_payload = _reasoning_event_payload(event)
    if event.kind in {"signature", "state"}:
        target = None
        if reasoning_parts and reasoning_parts[-1].data.get("provider") == event.provider:
            target = reasoning_parts[-1]
        if target is None:
            target = ReasoningPart(data={"provider": event.provider})
            reasoning_parts.append(target)
        target.data.update(event_payload)
        if event.text:
            target.text += event.text
        return

    if (
        reasoning_parts
        and reasoning_parts[-1].data.get("provider") == event.provider
        and reasoning_parts[-1].data.get("kind") == event.kind
        and {k: v for k, v in reasoning_parts[-1].data.items() if k not in {"provider", "kind"}}
        == event.data
    ):
        reasoning_parts[-1].text += event.text or ""
        reasoning_parts[-1].data.update(event_payload)
        return

    reasoning_parts.append(
        ReasoningPart(
            text=event.text or "",
            data=event_payload,
        )
    )


def _build_assistant_step_content(step: StepResult) -> str | list[Any]:
    assistant_content: list[Any] = []
    assistant_content.extend(step.reasoning_parts)
    if step.text:
        assistant_content.append(TextPart(text=step.text))
    for tool_call in step.tool_calls:
        assistant_content.append(ToolCallPart(tool_call=tool_call))
    return assistant_content if assistant_content else ""


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
    before_tool_call: OnBeforeToolCall | None = None,
    after_tool_call: OnAfterToolCall | None = None,
    provider_options: ProviderOptions | None = None,
    reasoning: ReasoningConfig | None = None,
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate text using an AI model."""
    final_messages = _build_initial_messages(
        prompt=prompt,
        system=system,
        messages=messages,
    )

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
                callback_result = await callback_result
            if callback_result:
                if callback_result.inject_messages:
                    current_messages.extend(
                        _normalize_injected_messages(callback_result.inject_messages)
                    )
                if callback_result.stop:
                    if steps:
                        final_result = GenerateTextResult(
                            text=accumulated_text,
                            steps=steps,
                            finish_reason="stop",
                            usage=total_usage,
                        )
                        break

        result = await model.provider.generate(
            model=model.model_id,
            messages=current_messages,
            tools=tools,
            provider_options=_apply_reasoning(model, provider_options, reasoning),
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
            reasoning_parts=list(result.reasoning_parts),
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
        should_terminate_after_step = False
        for tc in tool_calls:
            before_result = None
            if before_tool_call:
                before_event = BeforeToolCallEvent(
                    step_number=step_number,
                    tool_call=tc,
                    messages=current_messages,
                )
                before_result = before_tool_call(before_event)
                if inspect.isawaitable(before_result):
                    before_result = await before_result

            tool_def = tools.get(tc.name) if tools else None
            if before_result and before_result.block:
                tool_result = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: {before_result.reason or 'Tool execution was blocked'}",
                    is_error=True,
                )
            elif tool_def is None:
                tool_result = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: Unknown tool '{tc.name}'",
                    is_error=True,
                )
            else:
                try:
                    output = await tool_def.run(**tc.arguments)
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=output,
                    )
                except Exception as e:
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=f"Error: {e}",
                        is_error=True,
                    )

            if after_tool_call:
                after_event = AfterToolCallEvent(
                    step_number=step_number,
                    tool_call=tc,
                    tool_result=tool_result,
                    messages=current_messages,
                )
                after_result = after_tool_call(after_event)
                if inspect.isawaitable(after_result):
                    after_result = await after_result
                if after_result:
                    if after_result.result is not None:
                        tool_result.result = after_result.result
                    if after_result.is_error is not None:
                        tool_result.is_error = after_result.is_error
                    if after_result.terminate:
                        should_terminate_after_step = True

            tool_results.append(tool_result)

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

        should_stop = should_terminate_after_step
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

        current_messages.append(
            Message(
                role="assistant",
                content=_build_assistant_step_content(step),
            )
        )

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
    before_tool_call: OnBeforeToolCall | None = None,
    after_tool_call: OnAfterToolCall | None = None,
    on_reasoning_event: OnReasoningEvent | None = None,
    provider_options: ProviderOptions | None = None,
    reasoning: ReasoningConfig | None = None,
    signal: AbortSignal | None = None,
    **kwargs: Any,
) -> TextStreamResult:
    """Stream text from an AI model."""
    final_messages = _build_initial_messages(
        prompt=prompt,
        system=system,
        messages=messages,
    )

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
                    callback_result = await callback_result
                if callback_result:
                    if callback_result.inject_messages:
                        current_messages.extend(
                            _normalize_injected_messages(callback_result.inject_messages)
                        )
                    if callback_result.stop:
                        yield StreamChunk(
                            is_final=True,
                            usage=total_usage,
                            finish_reason="stop",
                        )
                        break

            stream = model.provider.stream(
                model=model.model_id,
                messages=current_messages,
                tools=tools,
                provider_options=_apply_reasoning(model, provider_options, reasoning),
                **kwargs,
            )

            step_text = ""
            step_tool_calls: list[ToolCall] = []
            step_reasoning_parts: list[ReasoningPart] = []
            step_finish_reason = None

            async for chunk in stream:
                if chunk.reasoning_events:
                    for event in chunk.reasoning_events:
                        _append_reasoning_event(step_reasoning_parts, event)
                        if not on_reasoning_event:
                            continue
                        callback_result = on_reasoning_event(event)
                        if inspect.isawaitable(callback_result):
                            await callback_result

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
                reasoning_parts=step_reasoning_parts,
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
            should_terminate_after_step = False
            for tc in step_tool_calls:
                before_result = None
                if before_tool_call:
                    before_event = BeforeToolCallEvent(
                        step_number=step_number,
                        tool_call=tc,
                        messages=current_messages,
                    )
                    before_result = before_tool_call(before_event)
                    if inspect.isawaitable(before_result):
                        before_result = await before_result

                tool_def = tools.get(tc.name) if tools else None
                if before_result and before_result.block:
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=f"Error: {before_result.reason or 'Tool execution was blocked'}",
                        is_error=True,
                    )
                elif tool_def is None:
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=f"Error: Unknown tool '{tc.name}'",
                        is_error=True,
                    )
                else:
                    try:
                        output = await tool_def.run(**tc.arguments)
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result=output,
                        )
                    except Exception as e:
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result=f"Error: {str(e)}",
                            is_error=True,
                        )

                if after_tool_call:
                    after_event = AfterToolCallEvent(
                        step_number=step_number,
                        tool_call=tc,
                        tool_result=tool_result,
                        messages=current_messages,
                    )
                    after_result = after_tool_call(after_event)
                    if inspect.isawaitable(after_result):
                        after_result = await after_result
                    if after_result:
                        if after_result.result is not None:
                            tool_result.result = after_result.result
                        if after_result.is_error is not None:
                            tool_result.is_error = after_result.is_error
                        if after_result.terminate:
                            should_terminate_after_step = True

                tool_results.append(tool_result)

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

            should_stop = should_terminate_after_step
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

            current_messages.append(
                Message(
                    role="assistant",
                    content=_build_assistant_step_content(step),
                )
            )

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
