"""Core generation functions for ai-query."""

from __future__ import annotations

import asyncio
import copy
import contextlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import inspect
import random
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    ReasoningEvent,
    ReasoningPart,
    StreamFinishedEvent,
    StreamReasoningEvent,
    StreamStepFinishedEvent,
    StreamStepStartedEvent,
    TextPart,
    TextDeltaEvent,
    TextStreamEvent,
    ToolCallReadyEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionStartedEvent,
    ToolResultEvent,
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
    OnRetry,
    EmbedResult,
    EmbedManyResult,
    EmbeddingUsage,
    AbortSignal,
    AbortError,
    ReasoningConfig,
    RetryEvent,
    RetryPolicy,
)
from ai_query.transport import HTTPStatusError
from ai_query.model import LanguageModel, EmbeddingModel


_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _copy_usage(usage: Usage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        total_tokens=usage.total_tokens,
    )


def _copy_tool_result(tool_result: ToolResult) -> ToolResult:
    try:
        result = copy.deepcopy(tool_result.result)
    except Exception:
        result = tool_result.result
    return ToolResult(
        tool_call_id=tool_result.tool_call_id,
        tool_name=tool_result.tool_name,
        result=result,
        is_error=tool_result.is_error,
    )


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.status_code in _RETRYABLE_HTTP_STATUS_CODES
    try:
        from aiohttp import ClientError
    except ImportError:
        ClientError = None
    if ClientError is not None and isinstance(exc, ClientError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)):
        return True
    return False


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


async def _run_tool_call(
    *,
    step_number: int,
    index: int,
    tool_call: ToolCall,
    tool_def: Any,
    signal: AbortSignal | None,
    publish_tool_event: Callable[[TextStreamEvent], Awaitable[None]] | None,
) -> ToolResult:
    started_at = time.perf_counter()
    execution_error: str | None = None
    if publish_tool_event:
        await publish_tool_event(ToolExecutionStartedEvent(
            type="tool_execution.started",
            step_number=step_number,
            index=index,
            tool_call=tool_call,
        ))

    try:
        output = await _await_with_abort(tool_def.run(**tool_call.arguments), signal)
        tool_result = ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=output,
        )
    except AbortError as exc:
        if publish_tool_event:
            await publish_tool_event(ToolExecutionFinishedEvent(
                type="tool_execution.finished",
                step_number=step_number,
                index=index,
                tool_call=tool_call,
                tool_result=None,
                duration=time.perf_counter() - started_at,
                error=str(exc),
                aborted=True,
            ))
        raise
    except Exception as exc:
        execution_error = str(exc)
        tool_result = ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=f"Error: {exc}",
            is_error=True,
        )

    if publish_tool_event:
        await publish_tool_event(ToolExecutionFinishedEvent(
            type="tool_execution.finished",
            step_number=step_number,
            index=index,
            tool_call=tool_call,
            tool_result=_copy_tool_result(tool_result),
            duration=time.perf_counter() - started_at,
            error=execution_error,
        ))
    return tool_result


async def _execute_tool_calls(
    *,
    step_number: int,
    tool_calls: list[ToolCall],
    tools: ToolSet | None,
    messages: list[Message],
    before_tool_call: OnBeforeToolCall | None,
    after_tool_call: OnAfterToolCall | None,
    signal: AbortSignal | None,
    publish_tool_event: Callable[[TextStreamEvent], Awaitable[None]] | None = None,
) -> tuple[list[ToolResult], bool]:
    ordered_results: list[ToolResult | None] = []
    runnable_tools: list[tuple[int, ToolCall, Any]] = []

    for index, tool_call in enumerate(tool_calls):
        before_result = None
        if before_tool_call:
            before_event = BeforeToolCallEvent(
                step_number=step_number,
                tool_call=tool_call,
                messages=messages,
            )
            before_result = before_tool_call(before_event)
            if inspect.isawaitable(before_result):
                before_result = await before_result

        tool_def = tools.get(tool_call.name) if tools else None
        if before_result and before_result.block:
            ordered_results.append(
                ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=f"Error: {before_result.reason or 'Tool execution was blocked'}",
                    is_error=True,
                )
            )
        elif tool_def is None:
            ordered_results.append(
                ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=f"Error: Unknown tool '{tool_call.name}'",
                    is_error=True,
                )
            )
        else:
            ordered_results.append(None)
            runnable_tools.append((index, tool_call, tool_def))

    if runnable_tools:
        tool_results = await asyncio.gather(
            *(
                _run_tool_call(
                    step_number=step_number,
                    index=index,
                    tool_call=tool_call,
                    tool_def=tool_def,
                    signal=signal,
                    publish_tool_event=publish_tool_event,
                )
                for index, tool_call, tool_def in runnable_tools
            )
        )
        for (index, _, _), tool_result in zip(runnable_tools, tool_results):
            ordered_results[index] = tool_result

    should_terminate_after_step = False
    final_results: list[ToolResult] = []
    for index, (tool_call, tool_result) in enumerate(zip(tool_calls, ordered_results)):
        if tool_result is None:
            raise RuntimeError(f"Tool call '{tool_call.id}' did not produce a result")

        if after_tool_call:
            after_event = AfterToolCallEvent(
                step_number=step_number,
                tool_call=tool_call,
                tool_result=tool_result,
                messages=messages,
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

        if publish_tool_event:
            await publish_tool_event(ToolResultEvent(
                type="tool_result",
                step_number=step_number,
                index=index,
                tool_call=tool_call,
                tool_result=_copy_tool_result(tool_result),
            ))

        final_results.append(tool_result)

    return final_results, should_terminate_after_step


def _should_retry(exc: Exception, retry: RetryPolicy | None, attempt: int) -> bool:
    if retry is None or retry.max_attempts <= 1 or attempt >= retry.max_attempts:
        return False
    if isinstance(exc, AbortError):
        return False
    if retry.retry_on is not None:
        return retry.retry_on(exc)
    return _is_retryable_exception(exc)


def _retry_after_delay(exc: Exception) -> float | None:
    if not isinstance(exc, HTTPStatusError):
        return None
    retry_after = None
    for key, value in exc.headers.items():
        if key.lower() == "retry-after":
            retry_after = value
            break
    if retry_after is None:
        return None
    try:
        return max(float(retry_after), 0)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0)


def _retry_delay(retry: RetryPolicy, attempt: int, exc: Exception) -> float:
    retry_after = _retry_after_delay(exc)
    if retry_after is not None:
        return min(retry_after, retry.max_delay)
    delay = min(retry.initial_delay * (retry.backoff ** (attempt - 1)), retry.max_delay)
    if retry.jitter and delay > 0:
        delay = random.uniform(0, delay)
    return delay


async def _notify_retry(on_retry: OnRetry | None, event: RetryEvent) -> None:
    if on_retry is None:
        return
    callback_result = on_retry(event)
    if inspect.isawaitable(callback_result):
        await callback_result


async def _sleep_before_retry(delay: float, signal: AbortSignal | None) -> None:
    if delay <= 0:
        return
    if signal is None:
        await asyncio.sleep(delay)
        return
    sleep_task = asyncio.create_task(asyncio.sleep(delay))
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait(
        [sleep_task, abort_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if abort_task in done:
        raise AbortError(signal.reason)


async def _await_with_abort(awaitable: Any, signal: AbortSignal | None) -> Any:
    if signal is None:
        return await awaitable
    if signal.aborted:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise AbortError(signal.reason)
    operation_task = asyncio.ensure_future(awaitable)
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait(
        [operation_task, abort_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if abort_task in done:
        operation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await operation_task
        raise AbortError(signal.reason)
    abort_task.cancel()
    for task in pending:
        task.cancel()
    return operation_task.result()


async def _anext_with_abort(
    iterator: AsyncIterator[StreamChunk],
    signal: AbortSignal | None,
) -> StreamChunk:
    if signal is None:
        return await anext(iterator)
    signal.throw_if_aborted()
    next_task = asyncio.create_task(anext(iterator))
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait(
        [next_task, abort_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if abort_task in done:
        next_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await next_task
        with contextlib.suppress(Exception):
            await iterator.aclose()
        raise AbortError(signal.reason)
    abort_task.cancel()
    for task in pending:
        task.cancel()
    return next_task.result()


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
    retry: RetryPolicy | None = None,
    on_retry: OnRetry | None = None,
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
    latest_usage: Usage | None = None
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
                            usage=_copy_usage(latest_usage) if latest_usage else None,
                        )
                        break

        attempt = 1
        while True:
            try:
                result = await _await_with_abort(
                    model.provider.generate(
                        model=model.model_id,
                        messages=current_messages,
                        tools=tools,
                        provider_options=_apply_reasoning(model, provider_options, reasoning),
                        **kwargs,
                    ),
                    signal,
                )
                break
            except Exception as exc:
                if not _should_retry(exc, retry, attempt):
                    raise
                delay = _retry_delay(retry, attempt, exc)
                await _notify_retry(
                    on_retry,
                    RetryEvent(
                        step_number=step_number,
                        attempt=attempt + 1,
                        max_attempts=retry.max_attempts,
                        delay=delay,
                        error=str(exc),
                        exception=exc,
                    ),
                )
                await _sleep_before_retry(delay, signal)
                attempt += 1

        step_usage = _copy_usage(result.usage) if result.usage else None
        latest_usage = step_usage

        tool_calls: list[ToolCall] = result.response.get("tool_calls", [])

        step = StepResult(
            text=result.text,
            tool_calls=tool_calls,
            tool_results=[],
            reasoning_parts=list(result.reasoning_parts),
            finish_reason=result.finish_reason,
            usage=step_usage,
        )

        if result.text:
            accumulated_text += result.text

        if not tool_calls:
            final_result = result
            final_result.usage = _copy_usage(latest_usage) if latest_usage else None
            steps.append(step)

            if on_step_finish:
                finish_event = StepFinishEvent(
                    step_number=step_number,
                    step=step,
                    text=accumulated_text,
                    usage=_copy_usage(step_usage) if step_usage else None,
                    steps=steps,
                )
                callback_result = on_step_finish(finish_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

            break

        tool_results, should_terminate_after_step = await _execute_tool_calls(
            step_number=step_number,
            tool_calls=tool_calls,
            tools=tools,
            messages=current_messages,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            signal=signal,
        )

        step.tool_results = tool_results
        steps.append(step)

        if on_step_finish:
            finish_event = StepFinishEvent(
                step_number=step_number,
                step=step,
                text=accumulated_text,
                usage=_copy_usage(step_usage) if step_usage else None,
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
            final_result.usage = _copy_usage(latest_usage) if latest_usage else None
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
    retry: RetryPolicy | None = None,
    on_retry: OnRetry | None = None,
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

    async def _stream_generator() -> AsyncIterator[TextStreamEvent]:
        nonlocal shared_steps
        steps: list[StepResult] = shared_steps
        current_messages = list(final_messages)
        latest_usage: Usage | None = None
        accumulated_text = ""
        step_number = 0

        if signal:
            signal.throw_if_aborted()

        while True:
            step_number += 1

            if signal and signal.aborted:
                raise AbortError(signal.reason)

            stop_before_step = False
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
                        stop_before_step = True

            yield StreamStepStartedEvent(
                type="step.started",
                step_number=step_number,
                messages=list(current_messages),
                tools=tools,
            )

            if stop_before_step:
                yield StreamFinishedEvent(
                    type="stream.finished",
                    text=accumulated_text,
                    usage=_copy_usage(latest_usage) if latest_usage else None,
                    finish_reason="stop",
                    steps=list(steps),
                )
                break

            step_text = ""
            step_tool_calls: list[ToolCall] = []
            step_reasoning_parts: list[ReasoningPart] = []
            step_finish_reason = None
            step_usage: Usage | None = None

            attempt = 1
            while True:
                saw_provider_output = False
                try:
                    stream = model.provider.stream(
                        model=model.model_id,
                        messages=current_messages,
                        tools=tools,
                        provider_options=_apply_reasoning(model, provider_options, reasoning),
                        **kwargs,
                    )

                    while True:
                        try:
                            chunk = await _anext_with_abort(stream, signal)
                        except StopAsyncIteration:
                            break
                        saw_provider_output = True
                        if chunk.reasoning_events:
                            for event in chunk.reasoning_events:
                                _append_reasoning_event(step_reasoning_parts, event)
                                if on_reasoning_event:
                                    callback_result = on_reasoning_event(event)
                                    if inspect.isawaitable(callback_result):
                                        await callback_result
                                yield StreamReasoningEvent(
                                    type=f"reasoning.{event.kind}",
                                    event=event,
                                    step_number=step_number,
                                )

                        if chunk.is_final:
                            if chunk.usage:
                                step_usage = _copy_usage(chunk.usage)
                                latest_usage = step_usage
                            step_finish_reason = chunk.finish_reason
                            if chunk.tool_calls:
                                step_tool_calls = chunk.tool_calls
                        else:
                            if chunk.text:
                                if signal and signal.aborted:
                                    raise AbortError(signal.reason)
                                step_text += chunk.text
                                accumulated_text += chunk.text
                                yield TextDeltaEvent(
                                    type="text.delta",
                                    text=chunk.text,
                                    step_number=step_number,
                                )
                    break
                except Exception as exc:
                    if saw_provider_output or not _should_retry(exc, retry, attempt):
                        raise
                    delay = _retry_delay(retry, attempt, exc)
                    await _notify_retry(
                        on_retry,
                        RetryEvent(
                            step_number=step_number,
                            attempt=attempt + 1,
                            max_attempts=retry.max_attempts,
                            delay=delay,
                            error=str(exc),
                            exception=exc,
                        ),
                    )
                    await _sleep_before_retry(delay, signal)
                    attempt += 1

            step = StepResult(
                text=step_text,
                tool_calls=step_tool_calls,
                tool_results=[],
                reasoning_parts=step_reasoning_parts,
                finish_reason=step_finish_reason,
                usage=step_usage,
            )

            if not step_tool_calls:
                steps.append(step)

                if on_step_finish:
                    finish_event = StepFinishEvent(
                        step_number=step_number,
                        step=step,
                        text=accumulated_text,
                        usage=_copy_usage(step_usage) if step_usage else None,
                        steps=steps,
                    )
                    callback_result = on_step_finish(finish_event)
                    if inspect.isawaitable(callback_result):
                        await callback_result

                yield StreamStepFinishedEvent(
                    type="step.finished",
                    step_number=step_number,
                    step=step,
                    text=accumulated_text,
                    usage=_copy_usage(step_usage) if step_usage else None,
                    steps=list(steps),
                )
                yield StreamFinishedEvent(
                    type="stream.finished",
                    text=accumulated_text,
                    usage=_copy_usage(latest_usage) if latest_usage else None,
                    finish_reason=step_finish_reason,
                    steps=list(steps),
                )
                break

            for index, tool_call in enumerate(step_tool_calls):
                yield ToolCallReadyEvent(
                    type="tool_call.ready",
                    step_number=step_number,
                    index=index,
                    tool_call=tool_call,
                )

            tool_event_queue: asyncio.Queue[TextStreamEvent | None] = asyncio.Queue()

            async def publish_tool_event(event: TextStreamEvent) -> None:
                await tool_event_queue.put(event)

            async def execute_tools() -> tuple[list[ToolResult], bool]:
                try:
                    return await _execute_tool_calls(
                        step_number=step_number,
                        tool_calls=step_tool_calls,
                        tools=tools,
                        messages=current_messages,
                        before_tool_call=before_tool_call,
                        after_tool_call=after_tool_call,
                        signal=signal,
                        publish_tool_event=publish_tool_event,
                    )
                finally:
                    await tool_event_queue.put(None)

            execution_task = asyncio.create_task(execute_tools())
            try:
                while True:
                    tool_event = await tool_event_queue.get()
                    if tool_event is None:
                        break
                    yield tool_event
                tool_results, should_terminate_after_step = await execution_task
            finally:
                if not execution_task.done():
                    execution_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await execution_task

            step.tool_results = tool_results
            steps.append(step)

            if on_step_finish:
                finish_event = StepFinishEvent(
                    step_number=step_number,
                    step=step,
                    text=accumulated_text,
                    usage=_copy_usage(step_usage) if step_usage else None,
                    steps=steps,
                )
                callback_result = on_step_finish(finish_event)
                if inspect.isawaitable(callback_result):
                    await callback_result

            yield StreamStepFinishedEvent(
                type="step.finished",
                step_number=step_number,
                step=step,
                text=accumulated_text,
                usage=_copy_usage(step_usage) if step_usage else None,
                steps=list(steps),
            )

            should_stop = should_terminate_after_step
            for condition in stop_conditions:
                cond_result = condition(steps)
                if inspect.isawaitable(cond_result):
                    cond_result = await cond_result
                if cond_result:
                    should_stop = True
                    break

            if should_stop:
                yield StreamFinishedEvent(
                    type="stream.finished",
                    text=accumulated_text,
                    usage=_copy_usage(latest_usage) if latest_usage else None,
                    finish_reason=step_finish_reason,
                    steps=list(steps),
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
        return await _await_with_abort(
            model.provider.embed(
                model=model.model_id,
                value=value,
                provider_options=provider_options,
                **kwargs,
            ),
            signal,
        )
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
