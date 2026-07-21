"""Deterministic provider for tests, examples, and local agent harnesses."""

from __future__ import annotations

import copy
import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Literal

from ai_query.model import LanguageModel
from ai_query.providers.base import BaseProvider
from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    ReasoningEvent,
    ReasoningPart,
    StreamChunk,
    ToolCall,
    ToolSet,
    Usage,
)


@dataclass
class FauxResponse:
    """One scripted provider response.

    ``chunks`` controls text chunk boundaries for streaming. When omitted, the
    full ``text`` value is emitted as one chunk. ``tool_calls`` are placed on the
    final stream chunk so the normal ai-query tool loop executes them.
    """

    text: str = ""
    chunks: list[str] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_events: list[ReasoningEvent] = field(default_factory=list)
    finish_reason: str | None = "stop"
    usage: Usage | None = field(default_factory=Usage)
    response: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    error: BaseException | None = None


@dataclass(frozen=True)
class FauxCall:
    """Recorded call received by :class:`FauxProvider`."""

    method: Literal["generate", "stream"]
    model: str
    messages: list[Message]
    tools: ToolSet | None
    provider_options: ProviderOptions | None
    kwargs: dict[str, Any]


FauxResponseFactory = Callable[
    [FauxCall],
    FauxResponse | Awaitable[FauxResponse],
]
FauxResponseStep = FauxResponse | FauxResponseFactory


class FauxProvider(BaseProvider):
    """Queue-backed provider that never performs network requests.

    Responses are consumed in call order. Exhausting the queue raises a clear
    error so an under-specified test cannot accidentally pass with fallback text.
    """

    name = "faux"

    def __init__(
        self,
        responses: Iterable[FauxResponseStep] = (),
    ) -> None:
        super().__init__(api_key="faux")
        self._responses: deque[FauxResponseStep] = deque(responses)
        self.calls: list[FauxCall] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def pending_response_count(self) -> int:
        return len(self._responses)

    def set_responses(self, responses: Iterable[FauxResponseStep]) -> None:
        """Replace all pending responses."""
        self._responses = deque(responses)

    def append_responses(self, responses: Iterable[FauxResponseStep]) -> None:
        """Append responses after the currently queued steps."""
        self._responses.extend(responses)

    def assert_exhausted(self) -> None:
        """Raise when a test left scripted responses unused."""
        if self._responses:
            raise AssertionError(
                f"{len(self._responses)} faux response(s) were not consumed"
            )

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        response = await self._next_response(
            method="generate",
            model=model,
            messages=messages,
            tools=tools,
            provider_options=provider_options,
            kwargs=kwargs,
        )
        if response.error is not None:
            raise response.error

        raw_response = copy.deepcopy(response.response)
        raw_response["tool_calls"] = copy.deepcopy(response.tool_calls)
        return GenerateTextResult(
            text=response.text,
            reasoning_parts=_reasoning_parts(response.reasoning_events),
            finish_reason=response.finish_reason,
            usage=copy.deepcopy(response.usage),
            response=raw_response,
            provider_metadata=copy.deepcopy(response.provider_metadata),
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
        response = await self._next_response(
            method="stream",
            model=model,
            messages=messages,
            tools=tools,
            provider_options=provider_options,
            kwargs=kwargs,
        )
        if response.error is not None:
            raise response.error

        if response.reasoning_events:
            yield StreamChunk(
                reasoning_events=copy.deepcopy(response.reasoning_events)
            )

        chunks = response.chunks if response.chunks is not None else [response.text]
        for chunk in chunks:
            if chunk:
                yield StreamChunk(text=chunk)

        yield StreamChunk(
            is_final=True,
            usage=copy.deepcopy(response.usage),
            finish_reason=response.finish_reason,
            tool_calls=copy.deepcopy(response.tool_calls) or None,
        )

    async def _next_response(
        self,
        *,
        method: Literal["generate", "stream"],
        model: str,
        messages: list[Message],
        tools: ToolSet | None,
        provider_options: ProviderOptions | None,
        kwargs: dict[str, Any],
    ) -> FauxResponse:
        call = FauxCall(
            method=method,
            model=model,
            messages=copy.deepcopy(messages),
            tools=dict(tools) if tools is not None else None,
            provider_options=copy.deepcopy(provider_options),
            kwargs=dict(kwargs),
        )
        self.calls.append(call)

        if not self._responses:
            raise RuntimeError(
                f"No faux response queued for {method} call {self.call_count}"
            )

        step = self._responses.popleft()
        response = step(call) if callable(step) else step
        if inspect.isawaitable(response):
            response = await response
        if not isinstance(response, FauxResponse):
            raise TypeError(
                "Faux response factories must return FauxResponse, got "
                f"{type(response).__name__}"
            )
        return copy.deepcopy(response)


def faux(
    model: str = "faux-1",
    *,
    responses: Iterable[FauxResponseStep] = (),
) -> LanguageModel:
    """Create a deterministic language model backed by :class:`FauxProvider`."""
    return LanguageModel(
        provider=FauxProvider(responses=responses),
        model_id=model,
    )


def _reasoning_parts(events: list[ReasoningEvent]) -> list[ReasoningPart]:
    return [
        ReasoningPart(
            text=event.text or "",
            data={"provider": event.provider, "kind": event.kind, **event.data},
        )
        for event in events
    ]
