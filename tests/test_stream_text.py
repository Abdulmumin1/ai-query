"""Tests for stream_text function."""

from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import AsyncMock

from ai_query import stream_text, tool, Field, step_count_is
from ai_query.types import (
    Message,
    AbortController,
    AbortError,
    ReasoningEvent,
    ReasoningPart,
    RetryPolicy,
    StepControl,
    Usage,
    StreamChunk,
    ToolCall,
    StepStartEvent,
    StepFinishEvent,
)
from ai_query.model import LanguageModel
from ai_query.transport import HTTPStatusError


# Import fixtures from conftest
from tests.conftest import MockProvider, make_stream_chunks, make_tool_call


# =============================================================================
# Basic stream_text Tests
# =============================================================================


class TestStreamTextBasic:
    """Basic tests for stream_text function."""

    @pytest.mark.asyncio
    async def test_simple_stream(self):
        """stream_text should yield text chunks."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Hello "),
                StreamChunk(text="world!"),
                StreamChunk(
                    is_final=True,
                    usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
                    finish_reason="stop",
                ),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Hi")

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]

    @pytest.mark.asyncio
    async def test_retries_stream_before_output(self):
        """stream_text should retry when the provider fails before any chunk."""

        class FlakyProvider(MockProvider):
            async def stream(self, **kwargs):
                self.last_messages = kwargs["messages"]
                self.stream_call_count += 1
                if self.stream_call_count == 1:
                    raise ConnectionError("connection reset")
                yield StreamChunk(text="Recovered")
                yield StreamChunk(is_final=True, finish_reason="stop")

        retry_events = []
        provider = FlakyProvider()
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Hello",
            retry=RetryPolicy(max_attempts=2, initial_delay=0, jitter=False),
            on_retry=retry_events.append,
        )

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        assert chunks == ["Recovered"]
        assert provider.stream_call_count == 2
        assert len(retry_events) == 1
        assert retry_events[0].attempt == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_stream_non_transient_http_status(self):
        """stream_text should not retry non-transient provider request errors."""

        class BadRequestProvider(MockProvider):
            async def stream(self, **kwargs):
                self.stream_call_count += 1
                raise HTTPStatusError(401, "unauthorized")
                yield StreamChunk(text="unreachable")

        provider = BadRequestProvider()
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Hello",
            retry=RetryPolicy(max_attempts=3, initial_delay=0, jitter=False),
        )

        with pytest.raises(HTTPStatusError) as exc_info:
            async for _ in result.text_stream:
                pass

        assert exc_info.value.status_code == 401
        assert provider.stream_call_count == 1

    @pytest.mark.asyncio
    async def test_does_not_retry_stream_after_output(self):
        """stream_text should not replay a stream after user-visible output."""

        class FailingAfterOutputProvider(MockProvider):
            async def stream(self, **kwargs):
                self.stream_call_count += 1
                yield StreamChunk(text="partial")
                raise ConnectionError("lost connection")

        provider = FailingAfterOutputProvider()
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Hello",
            retry=RetryPolicy(max_attempts=2, initial_delay=0, jitter=False),
        )

        chunks = []
        with pytest.raises(ConnectionError, match="lost connection"):
            async for chunk in result.text_stream:
                chunks.append(chunk)

        assert chunks == ["partial"]
        assert provider.stream_call_count == 1

    @pytest.mark.asyncio
    async def test_abort_cancels_in_flight_stream_read(self):
        """stream_text should abort while waiting for the next provider chunk."""

        class SlowStreamProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.cancelled = False

            async def stream(self, **kwargs):
                self.stream_call_count += 1
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                yield StreamChunk(text="late")

        provider = SlowStreamProvider()
        model = LanguageModel(provider=provider, model_id="test-model")
        controller = AbortController()
        result = stream_text(
            model=model,
            prompt="Hello",
            signal=controller.signal,
        )

        async def consume():
            async for _ in result.text_stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        controller.abort("user cancelled")

        with pytest.raises(AbortError, match="user cancelled"):
            await task

        assert provider.cancelled

    @pytest.mark.asyncio
    async def test_stream_with_usage(self):
        """stream_text should provide usage after completion."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Test"),
                StreamChunk(
                    is_final=True,
                    usage=Usage(input_tokens=10, output_tokens=1, total_tokens=11),
                    finish_reason="stop",
                ),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        # Consume stream first
        async for _ in result.text_stream:
            pass

        usage = await result.usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 1
        assert usage.total_tokens == 11

    @pytest.mark.asyncio
    async def test_stream_with_system_and_messages_prepends_system(self):
        """stream_text should prepend system when messages lack a leading system message."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="ok"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            system="repo rules",
            messages=[{"role": "user", "content": "hi"}],
        )

        async for _ in result.text_stream:
            pass

        assert [msg.role for msg in provider.last_messages] == ["system", "user"]
        assert provider.last_messages[0].content == "repo rules"

    @pytest.mark.asyncio
    async def test_stream_with_system_and_existing_system_message_does_not_duplicate(self):
        """stream_text should not duplicate a leading system message."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="ok"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            system="repo rules",
            messages=[
                Message(role="system", content="existing rules"),
                Message(role="user", content="hi"),
            ],
        )

        async for _ in result.text_stream:
            pass

        assert [msg.role for msg in provider.last_messages] == ["system", "user"]
        assert provider.last_messages[0].content == "existing rules"

    @pytest.mark.asyncio
    async def test_stream_full_text(self):
        """stream_text should provide full text after completion."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Hello "),
                StreamChunk(text="world "),
                StreamChunk(text="!"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Greet")

        # Consume stream
        async for _ in result.text_stream:
            pass

        text = await result.text
        assert text == "Hello world !"

    @pytest.mark.asyncio
    async def test_stream_finish_reason(self):
        """stream_text should provide finish reason after completion."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        async for _ in result.text_stream:
            pass

        reason = await result.finish_reason
        assert reason == "stop"

    @pytest.mark.asyncio
    async def test_reasoning_is_mapped_into_stream_provider_options(self):
        """stream_text should map normalized reasoning before provider invocation."""

        class ReasoningMockProvider(MockProvider):
            def reasoning_capabilities(self, model=None):
                from ai_query.providers.base import ReasoningCapabilities

                return ReasoningCapabilities(supported=True, supports_effort=True)

            def apply_reasoning(self, provider_options, reasoning, *, model):
                options = self._clone_provider_options(provider_options)
                options.setdefault(self.name, {})["effort"] = reasoning["effort"]
                return options

        provider = ReasoningMockProvider(
            stream_chunks=[
                [
                    StreamChunk(text="Hello"),
                    StreamChunk(is_final=True, finish_reason="stop"),
                ]
            ]
        )
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Test reasoning",
            provider_options={"other": {"value": 1}},
            reasoning={"effort": "high"},
        )

        async for _ in result.text_stream:
            pass

        assert provider.last_provider_options == {
            "other": {"value": 1},
            "mock": {"effort": "high"},
        }

    @pytest.mark.asyncio
    async def test_reasoning_events_are_forwarded_separately(self):
        """stream_text should forward reasoning events without mixing them into text."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(
                    reasoning_events=[
                        ReasoningEvent(
                            kind="delta",
                            provider="mock",
                            text="thinking",
                        )
                    ]
                ),
                StreamChunk(text="Answer"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")
        reasoning_events = []

        result = stream_text(
            model=model,
            prompt="Test reasoning stream",
            on_reasoning_event=reasoning_events.append,
        )

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        assert chunks == ["Answer"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].kind == "delta"
        assert reasoning_events[0].text == "thinking"

    @pytest.mark.asyncio
    async def test_reasoning_event_callback_can_be_async(self):
        """stream_text should await async reasoning event callbacks."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(
                    reasoning_events=[
                        ReasoningEvent(kind="summary", provider="mock", text="summary")
                    ]
                ),
                StreamChunk(text="Answer"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")
        reasoning_events = []

        async def on_reasoning_event(event):
            reasoning_events.append(event)

        result = stream_text(
            model=model,
            prompt="Test async reasoning stream",
            on_reasoning_event=on_reasoning_event,
        )

        async for _ in result.text_stream:
            pass

        assert [event.text for event in reasoning_events] == ["summary"]

    @pytest.mark.asyncio
    async def test_stream_reasoning_parts_are_carried_into_next_tool_step(self):
        """stream_text should preserve assistant reasoning between tool steps."""

        @tool(description="Read file")
        def read_file(path: str) -> str:
            return f"contents of {path}"

        provider = MockProvider(
            stream_chunks=[
                [
                    StreamChunk(
                        reasoning_events=[
                            ReasoningEvent(
                                kind="delta",
                                provider="mock",
                                text="Read file A, then update file B.",
                            )
                        ]
                    ),
                    StreamChunk(
                        is_final=True,
                        finish_reason="tool_use",
                        tool_calls=[
                            make_tool_call("read_file", {"path": "file_a.txt"}, id="call_1")
                        ],
                    ),
                ],
                [
                    StreamChunk(text="Updated file B."),
                    StreamChunk(is_final=True, finish_reason="stop"),
                ],
            ]
        )
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Fix the bug",
            tools={"read_file": read_file},
            stop_when=step_count_is(10),
        )

        async for _ in result.text_stream:
            pass

        steps = await result.steps
        assert steps[0].reasoning_parts[0].text == "Read file A, then update file B."
        assistant_message = provider.last_messages[1]
        assert assistant_message.role == "assistant"
        assert isinstance(assistant_message.content, list)
        assert isinstance(assistant_message.content[0], ReasoningPart)
        assert assistant_message.content[0].text == "Read file A, then update file B."

    @pytest.mark.asyncio
    async def test_stream_direct_iteration(self):
        """stream_text result should be directly iterable."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Direct "),
                StreamChunk(text="iteration"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        chunks = []
        async for chunk in result:  # Direct iteration
            chunks.append(chunk)

        assert chunks == ["Direct ", "iteration"]

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self):
        """stream_text should handle system prompt."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Poem"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            system="You are a poet.",
            prompt="Write something.",
        )

        async for _ in result.text_stream:
            pass

        assert len(provider.last_messages) == 2
        assert provider.last_messages[0].role == "system"

    @pytest.mark.asyncio
    async def test_stream_with_messages(self):
        """stream_text should handle messages list."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Response"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        )

        async for _ in result.text_stream:
            pass

        assert len(provider.last_messages) == 3

    @pytest.mark.asyncio
    async def test_stream_requires_prompt_or_messages(self):
        """stream_text should raise error without prompt or messages."""
        provider = MockProvider()
        model = LanguageModel(provider=provider, model_id="test-model")

        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            stream_text(model=model)


# =============================================================================
# stream_text with Tools Tests
# =============================================================================


class TestStreamTextWithTools:
    """Tests for stream_text with tool calling."""

    @pytest.mark.asyncio
    async def test_stream_with_tool_call(self):
        """stream_text should handle tool calls."""
        @tool(description="Add")
        def add(a: int, b: int) -> int:
            return a + b

        provider = MockProvider(stream_chunks=[
            # First stream: tool call
            [
                StreamChunk(text="Let me calculate."),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})],
                ),
            ],
            # Second stream: final response
            [
                StreamChunk(text="The result is 5."),
                StreamChunk(is_final=True, finish_reason="stop"),
            ],
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="What is 2 + 3?",
            tools={"add": add},
            stop_when=step_count_is(10),
        )

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        # Should have chunks from both steps
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "calculate" in full_text.lower() or "5" in full_text

    @pytest.mark.asyncio
    async def test_multiple_async_tool_calls_run_in_parallel(self):
        """stream_text should execute same-step async tool calls concurrently."""
        @tool(description="Slow tool")
        async def slow_tool(name: str) -> str:
            await asyncio.sleep(0.2)
            return name

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[
                        ToolCall(id="call_1", name="slow_tool", arguments={"name": "a"}),
                        ToolCall(id="call_2", name="slow_tool", arguments={"name": "b"}),
                        ToolCall(id="call_3", name="slow_tool", arguments={"name": "c"}),
                    ],
                )
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        started = time.perf_counter()
        result = stream_text(
            model=model,
            prompt="Run all tools",
            tools={"slow_tool": slow_tool},
            stop_when=step_count_is(1),
        )
        async for _ in result.text_stream:
            pass
        elapsed = time.perf_counter() - started

        steps = await result.steps
        assert elapsed < 0.4
        assert [tr.result for tr in steps[0].tool_results] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stream_usage_is_from_last_step_with_tools(self):
        """stream_text usage should come from the last provider step."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Calling"),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})],
                ),
            ],
            [
                StreamChunk(text="Done"),
                StreamChunk(
                    is_final=True,
                    finish_reason="stop",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ],
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            stop_when=step_count_is(10),
        )

        async for _ in result.text_stream:
            pass

        usage = await result.usage
        assert usage.input_tokens == 20
        assert usage.output_tokens == 10

        steps = await result.steps
        assert steps[0].usage is not None
        assert steps[0].usage.total_tokens == 15
        assert steps[1].usage is not None
        assert steps[1].usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_stream_step_finish_event_exposes_step_usage(self):
        """Stream step finish events should expose provider usage for that step."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Calling"),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})],
                ),
            ],
            [
                StreamChunk(text="Done"),
                StreamChunk(
                    is_final=True,
                    finish_reason="stop",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ],
        ])
        model = LanguageModel(provider=provider, model_id="test-model")
        events: list[StepFinishEvent] = []

        result = stream_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            stop_when=step_count_is(10),
            on_step_finish=events.append,
        )

        async for _ in result.text_stream:
            pass

        assert [event.usage.total_tokens for event in events] == [15, 30]

        steps = await result.steps
        assert steps[0].usage is not None
        assert steps[0].usage.total_tokens == 15
        assert steps[1].usage is not None
        assert steps[1].usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_stream_steps_include_tool_messages(self):
        """stream_text should retain tool-call steps after consumption."""

        @tool(description="Add")
        def add(a: int, b: int) -> int:
            return a + b

        provider = MockProvider(
            stream_chunks=[
                [
                    StreamChunk(text="Let me calculate. "),
                    StreamChunk(
                        is_final=True,
                        finish_reason="tool_use",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="add",
                                arguments={"a": 2, "b": 3},
                            )
                        ],
                    ),
                ],
                [
                    StreamChunk(text="The result is 5."),
                    StreamChunk(is_final=True, finish_reason="stop"),
                ],
            ]
        )
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="What is 2 + 3?",
            tools={"add": add},
            stop_when=step_count_is(10),
        )

        async for _ in result.text_stream:
            pass

        steps = await result.steps
        assert len(steps) == 2
        assert steps[0].tool_calls[0].name == "add"
        assert steps[0].tool_results[0].result == 5


# =============================================================================
# stream_text Stop Conditions Tests
# =============================================================================


class TestStreamTextStopConditions:
    """Tests for stop conditions in stream_text."""

    @pytest.mark.asyncio
    async def test_stream_stop_when_step_count(self):
        """stream_text should stop at step count."""
        from ai_query import step_count_is

        @tool(description="Step")
        def step() -> str:
            return "stepped"

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text=f"Step {i}"),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[ToolCall(id=f"call_{i}", name="step", arguments={})],
                ),
            ]
            for i in range(10)
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(
            model=model,
            prompt="Keep stepping",
            tools={"step": step},
            stop_when=step_count_is(3),
        )

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        # Should have stopped after 3 steps
        # Count occurrences of "Step" in chunks
        step_count = sum(1 for c in chunks if "Step" in c)
        assert step_count <= 3


# =============================================================================
# stream_text Callbacks Tests
# =============================================================================


class TestStreamTextCallbacks:
    """Tests for step callbacks in stream_text."""

    @pytest.mark.asyncio
    async def test_stream_on_step_start(self):
        """on_step_start should be called in streaming."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Hello"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        start_events = []

        def on_start(event: StepStartEvent):
            start_events.append(event.step_number)

        result = stream_text(
            model=model,
            prompt="Hi",
            on_step_start=on_start,
        )

        async for _ in result.text_stream:
            pass

        assert start_events == [1]

    @pytest.mark.asyncio
    async def test_stream_on_step_finish(self):
        """on_step_finish should be called in streaming."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        finish_events = []

        def on_finish(event: StepFinishEvent):
            finish_events.append({
                "step": event.step_number,
                "text": event.text,
            })

        result = stream_text(
            model=model,
            prompt="Test",
            on_step_finish=on_finish,
        )

        async for _ in result.text_stream:
            pass

        assert len(finish_events) == 1
        assert finish_events[0]["step"] == 1
        assert "Done" in finish_events[0]["text"]

    @pytest.mark.asyncio
    async def test_stream_async_callbacks(self):
        """Async callbacks should work with streaming."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Test"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        events = []

        async def on_start(event: StepStartEvent):
            events.append(("start", event.step_number))

        async def on_finish(event: StepFinishEvent):
            events.append(("finish", event.step_number))

        result = stream_text(
            model=model,
            prompt="Test",
            on_step_start=on_start,
            on_step_finish=on_finish,
        )

        async for _ in result.text_stream:
            pass

        assert events == [("start", 1), ("finish", 1)]

    @pytest.mark.asyncio
    async def test_stream_callbacks_with_multiple_steps(self):
        """Callbacks should be called for each streaming step."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Calling"),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})],
                ),
            ],
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ],
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        step_numbers = []

        def on_finish(event: StepFinishEvent):
            step_numbers.append(event.step_number)

        result = stream_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            on_step_finish=on_finish,
            stop_when=step_count_is(10),
        )

        async for _ in result.text_stream:
            pass

        assert step_numbers == [1, 2]

    @pytest.mark.asyncio
    async def test_stream_step_control_accepts_dict_messages(self):
        """Streaming step control should normalize injected dict messages."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Calling"),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})],
                ),
            ],
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ],
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        def on_start(event: StepStartEvent):
            if event.step_number == 2:
                return StepControl(
                    inject_messages=[{"role": "user", "content": "Extra instruction"}]
                )
            return None

        result = stream_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            on_step_start=on_start,
            stop_when=step_count_is(10),
        )

        async for _ in result.text_stream:
            pass

        assert provider.last_messages[-1].role == "user"
        assert provider.last_messages[-1].content == "Extra instruction"


# =============================================================================
# TextStreamResult Tests
# =============================================================================


class TestTextStreamResult:
    """Tests for TextStreamResult behavior."""

    @pytest.mark.asyncio
    async def test_text_before_stream_consumed(self):
        """Accessing text before consuming stream should consume it."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Auto "),
                StreamChunk(text="consumed"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        # Access text directly without consuming stream first
        text = await result.text
        assert text == "Auto consumed"

    @pytest.mark.asyncio
    async def test_usage_before_stream_consumed(self):
        """Accessing usage before consuming stream should consume it."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Test"),
                StreamChunk(
                    is_final=True,
                    usage=Usage(input_tokens=5, output_tokens=1, total_tokens=6),
                    finish_reason="stop",
                ),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        # Access usage directly
        usage = await result.usage
        assert usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_multiple_property_access(self):
        """Multiple awaits on properties should return same values."""
        provider = MockProvider(stream_chunks=[
            [
                StreamChunk(text="Hello"),
                StreamChunk(
                    is_final=True,
                    usage=Usage(total_tokens=10),
                    finish_reason="stop",
                ),
            ]
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = stream_text(model=model, prompt="Test")

        # Consume stream
        async for _ in result.text_stream:
            pass

        # Access properties multiple times
        text1 = await result.text
        text2 = await result.text
        usage1 = await result.usage
        usage2 = await result.usage

        assert text1 == text2 == "Hello"
        assert usage1.total_tokens == usage2.total_tokens == 10
