"""Tests for stream_text function."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from ai_query import stream_text, tool, Field, step_count_is
from ai_query.types import (
    Message,
    Usage,
    StreamChunk,
    ToolCall,
    StepStartEvent,
    StepFinishEvent,
)
from ai_query.model import LanguageModel


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
    async def test_stream_accumulated_usage_with_tools(self):
        """stream_text should accumulate usage across tool calls."""
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
        assert usage.input_tokens == 30  # 10 + 20
        assert usage.output_tokens == 15  # 5 + 10


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
