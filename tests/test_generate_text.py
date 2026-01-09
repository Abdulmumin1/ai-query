"""Tests for generate_text function."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_query import generate_text, tool, Field, step_count_is
from ai_query.types import (
    GenerateTextResult,
    Message,
    Usage,
    ToolCall,
    StepStartEvent,
    StepFinishEvent,
)
from ai_query.model import LanguageModel


# Import fixtures from conftest
from tests.conftest import MockProvider, make_response, make_tool_call


# =============================================================================
# Basic generate_text Tests
# =============================================================================


class TestGenerateTextBasic:
    """Basic tests for generate_text function."""

    @pytest.mark.asyncio
    async def test_simple_prompt(self):
        """generate_text should work with simple prompt."""
        provider = MockProvider(responses=[
            make_response(text="Hello! I'm an AI assistant.")
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Hello, who are you?",
        )

        assert result.text == "Hello! I'm an AI assistant."
        assert result.finish_reason == "stop"
        assert provider.call_count == 1
        assert len(provider.last_messages) == 1
        assert provider.last_messages[0].role == "user"
        assert provider.last_messages[0].content == "Hello, who are you?"

    @pytest.mark.asyncio
    async def test_with_system_prompt(self):
        """generate_text should handle system prompt."""
        provider = MockProvider(responses=[
            make_response(text="I am a helpful poet.")
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            system="You are a helpful poet.",
            prompt="Who are you?",
        )

        assert len(provider.last_messages) == 2
        assert provider.last_messages[0].role == "system"
        assert provider.last_messages[0].content == "You are a helpful poet."
        assert provider.last_messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_with_messages(self):
        """generate_text should handle messages list."""
        provider = MockProvider(responses=[
            make_response(text="The capital is Paris.")
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )

        assert result.text == "The capital is Paris."
        assert len(provider.last_messages) == 1

    @pytest.mark.asyncio
    async def test_with_message_objects(self):
        """generate_text should handle Message objects."""
        provider = MockProvider(responses=[
            make_response(text="Hello!")
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            messages=[
                Message(role="user", content="Hi"),
            ],
        )

        assert result.text == "Hello!"

    @pytest.mark.asyncio
    async def test_requires_prompt_or_messages(self):
        """generate_text should raise error without prompt or messages."""
        provider = MockProvider()
        model = LanguageModel(provider=provider, model_id="test-model")

        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            await generate_text(model=model)

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_provider(self):
        """generate_text should pass kwargs to provider."""
        provider = MockProvider(responses=[make_response()])
        model = LanguageModel(provider=provider, model_id="test-model")

        await generate_text(
            model=model,
            prompt="Test",
            max_tokens=100,
            temperature=0.7,
        )

        assert provider.last_kwargs["max_tokens"] == 100
        assert provider.last_kwargs["temperature"] == 0.7


# =============================================================================
# Tool Calling Tests
# =============================================================================


class TestGenerateTextWithTools:
    """Tests for generate_text with tool calling."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """generate_text should handle single tool call."""
        @tool(description="Add two numbers")
        def add(a: int = Field(description="First"), b: int = Field(description="Second")) -> int:
            return a + b

        # First response: model calls the tool
        # Second response: model gives final answer
        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("add", {"a": 2, "b": 3}, id="call_1")],
            ),
            make_response(
                text="The result is 5.",
                finish_reason="stop",
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="What is 2 + 3?",
            tools={"add": add},
            stop_when=step_count_is(10),
        )

        assert "5" in result.text
        assert provider.call_count == 2
        assert len(result.steps) > 0
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """generate_text should handle multiple tool calls."""
        @tool(description="Add")
        def add(a: int, b: int) -> int:
            return a + b

        @tool(description="Multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[
                    make_tool_call("add", {"a": 2, "b": 3}, id="call_1"),
                    make_tool_call("multiply", {"a": 4, "b": 5}, id="call_2"),
                ],
            ),
            make_response(
                text="2+3=5 and 4*5=20",
                finish_reason="stop",
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Calculate 2+3 and 4*5",
            tools={"add": add, "multiply": multiply},
            stop_when=step_count_is(10),
        )

        assert "5" in result.text and "20" in result.text

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """generate_text should handle unknown tool gracefully."""
        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("unknown_tool", {}, id="call_1")],
            ),
            make_response(
                text="I couldn't find that tool.",
                finish_reason="stop",
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Use the unknown tool",
            tools={},
        )

        # Should not crash, should report error
        steps = result.steps
        assert any(
            any(tr.is_error for tr in step.tool_results)
            for step in steps
            if step.tool_results
        )

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """generate_text should handle tool errors gracefully."""
        @tool(description="Divide")
        def divide(a: int, b: int) -> float:
            return a / b  # Will raise ZeroDivisionError

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("divide", {"a": 10, "b": 0}, id="call_1")],
            ),
            make_response(
                text="Division by zero error occurred.",
                finish_reason="stop",
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Divide 10 by 0",
            tools={"divide": divide},
        )

        # Should not crash
        assert result.text is not None
        steps = result.steps
        assert any(
            any(tr.is_error for tr in step.tool_results)
            for step in steps
            if step.tool_results
        )

    @pytest.mark.asyncio
    async def test_async_tool(self):
        """generate_text should work with async tools."""
        @tool(description="Async fetch")
        async def fetch(url: str = Field(description="URL")) -> str:
            return f"Content from {url}"

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("fetch", {"url": "http://example.com"}, id="call_1")],
            ),
            make_response(
                text="I fetched the content.",
                finish_reason="stop",
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Fetch example.com",
            tools={"fetch": fetch},
            stop_when=step_count_is(10),
        )

        assert "fetched" in result.text.lower()


# =============================================================================
# Stop Conditions Tests
# =============================================================================


class TestGenerateTextStopConditions:
    """Tests for stop conditions in generate_text."""

    @pytest.mark.asyncio
    async def test_stop_when_step_count(self):
        """generate_text should stop at step count."""
        from ai_query import step_count_is

        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        # Create many responses to ensure loop would continue
        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("echo", {"msg": "1"}, id=f"call_{i}")],
            )
            for i in range(10)
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Keep echoing",
            tools={"echo": echo},
            stop_when=step_count_is(3),
        )

        # Should stop after 3 steps
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_stop_when_has_tool_call(self):
        """generate_text should stop when specific tool is called."""
        from ai_query import has_tool_call

        @tool(description="Search")
        def search(q: str) -> str:
            return f"Results for {q}"

        @tool(description="Done")
        def done(answer: str) -> str:
            return answer

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("search", {"q": "test"}, id="call_1")],
            ),
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("done", {"answer": "Found it!"}, id="call_2")],
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Search and finish",
            tools={"search": search, "done": done},
            stop_when=has_tool_call("done"),
        )

        # Should stop when "done" is called
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_multiple_stop_conditions(self):
        """generate_text should stop when any condition is met."""
        from ai_query import step_count_is, has_tool_call

        @tool(description="Step")
        def step() -> str:
            return "stepped"

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("step", {}, id=f"call_{i}")],
            )
            for i in range(10)
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Keep stepping",
            tools={"step": step},
            stop_when=[step_count_is(5), has_tool_call("done")],
        )

        # Should stop at 5 (step_count condition met first)
        assert len(result.steps) == 5

    @pytest.mark.asyncio
    async def test_default_stop_condition(self):
        """generate_text should default to step_count_is(1)."""
        @tool(description="Loop")
        def loop() -> str:
            return "looped"

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("loop", {}, id=f"call_{i}")],
            )
            for i in range(20)
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Loop forever",
            tools={"loop": loop},
        )

        # Should stop at default 1
        assert len(result.steps) == 1


# =============================================================================
# Step Callbacks Tests
# =============================================================================


class TestGenerateTextCallbacks:
    """Tests for step callbacks in generate_text."""

    @pytest.mark.asyncio
    async def test_on_step_start_called(self):
        """on_step_start should be called before each step."""
        provider = MockProvider(responses=[make_response(text="Done")])
        model = LanguageModel(provider=provider, model_id="test-model")

        start_events = []

        def on_start(event: StepStartEvent):
            start_events.append(event)

        await generate_text(
            model=model,
            prompt="Test",
            on_step_start=on_start,
        )

        assert len(start_events) == 1
        assert start_events[0].step_number == 1
        assert len(start_events[0].messages) == 1

    @pytest.mark.asyncio
    async def test_on_step_finish_called(self):
        """on_step_finish should be called after each step."""
        provider = MockProvider(responses=[make_response(text="Done")])
        model = LanguageModel(provider=provider, model_id="test-model")

        finish_events = []

        def on_finish(event: StepFinishEvent):
            finish_events.append(event)

        await generate_text(
            model=model,
            prompt="Test",
            on_step_finish=on_finish,
        )

        assert len(finish_events) == 1
        assert finish_events[0].step_number == 1
        assert finish_events[0].step.text == "Done"

    @pytest.mark.asyncio
    async def test_async_callbacks(self):
        """Async callbacks should work."""
        provider = MockProvider(responses=[make_response(text="Done")])
        model = LanguageModel(provider=provider, model_id="test-model")

        events = []

        async def on_start(event: StepStartEvent):
            events.append(("start", event.step_number))

        async def on_finish(event: StepFinishEvent):
            events.append(("finish", event.step_number))

        await generate_text(
            model=model,
            prompt="Test",
            on_step_start=on_start,
            on_step_finish=on_finish,
        )

        assert events == [("start", 1), ("finish", 1)]

    @pytest.mark.asyncio
    async def test_callbacks_with_tools(self):
        """Callbacks should be called for each step with tools."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(responses=[
            make_response(
                text="Calling echo",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("echo", {"msg": "hi"}, id="call_1")],
            ),
            make_response(text="Done", finish_reason="stop"),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        step_numbers = []

        def on_finish(event: StepFinishEvent):
            step_numbers.append(event.step_number)

        await generate_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            on_step_finish=on_finish,
            stop_when=step_count_is(10),
        )

        assert step_numbers == [1, 2]

    @pytest.mark.asyncio
    async def test_callback_can_modify_messages(self):
        """on_step_start can modify messages."""
        provider = MockProvider(responses=[make_response(text="Modified!")])
        model = LanguageModel(provider=provider, model_id="test-model")

        def on_start(event: StepStartEvent):
            # Add a message
            event.messages.append(Message(role="user", content="Extra instruction"))

        await generate_text(
            model=model,
            prompt="Test",
            on_step_start=on_start,
        )

        # Provider should receive modified messages
        assert len(provider.last_messages) == 2
        assert provider.last_messages[1].content == "Extra instruction"


# =============================================================================
# Usage Accumulation Tests
# =============================================================================


class TestGenerateTextUsage:
    """Tests for usage tracking in generate_text."""

    @pytest.mark.asyncio
    async def test_usage_returned(self):
        """generate_text should return usage stats."""
        provider = MockProvider(responses=[
            make_response(
                text="Hello",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(model=model, prompt="Hi")

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_usage_accumulated_with_tools(self):
        """Usage should accumulate across tool calls."""
        @tool(description="Echo")
        def echo(msg: str) -> str:
            return msg

        provider = MockProvider(responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("echo", {"msg": "1"}, id="call_1")],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            make_response(
                text="Done",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        ])
        model = LanguageModel(provider=provider, model_id="test-model")

        result = await generate_text(
            model=model,
            prompt="Test",
            tools={"echo": echo},
            stop_when=step_count_is(10),
        )

        # Usage should be accumulated
        assert result.usage.input_tokens == 30  # 10 + 20
        assert result.usage.output_tokens == 15  # 5 + 10
        assert result.usage.total_tokens == 45  # 15 + 30
