"""Tests for ai_query.types module."""

from __future__ import annotations

import pytest
from typing import Literal

from ai_query.types import (
    Field,
    Tool,
    tool,
    ToolCall,
    ToolResult,
    ToolCallPart,
    ToolResultPart,
    Message,
    TextPart,
    ImagePart,
    FilePart,
    Usage,
    GenerateTextResult,
    StreamChunk,
    TextStreamResult,
    StepResult,
    StepStartEvent,
    StepFinishEvent,
    step_count_is,
    has_tool_call,
    _python_type_to_json_schema,
    _MISSING,
)


# =============================================================================
# Field Tests
# =============================================================================


class TestField:
    """Tests for the Field class."""

    def test_field_with_description(self):
        """Field should store description."""
        field = Field(description="A test field")
        assert field.description == "A test field"
        assert field.default is _MISSING

    def test_field_with_default(self):
        """Field should store default value."""
        field = Field(description="With default", default=42)
        assert field.default == 42

    def test_field_with_enum(self):
        """Field should store enum values."""
        field = Field(description="Color", enum=["red", "green", "blue"])
        assert field.enum == ["red", "green", "blue"]

    def test_field_with_min_max(self):
        """Field should store min/max values."""
        field = Field(description="Age", min_value=0, max_value=150)
        assert field.min_value == 0
        assert field.max_value == 150

    def test_field_repr(self):
        """Field should have a useful repr."""
        field = Field(description="Test", default=10)
        assert "description='Test'" in repr(field)
        assert "default=10" in repr(field)


# =============================================================================
# Type Conversion Tests
# =============================================================================


class TestTypeConversion:
    """Tests for Python type to JSON Schema conversion."""

    def test_string_type(self):
        """str should convert to string."""
        schema = _python_type_to_json_schema(str)
        assert schema == {"type": "string"}

    def test_int_type(self):
        """int should convert to integer."""
        schema = _python_type_to_json_schema(int)
        assert schema == {"type": "integer"}

    def test_float_type(self):
        """float should convert to number."""
        schema = _python_type_to_json_schema(float)
        assert schema == {"type": "number"}

    def test_bool_type(self):
        """bool should convert to boolean."""
        schema = _python_type_to_json_schema(bool)
        assert schema == {"type": "boolean"}

    def test_list_type(self):
        """list should convert to array."""
        schema = _python_type_to_json_schema(list)
        assert schema == {"type": "array"}

    def test_list_with_item_type(self):
        """list[str] should convert to array with items."""
        schema = _python_type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_dict_type(self):
        """dict should convert to object."""
        schema = _python_type_to_json_schema(dict)
        assert schema == {"type": "object"}

    def test_optional_type(self):
        """Optional[str] should unwrap to string."""
        from typing import Optional
        schema = _python_type_to_json_schema(Optional[str])
        assert schema == {"type": "string"}

    def test_literal_type(self):
        """Literal should convert to enum."""
        schema = _python_type_to_json_schema(Literal["a", "b", "c"])
        assert schema == {"type": "string", "enum": ["a", "b", "c"]}

    def test_union_type(self):
        """Union should convert to anyOf."""
        from typing import Union
        schema = _python_type_to_json_schema(Union[str, int])
        assert "anyOf" in schema
        assert {"type": "string"} in schema["anyOf"]
        assert {"type": "integer"} in schema["anyOf"]


# =============================================================================
# Tool Tests
# =============================================================================


class TestTool:
    """Tests for the Tool dataclass."""

    def test_tool_creation(self):
        """Tool should be created with required fields."""
        tool_obj = Tool(
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            execute=lambda: "result",
        )
        assert tool_obj.description == "A test tool"
        assert tool_obj.parameters == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_tool_run_sync(self):
        """Tool.run should handle sync functions."""
        tool_obj = Tool(
            description="Sync tool",
            parameters={"type": "object", "properties": {}},
            execute=lambda x: x * 2,
        )
        result = await tool_obj.run(x=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_tool_run_async(self):
        """Tool.run should handle async functions."""
        async def async_fn(x: int) -> int:
            return x * 3

        tool_obj = Tool(
            description="Async tool",
            parameters={"type": "object", "properties": {}},
            execute=async_fn,
        )
        result = await tool_obj.run(x=5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_tool_run_with_field_defaults(self):
        """Tool.run should use Field defaults when args not provided."""
        def fn(x: int = Field(description="X", default=10)) -> int:
            return x * 2

        tool_obj = Tool(
            description="Tool with default",
            parameters={"type": "object", "properties": {}},
            execute=fn,
        )
        result = await tool_obj.run()
        assert result == 20


# =============================================================================
# Tool Decorator Tests
# =============================================================================


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_simple(self):
        """@tool should create a Tool from function."""
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.description == "Greet someone."
        assert "name" in greet.parameters["properties"]

    def test_tool_decorator_with_description(self):
        """@tool(description=...) should use provided description."""
        @tool(description="Custom greeting tool")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert greet.description == "Custom greeting tool"

    def test_tool_decorator_infers_types(self):
        """@tool should infer parameter types from hints."""
        @tool
        def calculate(a: int, b: float, flag: bool) -> str:
            return str(a + b)

        props = calculate.parameters["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_tool_decorator_with_field(self):
        """@tool should use Field for descriptions."""
        @tool(description="Add numbers")
        def add(
            a: int = Field(description="First number"),
            b: int = Field(description="Second number"),
        ) -> int:
            return a + b

        props = add.parameters["properties"]
        assert props["a"]["description"] == "First number"
        assert props["b"]["description"] == "Second number"

    def test_tool_decorator_required_params(self):
        """@tool should mark params without defaults as required."""
        @tool
        def fn(required: str, optional: str = Field(default="default")) -> str:
            return required

        assert "required" in fn.parameters["required"]
        assert "optional" not in fn.parameters["required"]

    def test_tool_decorator_field_enum(self):
        """@tool should include enum from Field."""
        @tool
        def choose(color: str = Field(description="Color", enum=["red", "blue"])) -> str:
            return color

        assert choose.parameters["properties"]["color"]["enum"] == ["red", "blue"]

    def test_tool_decorator_field_min_max(self):
        """@tool should include min/max from Field."""
        @tool
        def rate(score: int = Field(description="Score", min_value=1, max_value=10)) -> int:
            return score

        props = rate.parameters["properties"]["score"]
        assert props["minimum"] == 1
        assert props["maximum"] == 10

    def test_tool_factory_mode(self):
        """tool() can be used in factory mode."""
        def my_fn(x: int) -> int:
            return x * 2

        my_tool = tool(
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            execute=my_fn,
        )

        assert isinstance(my_tool, Tool)
        assert my_tool.description == "Double a number"

    @pytest.mark.asyncio
    async def test_tool_decorator_async_function(self):
        """@tool should work with async functions."""
        @tool(description="Async fetch")
        async def fetch(url: str = Field(description="URL")) -> str:
            return f"Fetched {url}"

        assert isinstance(fetch, Tool)
        result = await fetch.run(url="http://example.com")
        assert result == "Fetched http://example.com"


# =============================================================================
# ToolCall and ToolResult Tests
# =============================================================================


class TestToolCallAndResult:
    """Tests for ToolCall and ToolResult."""

    def test_tool_call_creation(self):
        """ToolCall should store call info."""
        tc = ToolCall(
            id="call_123",
            name="weather",
            arguments={"city": "Paris"},
        )
        assert tc.id == "call_123"
        assert tc.name == "weather"
        assert tc.arguments == {"city": "Paris"}

    def test_tool_call_with_metadata(self):
        """ToolCall should support metadata."""
        tc = ToolCall(
            id="call_123",
            name="weather",
            arguments={},
            metadata={"provider_id": "abc"},
        )
        assert tc.metadata == {"provider_id": "abc"}

    def test_tool_result_creation(self):
        """ToolResult should store result info."""
        tr = ToolResult(
            tool_call_id="call_123",
            tool_name="weather",
            result="Sunny, 72°F",
        )
        assert tr.tool_call_id == "call_123"
        assert tr.tool_name == "weather"
        assert tr.result == "Sunny, 72°F"
        assert tr.is_error is False

    def test_tool_result_error(self):
        """ToolResult should mark errors."""
        tr = ToolResult(
            tool_call_id="call_123",
            tool_name="weather",
            result="City not found",
            is_error=True,
        )
        assert tr.is_error is True


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for Message class."""

    def test_message_with_string_content(self):
        """Message should handle string content."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_message_with_parts(self):
        """Message should handle list of parts."""
        msg = Message(
            role="user",
            content=[
                TextPart(text="Check this image:"),
                ImagePart(image="base64data", media_type="image/png"),
            ],
        )
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextPart)
        assert isinstance(msg.content[1], ImagePart)

    def test_message_to_dict_string(self):
        """Message.to_dict should handle string content."""
        msg = Message(role="assistant", content="Response")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "Response"}

    def test_message_to_dict_parts(self):
        """Message.to_dict should handle parts."""
        msg = Message(
            role="user",
            content=[
                TextPart(text="Hello"),
                ImagePart(image="data", media_type="image/png"),
            ],
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert len(d["content"]) == 2
        assert d["content"][0]["type"] == "text"
        assert d["content"][1]["type"] == "image"


# =============================================================================
# Usage Tests
# =============================================================================


class TestUsage:
    """Tests for Usage class."""

    def test_usage_defaults(self):
        """Usage should have zero defaults."""
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_values(self):
        """Usage should store token counts."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=20,
            total_tokens=150,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 20
        assert usage.total_tokens == 150


# =============================================================================
# Result Types Tests
# =============================================================================


class TestGenerateTextResult:
    """Tests for GenerateTextResult."""

    def test_result_creation(self):
        """GenerateTextResult should store response data."""
        result = GenerateTextResult(
            text="Hello!",
            finish_reason="stop",
            usage=Usage(total_tokens=10),
            response={"id": "123"},
            provider_metadata={"model": "gpt-4"},
        )
        assert result.text == "Hello!"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 10
        assert result.response["id"] == "123"
        assert result.provider_metadata["model"] == "gpt-4"


class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_text_chunk(self):
        """StreamChunk should store text."""
        chunk = StreamChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.is_final is False

    def test_final_chunk(self):
        """Final chunk should have metadata."""
        chunk = StreamChunk(
            is_final=True,
            usage=Usage(total_tokens=20),
            finish_reason="stop",
        )
        assert chunk.is_final is True
        assert chunk.usage.total_tokens == 20

    def test_chunk_with_tool_calls(self):
        """StreamChunk can contain tool calls."""
        tc = ToolCall(id="1", name="test", arguments={})
        chunk = StreamChunk(is_final=True, tool_calls=[tc])
        assert chunk.tool_calls == [tc]


# =============================================================================
# StepResult Tests
# =============================================================================


class TestStepResult:
    """Tests for StepResult."""

    def test_step_result_creation(self):
        """StepResult should store step data."""
        step = StepResult(
            text="Thinking...",
            tool_calls=[ToolCall(id="1", name="calc", arguments={"x": 1})],
            tool_results=[ToolResult(tool_call_id="1", tool_name="calc", result="2")],
            finish_reason="tool_use",
        )
        assert step.text == "Thinking..."
        assert len(step.tool_calls) == 1
        assert len(step.tool_results) == 1


# =============================================================================
# Stop Condition Tests
# =============================================================================


class TestStopConditions:
    """Tests for stop condition functions."""

    def test_step_count_is(self):
        """step_count_is should return True at count."""
        condition = step_count_is(3)

        # Build fake steps
        steps = []
        assert condition(steps) is False

        steps = [StepResult(text="", tool_calls=[], tool_results=[])]
        assert condition(steps) is False

        steps = [
            StepResult(text="", tool_calls=[], tool_results=[]),
            StepResult(text="", tool_calls=[], tool_results=[]),
        ]
        assert condition(steps) is False

        steps = [
            StepResult(text="", tool_calls=[], tool_results=[]),
            StepResult(text="", tool_calls=[], tool_results=[]),
            StepResult(text="", tool_calls=[], tool_results=[]),
        ]
        assert condition(steps) is True

    def test_has_tool_call(self):
        """has_tool_call should detect specific tool."""
        condition = has_tool_call("final_answer")

        # Empty steps
        assert condition([]) is False

        # Step without the tool
        step1 = StepResult(
            text="",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
            tool_results=[],
        )
        assert condition([step1]) is False

        # Step with the tool
        step2 = StepResult(
            text="",
            tool_calls=[ToolCall(id="2", name="final_answer", arguments={})],
            tool_results=[],
        )
        assert condition([step1, step2]) is True


# =============================================================================
# Step Event Tests
# =============================================================================


class TestStepEvents:
    """Tests for StepStartEvent and StepFinishEvent."""

    def test_step_start_event(self):
        """StepStartEvent should store event data."""
        messages = [Message(role="user", content="Hi")]
        tools = {"test": Tool(description="", parameters={}, execute=lambda: None)}

        event = StepStartEvent(
            step_number=1,
            messages=messages,
            tools=tools,
        )
        assert event.step_number == 1
        assert event.messages == messages
        assert event.tools == tools

    def test_step_finish_event(self):
        """StepFinishEvent should store event data."""
        step = StepResult(text="Done", tool_calls=[], tool_results=[])
        usage = Usage(total_tokens=50)

        event = StepFinishEvent(
            step_number=2,
            step=step,
            text="Accumulated text",
            usage=usage,
            steps=[step],
        )
        assert event.step_number == 2
        assert event.step == step
        assert event.text == "Accumulated text"
        assert event.usage.total_tokens == 50
