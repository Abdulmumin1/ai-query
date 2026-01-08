"""Pytest configuration and fixtures for ai-query tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_query.types import (
    GenerateTextResult,
    Message,
    Usage,
    StreamChunk,
    Tool,
    ToolCall,
    ToolResult,
    ToolSet,
)
from ai_query.providers.base import BaseProvider
from ai_query.model import LanguageModel


# =============================================================================
# Async Event Loop
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock Provider
# =============================================================================


class MockProvider(BaseProvider):
    """Mock provider for testing without making real API calls."""

    name = "mock"

    def __init__(
        self,
        responses: list[GenerateTextResult] | None = None,
        stream_chunks: list[list[StreamChunk]] | None = None,
    ):
        super().__init__(api_key="mock-key")
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.call_count = 0
        self.stream_call_count = 0
        self.last_messages: list[Message] = []
        self.last_tools: ToolSet | None = None
        self.last_kwargs: dict[str, Any] = {}

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: dict | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        """Mock generate that returns pre-configured responses."""
        self.last_messages = messages
        self.last_tools = tools
        self.last_kwargs = kwargs

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = GenerateTextResult(
                text="Mock response",
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
                response={},
            )

        self.call_count += 1
        return response

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Mock stream that yields pre-configured chunks."""
        self.last_messages = messages
        self.last_tools = tools
        self.last_kwargs = kwargs

        if self.stream_call_count < len(self.stream_chunks):
            chunks = self.stream_chunks[self.stream_call_count]
        else:
            chunks = [
                StreamChunk(text="Mock "),
                StreamChunk(text="stream "),
                StreamChunk(text="response"),
                StreamChunk(
                    is_final=True,
                    usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
                    finish_reason="stop",
                ),
            ]

        self.stream_call_count += 1
        for chunk in chunks:
            yield chunk


@pytest.fixture
def mock_provider():
    """Create a basic mock provider."""
    return MockProvider()


@pytest.fixture
def mock_model(mock_provider):
    """Create a mock language model."""
    return LanguageModel(provider=mock_provider, model_id="mock-model")


# =============================================================================
# Response Factories
# =============================================================================


def make_response(
    text: str = "Test response",
    finish_reason: str = "stop",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> GenerateTextResult:
    """Factory for creating GenerateTextResult objects."""
    return GenerateTextResult(
        text=text,
        finish_reason=finish_reason,
        usage=usage or Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        response={"tool_calls": tool_calls or []},
    )


def make_tool_call(
    name: str,
    arguments: dict[str, Any],
    id: str = "call_123",
) -> ToolCall:
    """Factory for creating ToolCall objects."""
    return ToolCall(id=id, name=name, arguments=arguments)


def make_stream_chunks(
    text: str,
    tool_calls: list[ToolCall] | None = None,
) -> list[StreamChunk]:
    """Factory for creating stream chunks from text."""
    chunks = []
    words = text.split()
    for word in words:
        chunks.append(StreamChunk(text=word + " "))
    chunks.append(
        StreamChunk(
            is_final=True,
            usage=Usage(input_tokens=10, output_tokens=len(words), total_tokens=10 + len(words)),
            finish_reason="stop",
            tool_calls=tool_calls,
        )
    )
    return chunks


@pytest.fixture
def response_factory():
    """Fixture that returns the response factory function."""
    return make_response


@pytest.fixture
def tool_call_factory():
    """Fixture that returns the tool call factory function."""
    return make_tool_call


@pytest.fixture
def stream_chunks_factory():
    """Fixture that returns the stream chunks factory function."""
    return make_stream_chunks


# =============================================================================
# Sample Tools
# =============================================================================


@pytest.fixture
def sample_sync_tool():
    """A simple synchronous tool for testing."""
    from ai_query import tool, Field

    @tool(description="Add two numbers together")
    def add(
        a: int = Field(description="First number"),
        b: int = Field(description="Second number"),
    ) -> int:
        return a + b

    return add


@pytest.fixture
def sample_async_tool():
    """A simple asynchronous tool for testing."""
    from ai_query import tool, Field

    @tool(description="Multiply two numbers together")
    async def multiply(
        a: int = Field(description="First number"),
        b: int = Field(description="Second number"),
    ) -> int:
        return a * b

    return multiply


@pytest.fixture
def sample_tools(sample_sync_tool, sample_async_tool):
    """A ToolSet with sample tools."""
    return {
        "add": sample_sync_tool,
        "multiply": sample_async_tool,
    }


# =============================================================================
# Mock HTTP Responses
# =============================================================================


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
        },
    }


@pytest.fixture
def mock_google_response():
    """Mock Google API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help you?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }


# =============================================================================
# HTTP Mocking Helpers
# =============================================================================


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, json_data: dict, status: int = 200):
        self._json_data = json_data
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    async def json(self):
        return self._json_data

    async def text(self):
        return json.dumps(self._json_data)

    async def read(self):
        return json.dumps(self._json_data).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockStreamResponse:
    """Mock aiohttp streaming response."""

    def __init__(self, chunks: list[str], status: int = 200):
        self._chunks = chunks
        self.status = status
        self.headers = {"Content-Type": "text/event-stream"}
        self.content = self._create_content()

    def _create_content(self):
        """Create async iterator for content."""

        async def content_iter():
            for chunk in self._chunks:
                yield chunk.encode()

        return AsyncIterContent(content_iter())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class AsyncIterContent:
    """Async iterator wrapper for mock content."""

    def __init__(self, agen):
        self._agen = agen

    def __aiter__(self):
        return self._agen


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""

    class MockSession:
        def __init__(self):
            self.responses = []
            self.requests = []

        def add_response(self, response: MockResponse | MockStreamResponse):
            self.responses.append(response)

        async def post(self, url, **kwargs):
            self.requests.append(("POST", url, kwargs))
            if self.responses:
                return self.responses.pop(0)
            return MockResponse({"error": "No mock response configured"}, 500)

        async def get(self, url, **kwargs):
            self.requests.append(("GET", url, kwargs))
            if self.responses:
                return self.responses.pop(0)
            return MockResponse({"error": "No mock response configured"}, 500)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return MockSession()


# =============================================================================
# MCP Mocking
# =============================================================================


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools list."""

    class MockTool:
        def __init__(self, name: str, description: str, input_schema: dict):
            self.name = name
            self.description = description
            self.inputSchema = input_schema

    return [
        MockTool(
            name="calculator",
            description="Perform calculations",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
        MockTool(
            name="weather",
            description="Get weather info",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    ]


@pytest.fixture
def mock_mcp_session(mock_mcp_tools):
    """Mock MCP ClientSession."""

    class MockToolResult:
        def __init__(self, content):
            self.content = content

    class MockTextContent:
        def __init__(self, text):
            self.text = text

    class MockListToolsResponse:
        def __init__(self, tools):
            self.tools = tools

    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MockListToolsResponse(mock_mcp_tools))
    session.call_tool = AsyncMock(
        return_value=MockToolResult([MockTextContent("Tool result")])
    )

    return session
