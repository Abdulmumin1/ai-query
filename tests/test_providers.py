"""Tests for provider implementations."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from ai_query.types import Message, Usage, ToolCall, Tool
from ai_query.model import LanguageModel


# =============================================================================
# Base Provider Tests
# =============================================================================


class TestBaseProvider:
    """Tests for BaseProvider utility methods."""

    def test_get_provider_options(self):
        """get_provider_options should extract provider-specific options."""
        from ai_query.providers.base import BaseProvider

        class TestProvider(BaseProvider):
            name = "test"

            async def generate(self, **kwargs):
                pass

            async def stream(self, **kwargs):
                pass

        provider = TestProvider()

        # With matching provider
        options = provider.get_provider_options({
            "test": {"custom_param": "value"},
            "other": {"other_param": "ignored"},
        })
        assert options == {"custom_param": "value"}

        # Without matching provider
        options = provider.get_provider_options({
            "other": {"other_param": "ignored"},
        })
        assert options == {}

        # With None
        options = provider.get_provider_options(None)
        assert options == {}

    def test_parse_sse_line(self):
        """_parse_sse_line should extract data from SSE lines."""
        from ai_query.providers.base import BaseProvider

        class TestProvider(BaseProvider):
            name = "test"

            async def generate(self, **kwargs):
                pass

            async def stream(self, **kwargs):
                pass

        provider = TestProvider()

        # Valid data line (string)
        assert provider._parse_sse_line("data: hello") == "hello"

        # Valid data line (bytes)
        assert provider._parse_sse_line(b"data: hello") == "hello"

        # Empty line
        assert provider._parse_sse_line("") is None

        # Non-data line
        assert provider._parse_sse_line("event: message") is None

    def test_parse_sse_json(self):
        """_parse_sse_json should parse JSON from SSE lines."""
        from ai_query.providers.base import BaseProvider

        class TestProvider(BaseProvider):
            name = "test"

            async def generate(self, **kwargs):
                pass

            async def stream(self, **kwargs):
                pass

        provider = TestProvider()

        # Valid JSON
        result = provider._parse_sse_json('data: {"key": "value"}')
        assert result == {"key": "value"}

        # DONE marker
        assert provider._parse_sse_json("data: [DONE]") is None

        # Invalid JSON
        assert provider._parse_sse_json("data: not json") is None

    def test_accumulate_usage(self):
        """_accumulate_usage should add usage stats."""
        from ai_query.providers.base import BaseProvider

        class TestProvider(BaseProvider):
            name = "test"

            async def generate(self, **kwargs):
                pass

            async def stream(self, **kwargs):
                pass

        provider = TestProvider()

        total = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        delta = Usage(input_tokens=5, output_tokens=3, total_tokens=8)

        provider._accumulate_usage(total, delta)

        assert total.input_tokens == 15
        assert total.output_tokens == 8
        assert total.total_tokens == 23

    def test_apply_reasoning_unsupported_by_default(self):
        """BaseProvider should reject normalized reasoning unless implemented."""
        from ai_query.providers.base import BaseProvider

        class TestProvider(BaseProvider):
            name = "test"

            async def generate(self, **kwargs):
                pass

            async def stream(self, **kwargs):
                pass

        provider = TestProvider()

        with pytest.raises(ValueError, match="does not support normalized reasoning"):
            provider.apply_reasoning(None, {"effort": "high"}, model="test-model")


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_openai_function_creates_model(self):
        """openai() should create a LanguageModel."""
        from ai_query.providers import openai

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = openai("gpt-4o")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "gpt-4o"
        assert model.provider.name == "openai"

    def test_openai_with_custom_api_key(self):
        """openai() should accept custom API key."""
        from ai_query.providers import openai

        model = openai("gpt-4o", api_key="custom-key")
        assert model.provider.api_key == "custom-key"

    def test_openai_with_custom_base_url(self):
        """openai() should accept custom base URL."""
        from ai_query.providers import openai

        model = openai("gpt-4o", api_key="key", base_url="https://custom.api.com")
        assert model.provider.base_url == "https://custom.api.com"

    def test_openai_convert_tools(self):
        """OpenAI provider should convert tools correctly."""
        from ai_query.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test")

        tools = {
            "greet": Tool(
                description="Greet someone",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                execute=lambda name: f"Hello {name}",
            )
        }

        converted = provider._convert_tools(tools)

        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "greet"
        assert converted[0]["function"]["description"] == "Greet someone"

    @pytest.mark.asyncio
    async def test_openai_convert_messages_simple(self):
        """OpenAI provider should convert simple messages."""
        from ai_query.providers.openai import OpenAIProvider
        import aiohttp

        provider = OpenAIProvider(api_key="test")

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello!"),
        ]

        converted = await provider._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0] == {"role": "system", "content": "You are helpful."}
        assert converted[1] == {"role": "user", "content": "Hello!"}

    def test_openai_apply_reasoning_maps_effort(self):
        """OpenAI provider should map normalized reasoning effort."""
        from ai_query.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test")

        options = provider.apply_reasoning(None, {"effort": "high"}, model="gpt-5.4")

        assert options == {"openai": {"reasoning_effort": "high"}}

    def test_openai_apply_reasoning_rejects_conflict(self):
        """OpenAI provider should reject overlapping raw reasoning options."""
        from ai_query.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test")

        with pytest.raises(ValueError, match="Conflicting reasoning settings"):
            provider.apply_reasoning(
                {"openai": {"reasoning_effort": "low"}},
                {"effort": "high"},
                model="gpt-5.4",
            )

    @pytest.mark.asyncio
    async def test_openai_generate_maps_max_tokens_to_completion_tokens(self):
        """OpenAI provider should normalize max_tokens to max_completion_tokens."""
        from ai_query.providers.openai import OpenAIProvider

        class CaptureTransport:
            def __init__(self):
                self.last_json = None

            async def post(self, url, json, headers=None):
                self.last_json = json
                return {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            async def stream(self, url, json, headers=None):
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        transport = CaptureTransport()
        provider = OpenAIProvider(api_key="test", transport=transport)

        await provider.generate(
            model="gpt-5.4",
            messages=[Message(role="user", content="Hello")],
            max_tokens=123,
        )

        assert transport.last_json["max_completion_tokens"] == 123
        assert "max_tokens" not in transport.last_json

    @pytest.mark.asyncio
    async def test_openai_generate_rejects_conflicting_token_aliases(self):
        """OpenAI provider should reject conflicting token aliases."""
        from ai_query.providers.openai import OpenAIProvider

        class CaptureTransport:
            async def post(self, url, json, headers=None):
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

            async def stream(self, url, json, headers=None):
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        provider = OpenAIProvider(api_key="test", transport=CaptureTransport())

        with pytest.raises(ValueError, match="Conflicting token limit settings"):
            await provider.generate(
                model="gpt-5.4",
                messages=[Message(role="user", content="Hello")],
                max_tokens=100,
                max_completion_tokens=200,
            )

    @pytest.mark.asyncio
    async def test_openai_generate_uses_responses_for_reasoning_effort_with_tools(self):
        """OpenAI should route tools with reasoning_effort through Responses."""
        from ai_query.providers.openai import OpenAIProvider

        class CaptureTransport:
            def __init__(self):
                self.last_url = None
                self.last_json = None

            async def post(self, url, json, headers=None):
                self.last_url = url
                self.last_json = json
                return {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "id": "rs_1", "summary": []},
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "greet",
                            "arguments": "{}",
                        },
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }

            async def stream(self, url, json, headers=None):
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        transport = CaptureTransport()
        provider = OpenAIProvider(api_key="test", transport=transport)
        tools = {
            "greet": Tool(
                description="Greet someone",
                parameters={"type": "object", "properties": {}},
                execute=lambda: "hello",
            )
        }

        result = await provider.generate(
            model="gpt-5.4",
            messages=[Message(role="user", content="Hello")],
            tools=tools,
            provider_options={"openai": {"reasoning_effort": "high"}},
            max_tokens=123,
        )

        assert transport.last_url.endswith("/responses")
        assert transport.last_json["reasoning"] == {"effort": "high"}
        assert transport.last_json["max_output_tokens"] == 123
        assert transport.last_json["tools"] == [
            {
                "type": "function",
                "name": "greet",
                "description": "Greet someone",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert result.response["tool_calls"][0].id == "call_1"
        assert result.response["tool_calls"][0].metadata["openai_response_output"][0]["type"] == "reasoning"

    @pytest.mark.asyncio
    async def test_openai_stream_uses_responses_for_reasoning_effort_with_tools(self):
        """OpenAI streaming should use Responses fallback for tools with reasoning."""
        from ai_query.providers.openai import OpenAIProvider

        class CaptureTransport:
            def __init__(self):
                self.last_url = None

            async def post(self, url, json, headers=None):
                self.last_url = url
                return {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output_text": "Hello from a tool-capable reasoning response.",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Hello from a tool-capable reasoning response.",
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }

            async def stream(self, url, json, headers=None):
                raise AssertionError("request should not be sent")
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        transport = CaptureTransport()
        provider = OpenAIProvider(api_key="test", transport=transport)
        tools = {
            "greet": Tool(
                description="Greet someone",
                parameters={"type": "object", "properties": {}},
                execute=lambda: "hello",
            )
        }

        chunks = []
        async for chunk in provider.stream(
            model="gpt-5.4",
            messages=[Message(role="user", content="Hello")],
            tools=tools,
            provider_options={"openai": {"reasoning_effort": "high"}},
        ):
            chunks.append(chunk)

        assert transport.last_url.endswith("/responses")
        assert chunks[0].text == "Hello from a tool-capable reasoning response."
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_openai_responses_round_trips_reasoning_items_with_tool_outputs(self):
        """OpenAI Responses tool loops should preserve reasoning items across turns."""
        from ai_query import Field, generate_text, step_count_is, tool
        from ai_query.providers.openai import OpenAIProvider

        class CaptureTransport:
            def __init__(self):
                self.requests = []

            async def post(self, url, json, headers=None):
                self.requests.append(json)
                if len(self.requests) == 1:
                    return {
                        "id": "resp_1",
                        "model": "gpt-5.4",
                        "status": "completed",
                        "output": [
                            {"type": "reasoning", "id": "rs_1", "summary": []},
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "greet",
                                "arguments": '{"name":"Ada"}',
                            },
                        ],
                    }
                return {
                    "id": "resp_2",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output_text": "Hello Ada.",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Hello Ada."}],
                        }
                    ],
                }

            async def stream(self, url, json, headers=None):
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        @tool(description="Greet someone")
        def greet(name: str = Field(description="Name")) -> str:
            return f"Hello {name}."

        transport = CaptureTransport()
        provider = OpenAIProvider(api_key="test", transport=transport)
        model = LanguageModel(provider=provider, model_id="gpt-5.4")

        result = await generate_text(
            model=model,
            prompt="Greet Ada",
            tools={"greet": greet},
            reasoning={"effort": "high"},
            stop_when=step_count_is(10),
        )

        second_input = transport.requests[1]["input"]
        assert result.text == "Hello Ada."
        assert any(item.get("type") == "reasoning" for item in second_input)
        assert any(item.get("type") == "function_call" for item in second_input)
        assert any(item.get("type") == "function_call_output" for item in second_input)


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_anthropic_function_creates_model(self):
        """anthropic() should create a LanguageModel."""
        from ai_query.providers import anthropic

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            model = anthropic("claude-sonnet-4-20250514")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "claude-sonnet-4-20250514"
        assert model.provider.name == "anthropic"

    def test_anthropic_with_custom_api_key(self):
        """anthropic() should accept custom API key."""
        from ai_query.providers import anthropic

        model = anthropic("claude-sonnet-4-20250514", api_key="custom-key")
        assert model.provider.api_key == "custom-key"

    def test_anthropic_convert_tools(self):
        """Anthropic provider should convert tools correctly."""
        from ai_query.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        tools = {
            "search": Tool(
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                execute=lambda query: f"Results for {query}",
            )
        }

        converted = provider._convert_tools(tools)

        assert len(converted) == 1
        assert converted[0]["name"] == "search"
        assert converted[0]["description"] == "Search the web"
        assert "input_schema" in converted[0]

    def test_anthropic_apply_reasoning_maps_effort(self):
        """Anthropic provider should map normalized effort generically."""
        from ai_query.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        options = provider.apply_reasoning(None, {"effort": "medium"}, model="claude-opus-4-7")

        assert options == {
            "anthropic": {
                "output_config": {"effort": "medium"},
            }
        }

    def test_anthropic_apply_reasoning_maps_budget(self):
        """Anthropic provider should map normalized budget generically."""
        from ai_query.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        options = provider.apply_reasoning(None, {"budget": 8000}, model="claude-3-7-sonnet-20250219")

        assert options == {
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 8000},
            }
        }

    def test_anthropic_apply_reasoning_maps_effort_and_budget_together(self):
        """Anthropic provider should map both normalized fields without model gating."""
        from ai_query.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        options = provider.apply_reasoning(
            None,
            {"effort": "high", "budget": 8000},
            model="claude-opus-4-7",
        )

        assert options == {
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 8000},
                "output_config": {"effort": "high"},
            }
        }

    def test_anthropic_apply_reasoning_rejects_conflict(self):
        """Anthropic provider should reject overlap with raw thinking config."""
        from ai_query.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        with pytest.raises(ValueError, match="Conflicting reasoning settings"):
            provider.apply_reasoning(
                {"anthropic": {"thinking": {"type": "adaptive"}}},
                {"effort": "medium"},
                model="claude-opus-4-7",
            )

    @pytest.mark.asyncio
    async def test_anthropic_stream_emits_reasoning_events(self):
        """Anthropic stream should expose thinking deltas as reasoning events."""
        from ai_query.providers.anthropic import AnthropicProvider

        class FakeTransport:
            async def post(self, url, json, headers=None):
                return {}

            async def stream(self, url, json, headers=None):
                chunks = [
                    b'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n',
                    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}\n',
                    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}\n',
                    b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}\n',
                    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n',
                ]
                for chunk in chunks:
                    yield chunk

        provider = AnthropicProvider(api_key="test", transport=FakeTransport())

        chunks = []
        async for chunk in provider.stream(
            model="claude-sonnet-4-20250514",
            messages=[Message(role="user", content="Hello")],
        ):
            chunks.append(chunk)

        reasoning_events = [
            event
            for chunk in chunks
            for event in (chunk.reasoning_events or [])
        ]

        assert chunks[2].text == "Answer"
        assert reasoning_events[0].kind == "delta"
        assert reasoning_events[0].text == "think"
        assert reasoning_events[1].kind == "signature"
        assert reasoning_events[1].data == {"signature": "sig"}
        assert chunks[-1].is_final is True


# =============================================================================
# Workers AI Provider Tests
# =============================================================================


class TestWorkersAIProvider:
    """Tests for Workers AI provider."""

    def test_workers_ai_function_creates_model(self):
        """workers_ai() should create a LanguageModel."""
        from ai_query.providers import workers_ai

        with patch.dict(
            "os.environ",
            {
                "CLOUDFLARE_ACCOUNT_ID": "account-123",
                "CLOUDFLARE_API_TOKEN": "test-token",
            },
        ):
            model = workers_ai("@cf/moonshotai/kimi-k2.5")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "@cf/moonshotai/kimi-k2.5"
        assert model.provider.name == "workers_ai"

    @pytest.mark.asyncio
    async def test_workers_ai_generate_uses_max_tokens_upstream(self):
        """Workers AI should keep upstream max_tokens for OpenAI-compatible requests."""
        from ai_query.providers.workers_ai import WorkersAIProvider

        class CaptureTransport:
            def __init__(self):
                self.last_json = None

            async def post(self, url, json, headers=None):
                self.last_json = json
                return {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            async def stream(self, url, json, headers=None):
                if False:
                    yield b""

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        transport = CaptureTransport()
        provider = WorkersAIProvider(
            api_key="token",
            account_id="account-123",
            transport=transport,
        )

        await provider.generate(
            model="@cf/moonshotai/kimi-k2.5",
            messages=[Message(role="user", content="Hello")],
            max_tokens=123,
        )

        assert transport.last_json["max_tokens"] == 123
        assert "max_completion_tokens" not in transport.last_json

    @pytest.mark.asyncio
    async def test_workers_ai_stream_emits_openai_compatible_reasoning_events(self):
        """Workers AI should expose OpenAI-compatible reasoning deltas."""
        from ai_query.providers.workers_ai import WorkersAIProvider

        class FakeTransport:
            async def post(self, url, json, headers=None):
                return {}

            async def stream(self, url, json, headers=None):
                chunks = [
                    b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n',
                ]
                for chunk in chunks:
                    yield chunk

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        provider = WorkersAIProvider(
            api_key="token",
            account_id="account-123",
            transport=FakeTransport(),
        )

        chunks = []
        async for chunk in provider.stream(
            model="@cf/moonshotai/kimi-k2.5",
            messages=[Message(role="user", content="Hello")],
        ):
            chunks.append(chunk)

        reasoning_events = [
            event
            for chunk in chunks
            for event in (chunk.reasoning_events or [])
        ]

        assert chunks[1].text == "Answer"
        assert reasoning_events[0].kind == "delta"
        assert reasoning_events[0].provider == "workers_ai"
        assert reasoning_events[0].text == "think"
        assert reasoning_events[0].data == {"field": "reasoning_content"}

    @pytest.mark.asyncio
    async def test_workers_ai_stream_ignores_null_tool_call_fragments(self):
        """Workers AI should ignore null tool-call fragments in streamed deltas."""
        from ai_query.providers.workers_ai import WorkersAIProvider

        class FakeTransport:
            async def post(self, url, json, headers=None):
                return {}

            async def stream(self, url, json, headers=None):
                chunks = [
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":null,"function":{"name":null,"arguments":null}}]}}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\\"a\\":1"}}]}}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\\"b\\":2}"}}]}}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n',
                ]
                for chunk in chunks:
                    yield chunk

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        provider = WorkersAIProvider(
            api_key="token",
            account_id="account-123",
            transport=FakeTransport(),
        )

        chunks = []
        async for chunk in provider.stream(
            model="@cf/moonshotai/kimi-k2.5",
            messages=[Message(role="user", content="Hello")],
        ):
            chunks.append(chunk)

        assert chunks[-1].tool_calls[0].id == "call_1"
        assert chunks[-1].tool_calls[0].name == "add"
        assert chunks[-1].tool_calls[0].arguments == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_workers_ai_stream_ignores_null_delta_and_function(self):
        """Workers AI should ignore null delta and function objects in streamed chunks."""
        from ai_query.providers.workers_ai import WorkersAIProvider

        class FakeTransport:
            async def post(self, url, json, headers=None):
                return {}

            async def stream(self, url, json, headers=None):
                chunks = [
                    b'data: {"choices":[{"delta":null}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[null,{"index":0,"function":null}]}}]}\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\\"a\\":1,\\"b\\":2}"}}]}}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n',
                ]
                for chunk in chunks:
                    yield chunk

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        provider = WorkersAIProvider(
            api_key="token",
            account_id="account-123",
            transport=FakeTransport(),
        )

        chunks = []
        async for chunk in provider.stream(
            model="@cf/moonshotai/kimi-k2.5",
            messages=[Message(role="user", content="Hello")],
        ):
            chunks.append(chunk)

        assert chunks[-1].tool_calls[0].id == "call_1"
        assert chunks[-1].tool_calls[0].name == "add"
        assert chunks[-1].tool_calls[0].arguments == {"a": 1, "b": 2}

    def test_workers_ai_with_custom_base_url(self):
        """workers_ai() should accept custom base URL without account_id."""
        from ai_query.providers import workers_ai

        model = workers_ai(
            "@cf/moonshotai/kimi-k2.5",
            api_key="token",
            base_url="https://gateway.ai.cloudflare.com/v1/account/gateway/workers-ai",
        )

        assert model.provider.base_url == (
            "https://gateway.ai.cloudflare.com/v1/account/gateway/workers-ai"
        )

    def test_workers_ai_embedding_creates_model(self):
        """workers_ai.embedding() should create an EmbeddingModel."""
        from ai_query.providers import workers_ai
        from ai_query.model import EmbeddingModel

        model = workers_ai.embedding(
            "@cf/baai/bge-large-en-v1.5",
            api_key="token",
            account_id="account-123",
        )

        assert isinstance(model, EmbeddingModel)
        assert model.model_id == "@cf/baai/bge-large-en-v1.5"
        assert model.provider.name == "workers_ai"

    def test_workers_ai_requires_account_id_without_base_url(self):
        """WorkersAIProvider should require account_id when base_url is absent."""
        from ai_query.providers.workers_ai import WorkersAIProvider

        with pytest.raises(ValueError, match="Cloudflare account ID is missing"):
            WorkersAIProvider(api_key="token")


class TestGoogleReasoningProvider:
    """Tests for normalized reasoning mapping in Google provider."""

    def test_google_apply_reasoning_maps_budget(self):
        """Google provider should map normalized reasoning budget."""
        from ai_query.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")

        options = provider.apply_reasoning(None, {"budget": 1024}, model="gemini-2.5-pro")

        assert options == {"google": {"thinking_config": {"thinking_budget": 1024}}}

    def test_google_apply_reasoning_rejects_conflict(self):
        """Google provider should reject overlapping raw thinking config."""
        from ai_query.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")

        with pytest.raises(ValueError, match="Conflicting reasoning settings"):
            provider.apply_reasoning(
                {"google": {"thinking_config": {"thinking_budget": 2048}}},
                {"budget": 1024},
                model="gemini-2.5-pro",
            )

    def test_google_apply_reasoning_rejects_effort(self):
        """Google provider should reject normalized effort until mapped explicitly."""
        from ai_query.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")

        with pytest.raises(ValueError, match="does not support normalized reasoning.effort"):
            provider.apply_reasoning(None, {"effort": "high"}, model="gemini-2.5-pro")

    @pytest.mark.asyncio
    async def test_google_stream_emits_thought_reasoning_events(self):
        """Google stream should expose thought parts as reasoning events."""
        from ai_query.providers.google import GoogleProvider

        class FakeTransport:
            async def post(self, url, json, headers=None):
                return {}

            async def stream(self, url, json, headers=None):
                chunks = [
                    b'data: {"candidates":[{"content":{"parts":[{"text":"think","thought":true,"thoughtSignature":"sig"}]}}]}\n',
                    b'data: {"candidates":[{"content":{"parts":[{"text":"Answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}\n',
                ]
                for chunk in chunks:
                    yield chunk

            async def get(self, url, headers=None):
                return b"", "application/octet-stream"

        provider = GoogleProvider(api_key="test", transport=FakeTransport())

        chunks = []
        async for chunk in provider.stream(
            model="gemini-2.5-pro",
            messages=[Message(role="user", content="Hello")],
        ):
            chunks.append(chunk)

        reasoning_events = [
            event
            for chunk in chunks
            for event in (chunk.reasoning_events or [])
        ]

        assert chunks[2].text == "Answer"
        assert reasoning_events[0].kind == "delta"
        assert reasoning_events[0].provider == "google"
        assert reasoning_events[0].text == "think"
        assert reasoning_events[1].kind == "signature"
        assert reasoning_events[1].data == {
            "field": "thoughtSignature",
            "signature": "sig",
        }


# =============================================================================
# Google Provider Tests
# =============================================================================


class TestGoogleProvider:
    """Tests for Google provider."""

    def test_google_function_creates_model(self):
        """google() should create a LanguageModel."""
        from ai_query.providers import google

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            model = google("gemini-2.0-flash")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "gemini-2.0-flash"
        assert model.provider.name == "google"

    def test_google_with_custom_api_key(self):
        """google() should accept custom API key."""
        from ai_query.providers import google

        model = google("gemini-2.0-flash", api_key="custom-key")
        assert model.provider.api_key == "custom-key"

    def test_google_convert_tools(self):
        """Google provider should convert tools correctly."""
        from ai_query.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")

        tools = {
            "calculate": Tool(
                description="Calculate math",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                execute=lambda expression: str(eval(expression)),
            )
        }

        converted = provider._convert_tools(tools)

        # Google uses functionDeclarations
        assert "functionDeclarations" in converted[0]
        assert len(converted[0]["functionDeclarations"]) == 1
        assert converted[0]["functionDeclarations"][0]["name"] == "calculate"


# =============================================================================
# Provider Integration Tests (Mocked HTTP)
# =============================================================================


class TestProviderIntegration:
    """Integration tests for providers with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_openai_generate_mocked(self, mock_openai_response):
        """OpenAI generate should work with mocked response."""
        from ai_query.providers.openai import OpenAIProvider
        from tests.conftest import MockResponse

        provider = OpenAIProvider(api_key="test-key")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session.post = MagicMock(return_value=MockResponse(mock_openai_response))
            mock_session_class.return_value = mock_session

            result = await provider.generate(
                model="gpt-4o",
                messages=[Message(role="user", content="Hello")],
            )

        assert result.text == "Hello! How can I help you?"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_anthropic_generate_mocked(self, mock_anthropic_response):
        """Anthropic generate should work with mocked response."""
        from ai_query.providers.anthropic import AnthropicProvider
        from tests.conftest import MockResponse

        provider = AnthropicProvider(api_key="test-key")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session.post = MagicMock(return_value=MockResponse(mock_anthropic_response))
            mock_session_class.return_value = mock_session

            result = await provider.generate(
                model="claude-sonnet-4-20250514",
                messages=[Message(role="user", content="Hello")],
            )

        assert result.text == "Hello! How can I help you?"
        assert result.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_google_generate_mocked(self, mock_google_response):
        """Google generate should work with mocked response."""
        from ai_query.providers.google import GoogleProvider
        from tests.conftest import MockResponse

        provider = GoogleProvider(api_key="test-key")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session.post = MagicMock(return_value=MockResponse(mock_google_response))
            mock_session_class.return_value = mock_session

            result = await provider.generate(
                model="gemini-2.0-flash",
                messages=[Message(role="user", content="Hello")],
            )

        assert result.text == "Hello! How can I help you?"
        assert result.finish_reason == "STOP"


# =============================================================================
# Provider Error Handling Tests
# =============================================================================


class TestProviderErrors:
    """Tests for provider error handling."""

    @pytest.mark.asyncio
    async def test_openai_api_error(self):
        """OpenAI provider should handle API errors."""
        from ai_query.providers.openai import OpenAIProvider
        from tests.conftest import MockResponse

        provider = OpenAIProvider(api_key="test-key")

        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
            }
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=MockResponse(error_response, status=401))
            mock_session_class.return_value = mock_session

            with pytest.raises(Exception) as exc_info:
                await provider.generate(
                    model="gpt-4o",
                    messages=[Message(role="user", content="Hello")],
                )

            assert "401" in str(exc_info.value) or "Invalid API key" in str(exc_info.value)
