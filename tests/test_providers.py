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


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_openai_function_creates_model(self):
        """openai() should create a LanguageModel."""
        from ai_query import openai

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = openai("gpt-4o")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "gpt-4o"
        assert model.provider.name == "openai"

    def test_openai_with_custom_api_key(self):
        """openai() should accept custom API key."""
        from ai_query import openai

        model = openai("gpt-4o", api_key="custom-key")
        assert model.provider.api_key == "custom-key"

    def test_openai_with_custom_base_url(self):
        """openai() should accept custom base URL."""
        from ai_query import openai

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

        async with aiohttp.ClientSession() as session:
            converted = await provider._convert_messages(messages, session)

        assert len(converted) == 2
        assert converted[0] == {"role": "system", "content": "You are helpful."}
        assert converted[1] == {"role": "user", "content": "Hello!"}


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_anthropic_function_creates_model(self):
        """anthropic() should create a LanguageModel."""
        from ai_query import anthropic

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            model = anthropic("claude-sonnet-4-20250514")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "claude-sonnet-4-20250514"
        assert model.provider.name == "anthropic"

    def test_anthropic_with_custom_api_key(self):
        """anthropic() should accept custom API key."""
        from ai_query import anthropic

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


# =============================================================================
# Google Provider Tests
# =============================================================================


class TestGoogleProvider:
    """Tests for Google provider."""

    def test_google_function_creates_model(self):
        """google() should create a LanguageModel."""
        from ai_query import google

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            model = google("gemini-2.0-flash")

        assert isinstance(model, LanguageModel)
        assert model.model_id == "gemini-2.0-flash"
        assert model.provider.name == "google"

    def test_google_with_custom_api_key(self):
        """google() should accept custom API key."""
        from ai_query import google

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
