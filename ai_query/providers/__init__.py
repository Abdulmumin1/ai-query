"""Provider adapters for ai-query."""

from ai_query.providers.base import BaseProvider
from ai_query.providers.openai import OpenAIProvider, openai
from ai_query.providers.anthropic import AnthropicProvider, anthropic
from ai_query.providers.google import GoogleProvider, google

__all__ = [
    # Base class for custom providers
    "BaseProvider",
    # Provider classes
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    # Factory functions
    "openai",
    "anthropic",
    "google",
]
