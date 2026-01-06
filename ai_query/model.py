"""Model wrapper that combines provider and model name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_query.providers.base import BaseProvider


@dataclass
class LanguageModel:
    """A language model instance combining provider and model identifier.

    This is created by provider functions like google(), openai(), anthropic().
    """

    provider: BaseProvider
    model_id: str

    def __repr__(self) -> str:
        return f"LanguageModel({self.provider.name}/{self.model_id})"
