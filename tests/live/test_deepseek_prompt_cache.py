"""Opt-in live verification for DeepSeek's automatic prompt cache.

Run with:
    DEEPSEEK_API_KEY=... uv run pytest -q tests/live/test_deepseek_prompt_cache.py
"""

from __future__ import annotations

import os

import pytest

from ai_query import generate_text
from ai_query.model import LanguageModel
from ai_query.providers.deepseek import DeepSeekProvider
from ai_query.providers.openai import OpenAIProvider


pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is required for this live probe",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["deepseek", "openai-compatible"])
async def test_deepseek_repeated_prefix_reports_cache_usage(adapter: str) -> None:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    if adapter == "deepseek":
        provider = DeepSeekProvider(api_key=api_key)
    else:
        provider = OpenAIProvider(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    model = LanguageModel(
        provider=provider,
        model_id=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    )
    stable_prefix = (
        "Stable cache probe context. Preserve this exact prefix across requests. "
        * 700
    )
    usages = []

    for run in range(2):
        result = await generate_text(
            model=model,
            system=stable_prefix,
            prompt=f"Probe {run}: reply briefly.",
            max_tokens=64,
            temperature=0,
        )
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.cached_tokens >= 0
        assert result.usage.cache_miss_tokens >= 0
        assert (
            result.usage.cached_tokens + result.usage.cache_miss_tokens
            == result.usage.input_tokens
        )
        usages.append(result.usage)

    # Cache placement and timing are provider-controlled. A hit is diagnostic,
    # not a reliable test invariant, so the probe only records structural truth.
    assert len(usages) == 2
