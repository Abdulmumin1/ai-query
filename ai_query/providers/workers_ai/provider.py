"""Cloudflare Workers AI provider using OpenAI-compatible endpoints."""

from __future__ import annotations

import os
from typing import Any

from ai_query.model import EmbeddingModel, LanguageModel
from ai_query.providers.openai.provider import OpenAIProvider


def _resolve_cloudflare_api_key(api_key: str | None) -> str | None:
    return (
        api_key
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        or os.environ.get("CLOUDFLARE_API_KEY")
    )


class _WorkersAINamespace:
    """Namespace for Workers AI provider functions.

    Provides both language model and embedding model factory functions.

    Example:
        >>> from ai_query.providers import workers_ai
        >>> model = workers_ai("@cf/meta/llama-3.1-8b-instruct")
        >>> embedding_model = workers_ai.embedding("@cf/baai/bge-large-en-v1.5")
    """

    def __call__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> LanguageModel:
        global _default_provider

        if api_key or account_id or base_url:
            provider = WorkersAIProvider(
                api_key=api_key,
                account_id=account_id,
                base_url=base_url,
            )
        else:
            if _default_provider is None:
                _default_provider = WorkersAIProvider()
            provider = _default_provider

        return LanguageModel(provider=provider, model_id=model_id)

    def embedding(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> EmbeddingModel:
        global _default_embedding_provider

        if api_key or account_id or base_url:
            provider = WorkersAIProvider(
                api_key=api_key,
                account_id=account_id,
                base_url=base_url,
            )
        else:
            if _default_embedding_provider is None:
                _default_embedding_provider = WorkersAIProvider()
            provider = _default_embedding_provider

        return EmbeddingModel(provider=provider, model_id=model_id)


class WorkersAIProvider(OpenAIProvider):
    """Cloudflare Workers AI provider via OpenAI-compatible endpoints."""

    name = "workers_ai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        account_id: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        resolved_api_key = _resolve_cloudflare_api_key(api_key)
        if not resolved_api_key:
            raise ValueError(
                "Error: Cloudflare API token is missing. Pass it using the 'api_key' parameter "
                "or set CLOUDFLARE_API_TOKEN, CLOUDFLARE_AUTH_TOKEN, or CLOUDFLARE_API_KEY."
            )

        resolved_account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        resolved_base_url = base_url
        if resolved_base_url is None:
            if not resolved_account_id:
                raise ValueError(
                    "Error: Cloudflare account ID is missing. Pass it using the 'account_id' "
                    "parameter or set the CLOUDFLARE_ACCOUNT_ID environment variable."
                )
            resolved_base_url = (
                f"https://api.cloudflare.com/client/v4/accounts/{resolved_account_id}/ai/v1"
            )

        super().__init__(api_key=resolved_api_key, base_url=resolved_base_url, **kwargs)
        self.account_id = resolved_account_id


_default_provider: WorkersAIProvider | None = None
_default_embedding_provider: WorkersAIProvider | None = None


workers_ai = _WorkersAINamespace()
