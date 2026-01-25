"""Configuration and types for AgentServer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from aiohttp import web


@dataclass
class AgentServerConfig:
    """Configuration for AgentServer lifecycle and security.

    Attributes:
        idle_timeout: Seconds before evicting idle agents (None = never).
        max_agents: Maximum concurrent agents (None = unlimited).
        auth: Async function to validate requests. Return True to allow, False to reject.
        allowed_origins: List of allowed CORS origins (None = allow all).
        base_path: Base path for agent routes (default: "/agent").
        enable_rest_api: Enable state REST endpoints (GET/PUT /agent/{id}/state).
        enable_list_agents: Enable GET /agents endpoint (security risk, off by default).

    Example:
        config = AgentServerConfig(
            idle_timeout=300,  # 5 minutes
            max_agents=100,
            auth=my_auth_function,
            allowed_origins=["https://myapp.com"],
        )
    """

    # Lifecycle
    idle_timeout: float | None = 300.0
    max_agents: int | None = None

    # Security
    auth: Callable[[web.Request], Awaitable[bool]] | None = None
    allowed_origins: list[str] | None = None

    # Routes
    base_path: str = "/agent"
    enable_rest_api: bool = True
    enable_list_agents: bool = False


@dataclass
class AgentMeta:
    """Internal metadata for tracking agent lifecycle."""
    agent: Any  # Agent instance
    last_activity: float = field(default_factory=time.time)
    connection_count: int = 0
