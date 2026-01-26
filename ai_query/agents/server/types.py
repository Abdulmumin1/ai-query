"""Configuration and types for AgentServer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

from aiohttp import web


@dataclass
class AgentServerConfig:
    """Configuration for AgentServer lifecycle and security."""

    # Lifecycle
    idle_timeout: Union[float, None] = 300.0
    max_agents: Union[int, None] = None

    # Security
    auth: Union[Callable[[web.Request], Awaitable[bool]], None] = None
    allowed_origins: Union[list[str], None] = None

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
