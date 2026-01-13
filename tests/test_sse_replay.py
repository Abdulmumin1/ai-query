"""Tests for SSE event replay."""

import json
import pytest
import time
from unittest.mock import MagicMock, AsyncMock
from aiohttp import web
from ai_query.agents import ChatAgent, InMemoryAgent
from ai_query.agents.router import AgentServer, AgentServerConfig
from ai_query.types import AgentEvent

@pytest.mark.asyncio
async def test_sse_replay_typed(aiohttp_client):
    """Test that SSE replay preserves event types."""

    class ReplayBot(ChatAgent, InMemoryAgent):
        enable_event_log = True
        initial_state = {}

    bot = ReplayBot("test-sse-replay")

    # Pre-populate events
    await bot._save_event(AgentEvent(
        id=10,
        type="step_start",
        data={"step": 1},
        created_at=time.time()
    ))
    await bot._save_event(AgentEvent(
        id=11,
        type="ai_chunk",
        data={"content": "hello"},
        created_at=time.time()
    ))

    # Setup server normally
    server = AgentServer(ReplayBot)

    # Pre-populate agent and events
    agent = server.get_or_create("test-sse-replay")
    await agent.start()

    await agent._save_event(AgentEvent(
        id=10,
        type="step_start",
        data={"step": 1},
        created_at=time.time()
    ))
    await agent._save_event(AgentEvent(
        id=11,
        type="ai_chunk",
        data={"content": "hello"},
        created_at=time.time()
    ))

    config = AgentServerConfig()
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_get(f"{base}/{{agent_id}}/events", server._handle_sse)

    client = await aiohttp_client(app)

    # Connect with last_event_id=5 (should get 10 and 11)
    # Use headers for Last-Event-ID if query param fails? No, logic uses query param.
    async with client.get(f"/agent/test-sse-replay/events?last_event_id=5") as resp:
        assert resp.status == 200

        # Read line by line to parse SSE
        lines = []
        for _ in range(10): # Read just enough lines
            line = await resp.content.readline()
            if not line: break
            lines.append(line.decode('utf-8'))

        text = "".join(lines)

        # Check for correctly formatted SSE events
        assert "event: step_start\n" in text
        assert 'data: {"step": 1}\n\n' in text

        assert "event: ai_chunk\n" in text
        assert 'data: {"content": "hello"}\n\n' in text
