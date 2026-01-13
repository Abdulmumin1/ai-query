"""Tests for event persistence and replay."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from ai_query.agents import ChatAgent, InMemoryAgent, AgentServer
from ai_query.agents.output import WebSocketOutput
from ai_query.agents.router import AgentServerConfig
from aiohttp import web

class ReplayBot(ChatAgent, InMemoryAgent):
    enable_event_log = True
    initial_state = {}

    async def on_step_start(self, event):
        await self.output.send_status("Thinking...")
        # Send a message to verify message persistence too
        await self.output.send_message("Debug message")

@pytest.fixture(autouse=True)
def cleanup():
    """Clear storage before each test."""
    InMemoryAgent.clear_all()
    yield
    InMemoryAgent.clear_all()

@pytest.mark.asyncio
async def test_event_persistence():
    """Events should be saved when enable_event_log is True."""
    from unittest.mock import patch, MagicMock

    # Mock generate_text
    async def mock_generate_text(*args, **kwargs):
        # Call on_step_start if provided
        if "on_step_start" in kwargs and kwargs["on_step_start"]:
            from ai_query.types import StepStartEvent
            await kwargs["on_step_start"](StepStartEvent(step_number=1, messages=[], tools=None))

        mock_result = MagicMock()
        mock_result.text = "Hello there!"
        return mock_result

    with patch("ai_query.generate_text", side_effect=mock_generate_text):
        async with ReplayBot("test-replay") as agent:
            # Manually trigger output
            mock_conn = AsyncMock()
            output = WebSocketOutput(mock_conn)

            # This will wrap the output in PersistingOutput
            await agent.chat("Hello", output=output)

            # Check if events were saved
            events = await agent._get_events()
            assert len(events) > 0

            # Verify we captured the status event from the hook
            status_events = [e for e in events if e.type == "status"]
            assert len(status_events) > 0
            assert status_events[0].data["status"] == "Thinking..."

            # Verify we captured the message event
            message_events = [e for e in events if e.type == "message"]
            assert len(message_events) > 0
            assert message_events[0].data["content"] == "Debug message"

@pytest.mark.asyncio
async def test_manual_event_saving():
    """Test manual saving via _save_event."""
    async with ReplayBot("test-manual") as agent:
        from ai_query.types import AgentEvent
        import time

        event = AgentEvent(
            id=1,
            type="custom",
            data={"foo": "bar"},
            created_at=time.time()
        )

        await agent._save_event(event)

        events = await agent._get_events()
        assert len(events) == 1
        assert events[0].id == 1
        assert events[0].data["foo"] == "bar"

@pytest.mark.asyncio
async def test_replay_logic():
    """Test replay_events sends correct events."""
    async with ReplayBot("test-replay-logic") as agent:
        # Create some history
        from ai_query.types import AgentEvent
        import time

        for i in range(1, 4):
            await agent._save_event(AgentEvent(
                id=i,
                type="msg",
                data={"seq": i},
                created_at=time.time()
            ))

        # Replay from ID 1 (should get 2 and 3)
        mock_conn = MagicMock()
        mock_conn.send = AsyncMock()

        await agent.replay_events(mock_conn, after_id=1)

        assert mock_conn.send.call_count == 2

        # Verify call args
        args_list = mock_conn.send.call_args_list
        call1 = json.loads(args_list[0][0][0])
        call2 = json.loads(args_list[1][0][0])

        assert call1["seq"] == 2
        assert call2["seq"] == 3

@pytest.mark.asyncio
async def test_router_connection_recovery(aiohttp_client):
    """Test that router handles last_event_id."""

    # 1. Setup server
    server = AgentServer(ReplayBot)

    # Pre-populate agent with events
    agent = server.get_or_create("user-1")
    await agent.start()

    from ai_query.types import AgentEvent
    import time

    # Add events 1, 2, 3
    for i in range(1, 4):
        await agent._save_event(AgentEvent(
            id=i,
            type="status",
            data={"status": f"step {i}"},
            created_at=time.time()
        ))

    # 2. Connect with last_event_id=1
    config = AgentServerConfig()
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_get(f"{base}/{{agent_id}}/ws", server._handle_websocket)

    client = await aiohttp_client(app)

    async with client.ws_connect("/agent/user-1/ws?last_event_id=1") as ws:
        # We should receive events 2 and 3 immediately
        msg1 = await ws.receive_json()
        msg2 = await ws.receive_json()

        assert msg1["status"] == "step 2"
        assert msg2["status"] == "step 3"

        # Close
        await ws.close()
