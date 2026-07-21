"""Tests for SSE event replay."""

import asyncio
import importlib.util
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock
from aiohttp import web
from ai_query.agents import Agent, MemoryStorage, AgentServer, AgentServerConfig, Event


class OrderedConnections(list):
    """List with set-like discard for deterministic broadcast-order tests."""

    def discard(self, item):
        if item in self:
            self.remove(item)


class TestSSEReplay:
    """Tests for SSE event replay functionality."""

    @pytest.fixture
    def event_agent_class(self):
        """Create an Agent class with event replay support."""

        class EventAgent(Agent):
            enable_event_log = True

            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        return EventAgent

    @pytest.mark.asyncio
    async def test_sse_replay_preserves_event_types(self, event_agent_class):
        """SSE replay should preserve event types."""
        agent = event_agent_class("test")
        await agent.start()

        await agent.emit("step_start", {"step": 1})
        await agent.emit("ai_chunk", {"content": "Hello"})
        await agent.emit("ai_chunk", {"content": " World"})
        await agent.emit("step_finish", {"step": 1})

        events = []
        async for event in agent.replay_events(after_id=1):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == "ai_chunk"
        assert events[0].data == {"content": "Hello"}
        assert events[1].type == "ai_chunk"
        assert events[2].type == "step_finish"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_sse_replay_filters_by_last_event_id(self, event_agent_class):
        """SSE replay should only send events after last_event_id."""
        agent = event_agent_class("test")
        await agent.start()

        for i in range(1, 11):
            await agent.emit("message", {"seq": i})

        events = []
        async for event in agent.replay_events(after_id=7):
            events.append(event)

        assert len(events) == 3
        assert events[0].data["seq"] == 8
        assert events[1].data["seq"] == 9
        assert events[2].data["seq"] == 10

        await agent.stop()

    @pytest.mark.asyncio
    async def test_sse_replay_with_router(self, aiohttp_client, event_agent_class):
        """SSE replay should work through the router."""
        server = AgentServer(event_agent_class)

        # Pre-create agent to emit events
        agent = server.get_or_create("test-replay")
        await agent.start()

        await agent.emit("step_start", {"step": 1})
        await agent.emit("ai_chunk", {"content": "hello"})

        config = AgentServerConfig()
        server._config = config
        app = server.create_app()

        client = await aiohttp_client(app)

        async with client.get("/agent/test-replay/events?last_event_id=0") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "text/event-stream"
            # We don't read the whole stream as it's infinite, just verify connection

    @pytest.mark.asyncio
    async def test_sse_replay_with_router_includes_event_ids(self, aiohttp_client, event_agent_class):
        """SSE replay should include id lines so last_event_id recovery remains lossless."""
        server = AgentServer(event_agent_class)

        agent = server.get_or_create("test-replay-ids")
        await agent.start()

        await agent.emit("step_start", {"step": 1})
        await agent.emit("ai_chunk", {"content": "hello"})

        config = AgentServerConfig()
        server._config = config
        app = server.create_app()
        client = await aiohttp_client(app)

        async with client.get("/agent/test-replay-ids/events?last_event_id=0") as resp:
            assert resp.status == 200
            first_chunk = await resp.content.readuntil(b"\n\n")
            text = first_chunk.decode("utf-8")
            assert "id: 1\n" in text
            assert "event: step_start\n" in text

    @pytest.mark.asyncio
    async def test_sse_replay_skips_non_replayable_events_but_keeps_id_gaps(self, aiohttp_client, event_agent_class):
        """SSE replay should omit replay=False events while keeping global ids monotonic."""
        server = AgentServer(event_agent_class)

        agent = server.get_or_create("test-replay-gaps")
        await agent.start()

        live_events = []

        async def handler(event: str, data: dict, event_id: int):
            live_events.append((event, data, event_id))

        agent._emit_handler = handler

        id1 = await agent.emit("status", {"seq": 1})
        id2 = await agent.emit("open_document", {"path": "README.md"}, replay=False)
        id3 = await agent.emit("status", {"seq": 3})

        assert [id1, id2, id3] == [1, 2, 3]
        assert [event_id for _, _, event_id in live_events] == [1, 2, 3]

        config = AgentServerConfig()
        server._config = config
        app = server.create_app()
        client = await aiohttp_client(app)

        async with client.get("/agent/test-replay-gaps/events?last_event_id=1") as resp:
            assert resp.status == 200
            first_chunk = await resp.content.readuntil(b"\n\n")
            text = first_chunk.decode("utf-8")
            assert "id: 3\n" in text
            assert "event: status\n" in text
            assert "open_document" not in text

    @pytest.mark.asyncio
    async def test_live_event_cannot_overtake_replayed_events(
        self, aiohttp_client, event_agent_class
    ):
        """Registration and replay should form one ordered delivery boundary."""
        server = AgentServer(event_agent_class)
        agent = server.get_or_create("test-replay-order")
        await agent.start()

        await agent.emit("message", {"seq": 1})
        await agent.emit("message", {"seq": 2})

        original_replay_events = agent.replay_events
        replay_paused = asyncio.Event()
        resume_replay = asyncio.Event()

        async def paused_replay(after_id: int = 0):
            async for event in original_replay_events(after_id):
                yield event
                if event.id == 1:
                    replay_paused.set()
                    await resume_replay.wait()

        agent.replay_events = paused_replay

        server._config = AgentServerConfig()
        client = await aiohttp_client(server.create_app())

        async with client.get(
            "/agent/test-replay-order/events?last_event_id=0"
        ) as resp:
            first = await resp.content.readuntil(b"\n\n")
            await replay_paused.wait()

            live_emit = asyncio.create_task(agent.emit("message", {"seq": 3}))
            await asyncio.sleep(0)
            assert not live_emit.done()

            resume_replay.set()
            second = await resp.content.readuntil(b"\n\n")
            third = await resp.content.readuntil(b"\n\n")
            await live_emit

        delivered = [first.decode(), second.decode(), third.decode()]
        assert [message.split("\n", 1)[0] for message in delivered] == [
            "id: 1",
            "id: 2",
            "id: 3",
        ]


class TestSSEEventFormatting:
    """Tests for SSE event formatting via emit handler."""

    @pytest.mark.asyncio
    async def test_emit_with_handler_formats_correctly(self):
        """emit() with an injected handler should receive correct data."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        received = []

        async def handler(event: str, data: dict, event_id: int):
            received.append((event, data, event_id))

        agent._emit_handler = handler

        await agent.emit("ai_chunk", {"content": "test"})

        assert len(received) == 1
        assert received[0][0] == "ai_chunk"
        assert received[0][1] == {"content": "test"}
        assert received[0][2] == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_with_handler_includes_event_id(self):
        """emit() should pass incrementing event IDs to handler."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        received = []

        async def handler(event: str, data: dict, event_id: int):
            received.append(event_id)

        agent._emit_handler = handler

        await agent.emit("message", {"text": "hi"})
        await agent.emit("message", {"text": "there"})
        await agent.emit("message", {"text": "friend"})

        assert received == [1, 2, 3]

        await agent.stop()


class TestSSEConnectionManagement:
    """Tests for SSE connection lifecycle."""

    @pytest.mark.asyncio
    async def test_sse_connection_registration(self):
        """SSE connections should be tracked in _sse_connections."""
        agent = Agent("test", storage=MemoryStorage())

        mock_sse = MagicMock()
        agent._sse_connections.add(mock_sse)

        assert mock_sse in agent._sse_connections
        assert len(agent._sse_connections) == 1

        agent._sse_connections.discard(mock_sse)

        assert mock_sse not in agent._sse_connections

    @pytest.mark.asyncio
    async def test_emit_handler_can_cleanup_failed_connections(self):
        """An emit handler can track and cleanup failed connections."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        good = MagicMock()
        good.write = AsyncMock()

        bad = MagicMock()
        bad.write = AsyncMock(side_effect=ConnectionResetError())

        agent._sse_connections = {good, bad}

        # Create a handler that mimics what server.py does
        async def handler(event: str, data: dict, event_id: int):
            sse_msg = f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
            for sse in list(agent._sse_connections):
                try:
                    await sse.write(sse_msg.encode())
                except Exception:
                    agent._sse_connections.discard(sse)

        agent._emit_handler = handler

        await agent.emit("test", {})

        assert good in agent._sse_connections
        assert bad not in agent._sse_connections

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_handler_sends_to_all_connections(self):
        """Emit handler should send to all SSE connections."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        connections = []
        for i in range(5):
            mock = MagicMock()
            mock.write = AsyncMock()
            connections.append(mock)

        agent._sse_connections = set(connections)

        # Create a handler that mimics what server.py does
        async def handler(event: str, data: dict, event_id: int):
            sse_msg = f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
            for sse in list(agent._sse_connections):
                try:
                    await sse.write(sse_msg.encode())
                except Exception:
                    agent._sse_connections.discard(sse)

        agent._emit_handler = handler

        await agent.emit("broadcast", {"msg": "hello all"})

        for conn in connections:
            conn.write.assert_called_once()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_server_emit_handler_isolates_cancelled_sse_connection(self):
        """A cancelled SSE write should not stop delivery to other SSE clients."""
        server = AgentServer(Agent)
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        bad = MagicMock()
        bad.write = AsyncMock(side_effect=asyncio.CancelledError())

        good = MagicMock()
        good.write = AsyncMock()

        agent._sse_connections = OrderedConnections([bad, good])
        agent._emit_handler = server._create_emit_handler(agent)

        await agent.emit("broadcast", {"msg": "hello"})

        bad.write.assert_called_once()
        good.write.assert_called_once()
        assert bad not in agent._sse_connections
        assert good in agent._sse_connections

        await agent.stop()

    @pytest.mark.asyncio
    async def test_legacy_server_emit_handler_isolates_cancelled_sse_connection(self):
        """The legacy single-agent server should isolate cancelled SSE writes too."""
        server_file = Path(__file__).parents[1] / "ai_query" / "agents" / "server.py"
        spec = importlib.util.spec_from_file_location("legacy_agent_server", server_file)
        assert spec is not None
        legacy_server = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(legacy_server)

        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        bad = MagicMock()
        bad.write = AsyncMock(side_effect=asyncio.CancelledError())

        good = MagicMock()
        good.write = AsyncMock()

        agent._sse_connections = OrderedConnections([bad, good])
        agent._emit_handler = legacy_server._create_emit_handler(agent)

        await agent.emit("broadcast", {"msg": "hello"})

        bad.write.assert_called_once()
        good.write.assert_called_once()
        assert bad not in agent._sse_connections
        assert good in agent._sse_connections

        await agent.stop()
