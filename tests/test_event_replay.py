"""Tests for event persistence and replay."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_query.agents import Agent, MemoryStorage, Connection, Event


class TestEventLogBuiltin:
    """Tests for built-in event log functionality."""

    @pytest.mark.asyncio
    async def test_enable_event_log_default_false(self):
        """enable_event_log should default to False."""
        agent = Agent("test", storage=MemoryStorage())
        assert agent.enable_event_log is False

    @pytest.mark.asyncio
    async def test_enable_event_log_class_attribute(self):
        """enable_event_log can be set as class attribute."""

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=MemoryStorage())
        assert agent.enable_event_log is True

    @pytest.mark.asyncio
    async def test_emit_increments_counter(self):
        """emit() should increment event counter."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        id1 = await agent.emit("message", {"content": "Hello"})
        assert id1 == 1

        id2 = await agent.emit("message", {"content": "World"})
        assert id2 == 2

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_appends_to_log(self):
        """emit() should append to _event_log."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("message", {"content": "Hello"})
        await agent.emit("status", {"status": "done"})

        assert len(agent._event_log) == 2
        assert agent._event_log[0]["type"] == "message"
        assert agent._event_log[1]["type"] == "status"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_persists_when_enabled(self):
        """emit() should persist to storage when enable_event_log is True."""
        storage = MemoryStorage()

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        await agent.emit("message", {"content": "Hello"})

        stored = await storage.get("test:event_log")
        assert stored is not None
        assert len(stored) == 1
        assert stored[0]["type"] == "message"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_does_not_persist_when_disabled(self):
        """emit() should not persist when enable_event_log is False."""
        storage = MemoryStorage()
        agent = Agent("test", storage=storage)
        await agent.start()

        await agent.emit("message", {"content": "Hello"})

        stored = await storage.get("test:event_log")
        assert stored is None

        await agent.stop()

    @pytest.mark.asyncio
    async def test_event_log_loaded_on_start(self):
        """Event log should be loaded from storage on start."""
        storage = MemoryStorage()
        await storage.set("test:event_log", [
            {"id": 1, "type": "message", "data": {"content": "Hello"}},
            {"id": 2, "type": "message", "data": {"content": "World"}},
        ])

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        assert len(agent._event_log) == 2
        assert agent._event_counter == 2

        await agent.stop()

    @pytest.mark.asyncio
    async def test_event_counter_restored_from_log(self):
        """Event counter should be restored to max ID from log."""
        storage = MemoryStorage()
        await storage.set("test:event_log", [
            {"id": 5, "type": "message", "data": {}},
            {"id": 10, "type": "message", "data": {}},
            {"id": 7, "type": "message", "data": {}},
        ])

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        assert agent._event_counter == 10

        # Next event should be 11
        event_id = await agent.emit("new", {})
        assert event_id == 11

        await agent.stop()

    @pytest.mark.asyncio
    async def test_clear_event_log(self):
        """clear_event_log should reset log and counter."""
        storage = MemoryStorage()

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        await agent.emit("message", {"content": "Hello"})
        await agent.emit("message", {"content": "World"})

        assert len(agent._event_log) == 2

        await agent.clear_event_log()

        assert len(agent._event_log) == 0
        assert agent._event_counter == 0

        stored = await storage.get("test:event_log")
        assert stored is None

        await agent.stop()


class TestEmitMethod:
    """Tests for the emit() method."""

    @pytest.mark.asyncio
    async def test_emit_returns_event_id(self):
        """emit() should return the event ID."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        event_id = await agent.emit("message", {"content": "Hello"})

        assert event_id == 1
        assert len(agent._event_log) == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_calls_handler(self):
        """emit() should call _emit_handler if set."""
        received = []

        async def handler(event, data, event_id):
            received.append((event, data, event_id))

        agent = Agent("test", storage=MemoryStorage())
        agent._emit_handler = handler
        await agent.start()

        await agent.emit("message", {"content": "Hello"})

        assert len(received) == 1
        assert received[0] == ("message", {"content": "Hello"}, 1)

        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_works_without_handler(self):
        """emit() should work even without _emit_handler."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        # Should not raise
        event_id = await agent.emit("status", {"text": "Hello"})
        assert event_id == 1

        await agent.stop()


class TestReplayEvents:
    """Tests for replay_events async iterator."""

    @pytest.mark.asyncio
    async def test_replay_events_yields_events(self):
        """replay_events() should yield Event objects."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("start", {})
        await agent.emit("chunk", {"text": "Hello"})
        await agent.emit("end", {"text": "Hello"})

        events = []
        async for event in agent.replay_events(after_id=0):
            events.append(event)

        assert len(events) == 3
        assert all(isinstance(e, Event) for e in events)
        assert events[0].type == "start"
        assert events[1].type == "chunk"
        assert events[2].type == "end"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_events_filters_by_id(self):
        """replay_events() should filter by after_id."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("a", {})
        await agent.emit("b", {})
        await agent.emit("c", {})

        events = []
        async for event in agent.replay_events(after_id=1):
            events.append(event)

        assert len(events) == 2
        assert events[0].id == 2
        assert events[1].id == 3

        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_events_empty_log_yields_nothing(self):
        """replay_events with empty log should yield nothing."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        events = []
        async for event in agent.replay_events(after_id=0):
            events.append(event)

        assert events == []
        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_events_data_structure(self):
        """Event objects should have correct structure."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("test", {"key": "value", "num": 42})

        async for event in agent.replay_events():
            assert event.id == 1
            assert event.type == "test"
            assert event.data == {"key": "value", "num": 42}

        await agent.stop()


class TestSSEConnections:
    """Tests for SSE connection tracking."""

    @pytest.mark.asyncio
    async def test_sse_connections_set_exists(self):
        """Agent should have _sse_connections set."""
        agent = Agent("test", storage=MemoryStorage())
        assert hasattr(agent, "_sse_connections")
        assert isinstance(agent._sse_connections, set)


class TestEventReplayWithStorage:
    """Tests for event replay with persistent storage."""

    @pytest.mark.asyncio
    async def test_events_persist_across_instances(self):
        """Events should persist and be replayable after restart."""
        storage = MemoryStorage()

        # First agent emits events
        class DurableAgent(Agent):
            enable_event_log = True

        agent1 = DurableAgent("test", storage=storage)
        await agent1.start()

        await agent1.emit("message", {"content": "Hello"})
        await agent1.emit("status", {"status": "processing"})
        await agent1.emit("message", {"content": "World"})

        await agent1.stop()

        # Second agent loads from storage
        agent2 = DurableAgent("test", storage=storage)
        await agent2.start()

        events = []
        async for event in agent2.replay_events(after_id=0):
            events.append(event)

        assert len(events) == 3
        assert events[0].data["content"] == "Hello"
        assert events[2].data["content"] == "World"

        await agent2.stop()

    @pytest.mark.asyncio
    async def test_event_log_continues_after_restart(self):
        """Event counter should continue from persisted log."""
        storage = MemoryStorage()

        class DurableAgent(Agent):
            enable_event_log = True

        agent1 = DurableAgent("test", storage=storage)
        await agent1.start()
        await agent1.emit("a", {})
        await agent1.emit("b", {})
        await agent1.stop()

        agent2 = DurableAgent("test", storage=storage)
        await agent2.start()
        event_id = await agent2.emit("c", {})

        assert event_id == 3
        await agent2.stop()


class TestConnectionRecovery:
    """Tests for connection recovery scenarios."""

    @pytest.mark.asyncio
    async def test_replay_for_reconnection(self):
        """replay_events enables client reconnection recovery."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("status", {"status": "started"})
        await agent.emit("message", {"content": "Hello"})
        await agent.emit("message", {"content": "World"})

        # Simulate client reconnecting after receiving event 1
        events = []
        async for event in agent.replay_events(after_id=1):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "message"
        assert events[0].data["content"] == "Hello"
        assert events[1].type == "message"
        assert events[1].data["content"] == "World"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_all_events(self):
        """replay_events(after_id=0) should return all events."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("ai_start", {})
        await agent.emit("ai_chunk", {"content": "Hello "})
        await agent.emit("ai_chunk", {"content": "World"})
        await agent.emit("ai_end", {"content": "Hello World"})

        events = []
        async for event in agent.replay_events(after_id=0):
            events.append(event)

        assert len(events) == 4
        assert events[0].type == "ai_start"
        assert events[-1].type == "ai_end"

        await agent.stop()
