"""Tests for the Agent class."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ai_query.agents import (
    Agent,
    MemoryStorage,
    SQLiteStorage,
    Connection,
    ConnectionContext,
)
from ai_query.types import Message, tool
from ai_query.model import LanguageModel


class TestAgentInitialization:
    """Tests for Agent initialization and configuration."""

    def test_basic_initialization(self):
        """Agent should initialize with minimal required parameters."""
        agent = Agent("test-agent")
        assert agent.id == "test-agent"
        assert isinstance(agent.storage, MemoryStorage)

    def test_initialization_with_custom_storage(self, tmp_path):
        """Agent should accept custom storage backend."""
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        agent = Agent("test-agent", storage=storage)
        assert agent.storage is storage
        storage.close()

    def test_initialization_with_initial_state(self):
        """Agent should accept initial_state parameter."""
        agent = Agent("test-agent", initial_state={"counter": 0, "name": "test"})
        assert agent._initial_state == {"counter": 0, "name": "test"}

    def test_initialization_with_system_prompt(self):
        """Agent should accept system parameter."""
        agent = Agent("test-agent", system="You are a helpful bot.")
        assert agent.system == "You are a helpful bot."

    def test_initialization_with_tools(self):
        """Agent should accept tools parameter."""

        def my_tool():
            return "result"

        agent = Agent("test-agent", tools={"my_tool": tool(my_tool)})
        assert "my_tool" in agent.tools


class TestAgentState:
    """Tests for Agent state management."""

    @pytest.fixture
    def storage(self):
        """Create a fresh MemoryStorage instance."""
        return MemoryStorage()

    @pytest.mark.asyncio
    async def test_initial_state_used_when_no_stored_state(self, storage):
        """Agent should use initial_state when storage is empty."""
        agent = Agent("test-agent", storage=storage, initial_state={"counter": 0})
        await agent.start()
        assert agent.state == {"counter": 0}
        await agent.stop()

    @pytest.mark.asyncio
    async def test_stored_state_takes_precedence(self, storage):
        """Agent should load stored state instead of initial_state."""
        await storage.set("test-agent:state", {"counter": 100})

        agent = Agent("test-agent", storage=storage, initial_state={"counter": 0})
        await agent.start()
        assert agent.state == {"counter": 100}
        await agent.stop()

    @pytest.mark.asyncio
    async def test_set_state_persists(self, storage):
        """set_state() should persist to storage."""
        agent = Agent("test-agent", storage=storage)
        await agent.start()
        await agent.set_state({"new": "state"})

        stored = await storage.get("test-agent:state")
        assert stored == {"new": "state"}
        await agent.stop()

    @pytest.mark.asyncio
    async def test_update_state_merges(self, storage):
        """update_state() should merge with existing state."""
        agent = Agent("test-agent", storage=storage, initial_state={"a": 1, "b": 2})
        await agent.start()
        await agent.update_state(b=20, c=3)

        assert agent.state == {"a": 1, "b": 20, "c": 3}
        await agent.stop()

    @pytest.mark.asyncio
    async def test_state_persists_across_instances(self, storage):
        """State should persist across agent instances."""
        agent1 = Agent("test-agent", storage=storage)
        await agent1.start()
        await agent1.set_state({"persisted": True})
        await agent1.stop()

        agent2 = Agent("test-agent", storage=storage)
        await agent2.start()
        assert agent2.state == {"persisted": True}
        await agent2.stop()

    def test_state_not_accessible_before_start(self):
        """Accessing state before start should raise RuntimeError."""
        agent = Agent("test-agent")
        with pytest.raises(RuntimeError, match="Agent not started"):
            _ = agent.state


class TestAgentMessages:
    """Tests for Agent message history."""

    @pytest.mark.asyncio
    async def test_messages_empty_initially(self):
        """Messages should be empty after start."""
        agent = Agent("test-agent", storage=MemoryStorage())
        await agent.start()
        assert agent.messages == []
        await agent.stop()

    @pytest.mark.asyncio
    async def test_messages_persist_across_instances(self):
        """Messages should persist in storage."""
        storage = MemoryStorage()

        agent1 = Agent("test-agent", storage=storage)
        await agent1.start()
        agent1._messages.append(Message(role="user", content="Hello"))
        agent1._messages.append(Message(role="assistant", content="Hi there!"))
        await storage.set(
            "test-agent:messages", [m.to_dict() for m in agent1._messages]
        )
        await agent1.stop()

        agent2 = Agent("test-agent", storage=storage)
        await agent2.start()
        assert len(agent2.messages) == 2
        assert agent2.messages[0].content == "Hello"
        await agent2.stop()

    @pytest.mark.asyncio
    async def test_clear_removes_messages(self):
        """clear() should remove all messages."""
        storage = MemoryStorage()
        agent = Agent("test-agent", storage=storage)
        await agent.start()
        agent._messages.append(Message(role="user", content="Hello"))
        await agent.clear()
        assert agent.messages == []
        await agent.stop()


class TestAgentLifecycle:
    """Tests for Agent lifecycle management."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Agent should work as async context manager."""
        async with Agent("test-agent", storage=MemoryStorage()) as agent:
            assert agent._running is True

    @pytest.mark.asyncio
    async def test_start_calls_on_start(self):
        """start() should call on_start hook."""
        on_start_called = []

        class MyAgent(Agent):
            async def on_start(self):
                on_start_called.append(True)

        agent = MyAgent("test", storage=MemoryStorage())
        await agent.start()
        assert on_start_called == [True]
        await agent.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_processor(self):
        """stop() should cancel the mailbox processor."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()
        assert agent._processor_task is not None

        await agent.stop()
        assert agent._processor_task is None


class TestAgentConnections:
    """Tests for Agent connection management."""

    @pytest.mark.asyncio
    async def test_on_connect_adds_connection(self):
        """on_connect should add connection to set."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        mock_conn = MagicMock(spec=Connection)
        ctx = ConnectionContext(request=None, metadata={})
        await agent.on_connect(mock_conn, ctx)

        assert mock_conn in agent._connections
        await agent.stop()

    @pytest.mark.asyncio
    async def test_on_close_removes_connection(self):
        """on_close should remove connection from set."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        mock_conn = MagicMock(spec=Connection)
        agent._connections.add(mock_conn)
        await agent.on_close(mock_conn, 1000, "Normal")

        assert mock_conn not in agent._connections
        await agent.stop()


class TestAgentChat:
    """Tests for Agent chat functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LanguageModel."""
        model = MagicMock(spec=LanguageModel)
        model.provider = MagicMock()
        model.model_id = "test-model"
        return model

    @pytest.fixture
    def mock_stream_text(self):
        """Create a mock stream_text function."""

        async def mock_text_stream():
            yield "Hello "
            yield "World!"

        mock_result = MagicMock()
        mock_result.text_stream = mock_text_stream()

        with patch("ai_query.stream_text", return_value=mock_result) as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_chat_adds_user_message(self, mock_stream_text, mock_model):
        """chat() should add user message to history."""
        agent = Agent("test", storage=MemoryStorage(), model=mock_model)
        await agent.start()

        await agent.chat("Hello")

        assert len(agent.messages) >= 1
        assert agent.messages[0].role == "user"
        assert agent.messages[0].content == "Hello"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_chat_adds_assistant_response(self, mock_stream_text, mock_model):
        """chat() should add assistant response to history."""
        agent = Agent("test", storage=MemoryStorage(), model=mock_model)
        await agent.start()

        await agent.chat("Hello")

        assert len(agent.messages) == 2
        assert agent.messages[1].role == "assistant"
        assert agent.messages[1].content == "Hello World!"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_chat_persists_messages(self, mock_stream_text, mock_model):
        """chat() should persist messages to storage."""
        storage = MemoryStorage()
        agent = Agent("test", storage=storage, model=mock_model)
        await agent.start()

        await agent.chat("Hello")

        stored = await storage.get("test:messages")
        assert stored is not None
        assert len(stored) == 2
        await agent.stop()


class TestAgentStream:
    """Tests for Agent streaming functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LanguageModel."""
        model = MagicMock(spec=LanguageModel)
        model.provider = MagicMock()
        model.model_id = "test-model"
        return model

    @pytest.fixture
    def mock_stream_text(self):
        """Create a mock stream_text function."""

        async def mock_text_stream():
            yield "Hello "
            yield "World!"

        mock_result = MagicMock()
        mock_result.text_stream = mock_text_stream()

        with patch("ai_query.stream_text", return_value=mock_result) as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, mock_stream_text, mock_model):
        """stream() should yield response chunks."""
        agent = Agent("test", storage=MemoryStorage(), model=mock_model)
        await agent.start()

        chunks = []
        async for chunk in agent.stream("Hello"):
            chunks.append(chunk)

        assert chunks == ["Hello ", "World!"]
        await agent.stop()

    @pytest.mark.asyncio
    async def test_stream_adds_messages(self, mock_stream_text, mock_model):
        """stream() should add messages to history."""
        agent = Agent("test", storage=MemoryStorage(), model=mock_model)
        await agent.start()

        async for _ in agent.stream("Hello"):
            pass

        assert len(agent.messages) == 2
        await agent.stop()


class TestAgentSubclassing:
    """Tests for Agent subclassing patterns."""

    @pytest.mark.asyncio
    async def test_subclass_with_defaults(self):
        """Subclass can set default model and system prompt."""

        class MyBot(Agent):
            def __init__(self, agent_id: str):
                super().__init__(
                    agent_id, system="I am a custom bot.", storage=MemoryStorage()
                )

        agent = MyBot("test")
        assert agent.system == "I am a custom bot."

    @pytest.mark.asyncio
    async def test_subclass_with_custom_hooks(self):
        """Subclass can override lifecycle hooks."""
        events = []

        class MyBot(Agent):
            async def on_start(self):
                events.append("start")

            async def on_stop(self):
                events.append("stop")

        agent = MyBot("test", storage=MemoryStorage())
        await agent.start()
        await agent.stop()

        assert events == ["start", "stop"]

    @pytest.mark.asyncio
    async def test_subclass_with_tools(self):
        """Subclass can define tools."""

        def search(query: str) -> str:
            return f"Results for: {query}"

        class SearchBot(Agent):
            def __init__(self, agent_id: str):
                super().__init__(
                    agent_id, tools={"search": tool(search)}, storage=MemoryStorage()
                )

        agent = SearchBot("test")
        assert "search" in agent.tools


class TestAgentMailbox:
    """Tests for Agent actor mailbox."""

    @pytest.mark.asyncio
    async def test_enqueue_connect_calls_on_connect(self):
        """Enqueued connect should call on_connect handler."""
        connected = []

        class MyAgent(Agent):
            async def on_connect(self, connection, ctx):
                await super().on_connect(connection, ctx)
                connected.append(connection)

        mock_conn = MagicMock(spec=Connection)
        ctx = ConnectionContext(request=None, metadata={})

        async with MyAgent("test", storage=MemoryStorage()) as agent:
            agent.enqueue("connect", None, connection=mock_conn, ctx=ctx)
            await agent._mailbox.join()

        assert mock_conn in connected

    @pytest.mark.asyncio
    async def test_enqueue_message_calls_on_message(self):
        """Enqueued message should call on_message handler."""
        messages = []

        class MyAgent(Agent):
            async def on_message(self, connection, message):
                messages.append(message)

        mock_conn = MagicMock(spec=Connection)

        async with MyAgent("test", storage=MemoryStorage()) as agent:
            agent.enqueue("message", "Hello!", connection=mock_conn)
            await agent._mailbox.join()

        assert "Hello!" in messages

    @pytest.mark.asyncio
    async def test_enqueue_close_calls_on_close(self):
        """Enqueued close should call on_close handler."""
        closed = []

        class MyAgent(Agent):
            async def on_close(self, connection, code, reason):
                await super().on_close(connection, code, reason)
                closed.append((code, reason))

        mock_conn = MagicMock(spec=Connection)
        mock_conn.send = AsyncMock()
        mock_conn.close = AsyncMock()

        async with MyAgent("test", storage=MemoryStorage()) as agent:
            agent._connections.add(mock_conn)
            agent.enqueue("close", (1000, "Normal"), connection=mock_conn)
            await agent._mailbox.join()

        assert (1000, "Normal") in closed


class TestAgentEmit:
    """Tests for Agent event emission."""

    @pytest.mark.asyncio
    async def test_emit_returns_event_id(self):
        """emit() should return incrementing event IDs."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        id1 = await agent.emit("status", {"text": "Hello"})
        id2 = await agent.emit("status", {"text": "World"})

        assert id1 == 1
        assert id2 == 2
        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_logs_to_event_log(self):
        """emit() should add events to _event_log."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("status", {"text": "Hello"})
        await agent.emit("chunk", {"content": "World"})

        assert len(agent._event_log) == 2
        assert agent._event_log[0]["type"] == "status"
        assert agent._event_log[1]["type"] == "chunk"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_persists_when_enabled(self):
        """emit() should persist to storage when enable_event_log=True."""
        storage = MemoryStorage()

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        await agent.emit("status", {"text": "Hello"})

        stored = await storage.get("test:event_log")
        assert stored is not None
        assert len(stored) == 1
        await agent.stop()

    @pytest.mark.asyncio
    async def test_emit_does_not_persist_when_disabled(self):
        """emit() should not persist when enable_event_log=False."""
        storage = MemoryStorage()
        agent = Agent("test", storage=storage)
        await agent.start()

        await agent.emit("status", {"text": "Hello"})

        stored = await storage.get("test:event_log")
        assert stored is None
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

        await agent.emit("status", {"text": "Hello"})

        assert len(received) == 1
        assert received[0] == ("status", {"text": "Hello"}, 1)
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


class TestAgentReplayEvents:
    """Tests for event replay."""

    @pytest.mark.asyncio
    async def test_replay_events_yields_events(self):
        """replay_events() should yield Event objects."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("a", {"n": 1})
        await agent.emit("b", {"n": 2})
        await agent.emit("c", {"n": 3})

        events = []
        async for event in agent.replay_events(after_id=0):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == "a"
        assert events[1].type == "b"
        assert events[2].type == "c"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_events_filters_by_id(self):
        """replay_events() should only yield events after the given ID."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        await agent.emit("a", {"n": 1})
        await agent.emit("b", {"n": 2})
        await agent.emit("c", {"n": 3})

        events = []
        async for event in agent.replay_events(after_id=1):
            events.append(event)

        assert len(events) == 2
        assert events[0].id == 2
        assert events[1].id == 3
        await agent.stop()

    @pytest.mark.asyncio
    async def test_replay_events_empty_when_no_events(self):
        """replay_events() should yield nothing when log is empty."""
        agent = Agent("test", storage=MemoryStorage())
        await agent.start()

        events = []
        async for event in agent.replay_events(after_id=0):
            events.append(event)

        assert events == []
        await agent.stop()


class TestAgentEventLogPersistence:
    """Tests for event log persistence."""

    @pytest.mark.asyncio
    async def test_event_log_loaded_on_start(self):
        """Event log should be loaded from storage on start."""
        storage = MemoryStorage()
        await storage.set(
            "test:event_log",
            [
                {"id": 1, "type": "a", "data": {}},
                {"id": 2, "type": "b", "data": {}},
            ],
        )

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        assert len(agent._event_log) == 2
        assert agent._event_counter == 2
        await agent.stop()

    @pytest.mark.asyncio
    async def test_event_counter_continues_from_log(self):
        """Event counter should continue from max ID in log."""
        storage = MemoryStorage()
        await storage.set(
            "test:event_log",
            [
                {"id": 5, "type": "a", "data": {}},
                {"id": 10, "type": "b", "data": {}},
            ],
        )

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        event_id = await agent.emit("c", {})
        assert event_id == 11
        await agent.stop()

    @pytest.mark.asyncio
    async def test_clear_event_log(self):
        """clear_event_log() should reset log and counter."""
        storage = MemoryStorage()

        class DurableAgent(Agent):
            enable_event_log = True

        agent = DurableAgent("test", storage=storage)
        await agent.start()

        await agent.emit("a", {})
        await agent.emit("b", {})
        await agent.clear_event_log()

        assert len(agent._event_log) == 0
        assert agent._event_counter == 0
        await agent.stop()
