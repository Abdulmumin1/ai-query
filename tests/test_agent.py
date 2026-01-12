"""Tests for the Agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_query.agents import (
    Agent,
    ChatAgent,
    InMemoryAgent,
    SQLiteAgent,
    Connection,
    ConnectionContext,
)
from ai_query.types import Message


class TestInMemoryAgent:
    """Tests for InMemoryAgent."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clear storage before each test."""
        InMemoryAgent.clear_all()
        yield
        InMemoryAgent.clear_all()
    
    @pytest.mark.asyncio
    async def test_initial_state(self):
        """Agent should use initial_state when no stored state exists."""
        class MyAgent(InMemoryAgent):
            initial_state = {"counter": 0}
        
        agent = MyAgent("test-1")
        await agent.start()
        
        assert agent.state == {"counter": 0}
    
    @pytest.mark.asyncio
    async def test_set_and_load_state(self):
        """State should persist across agent instances."""
        class MyAgent(InMemoryAgent):
            initial_state = {"counter": 0}
        
        # First instance sets state
        agent1 = MyAgent("test-1")
        await agent1.start()
        await agent1.set_state({"counter": 5})
        
        # Second instance should load persisted state
        agent2 = MyAgent("test-1")
        await agent2.start()
        
        assert agent2.state == {"counter": 5}
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Agent should work as async context manager."""
        class MyAgent(InMemoryAgent):
            initial_state = {"active": True}
        
        async with MyAgent("test-1") as agent:
            assert agent.state == {"active": True}
    
    @pytest.mark.asyncio
    async def test_message_persistence(self):
        """Messages should be saved and loaded."""
        class MyAgent(InMemoryAgent):
            initial_state = {}
        
        async with MyAgent("test-1") as agent:
            await agent.save_messages([
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ])
        
        async with MyAgent("test-1") as agent:
            assert len(agent.messages) == 2
            assert agent.messages[0].content == "Hello"
            assert agent.messages[1].content == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_on_start_hook(self):
        """on_start should be called during start()."""
        started = []
        
        class MyAgent(InMemoryAgent):
            initial_state = {}
            
            async def on_start(self):
                started.append(True)
        
        async with MyAgent("test-1"):
            pass
        
        assert len(started) == 1
    
    @pytest.mark.asyncio
    async def test_on_state_update_hook(self):
        """on_state_update should be called when state changes."""
        updates = []
        
        class MyAgent(InMemoryAgent):
            initial_state = {"v": 0}
            
            def on_state_update(self, state, source):
                updates.append((state, source))
        
        async with MyAgent("test-1") as agent:
            await agent.set_state({"v": 1})
            await agent.set_state({"v": 2})
        
        assert len(updates) == 2
        assert updates[0] == ({"v": 1}, "server")
        assert updates[1] == ({"v": 2}, "server")
    
    @pytest.mark.asyncio
    async def test_agent_not_started_error(self):
        """Accessing state before start() should raise error."""
        agent = InMemoryAgent("test-1")
        
        with pytest.raises(RuntimeError, match="Agent not started"):
            _ = agent.state


class TestSQLiteAgent:
    """Tests for SQLiteAgent."""
    
    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, tmp_path):
        """SQLiteAgent should persist state to database."""
        db_path = str(tmp_path / "test.db")
        
        class MyAgent(SQLiteAgent):
            initial_state = {"count": 0}
        
        # First agent sets state
        async with MyAgent("agent-1", db_path=db_path) as agent:
            await agent.set_state({"count": 10})
        
        # New agent instance should load persisted state
        async with MyAgent("agent-1", db_path=db_path) as agent:
            assert agent.state == {"count": 10}
    
    @pytest.mark.asyncio
    async def test_sql_queries(self, tmp_path):
        """sql() method should execute queries."""
        db_path = str(tmp_path / "test.db")
        
        class MyAgent(SQLiteAgent):
            initial_state = {}
        
        async with MyAgent("agent-1", db_path=db_path) as agent:
            # Create table
            await agent.sql("CREATE TABLE users (id TEXT, name TEXT)")

            # Insert
            await agent.sql("INSERT INTO users VALUES (?, ?)", "1", "Alice")

            # Query
            users = await agent.sql("SELECT * FROM users")

            assert len(users) == 1
            assert users[0]["id"] == "1"
            assert users[0]["name"] == "Alice"


class TestWebSocketHooks:
    """Tests for WebSocket lifecycle hooks."""
    
    @pytest.mark.asyncio
    async def test_on_connect(self):
        """on_connect should add connection to set."""
        class MyAgent(InMemoryAgent):
            initial_state = {}
        
        mock_conn = MagicMock(spec=Connection)
        mock_conn.send = AsyncMock()
        mock_conn.close = AsyncMock()
        
        async with MyAgent("test-1") as agent:
            ctx = ConnectionContext(request=None)
            await agent.on_connect(mock_conn, ctx)
            
            assert mock_conn in agent._connections
    
    @pytest.mark.asyncio
    async def test_on_close(self):
        """on_close should remove connection from set."""
        class MyAgent(InMemoryAgent):
            initial_state = {}
        
        mock_conn = MagicMock(spec=Connection)
        mock_conn.send = AsyncMock()
        mock_conn.close = AsyncMock()
        
        async with MyAgent("test-1") as agent:
            ctx = ConnectionContext(request=None)
            await agent.on_connect(mock_conn, ctx)
            assert mock_conn in agent._connections
            
            await agent.on_close(mock_conn, 1000, "Normal close")
            assert mock_conn not in agent._connections
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """broadcast should send to all connections."""
        class MyAgent(InMemoryAgent):
            initial_state = {}
        
        conn1 = MagicMock(spec=Connection)
        conn1.send = AsyncMock()
        conn1.close = AsyncMock()
        
        conn2 = MagicMock(spec=Connection)
        conn2.send = AsyncMock()
        conn2.close = AsyncMock()
        
        async with MyAgent("test-1") as agent:
            ctx = ConnectionContext(request=None)
            await agent.on_connect(conn1, ctx)
            await agent.on_connect(conn2, ctx)
            
            await agent.broadcast("Hello everyone!")
            
            conn1.send.assert_called_once_with("Hello everyone!")
            conn2.send.assert_called_once_with("Hello everyone!")
    
    @pytest.mark.asyncio
    async def test_state_broadcast(self):
        """set_state should broadcast state to all connections."""
        class MyAgent(InMemoryAgent):
            initial_state = {"v": 0}
        
        conn = MagicMock(spec=Connection)
        conn.send = AsyncMock()
        conn.close = AsyncMock()
        
        async with MyAgent("test-1") as agent:
            ctx = ConnectionContext(request=None)
            await agent.on_connect(conn, ctx)
            
            await agent.set_state({"v": 1})
            
            # Should have broadcast JSON state
            conn.send.assert_called()
            call_arg = conn.send.call_args[0][0]
            assert '"type": "state"' in call_arg
            assert '"data"' in call_arg


class TestChatAgentIntegration:
    """Integration tests for ChatAgent (requires mocking LLM)."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clear storage before each test."""
        InMemoryAgent.clear_all()
        yield
        InMemoryAgent.clear_all()
    
    @pytest.mark.asyncio
    async def test_chat_agent_message_history(self):
        """ChatAgent should manage message history."""
        from unittest.mock import patch, MagicMock, AsyncMock
        
        # Mock generate_text at the module where it's imported
        mock_result = MagicMock()
        mock_result.text = "I'm doing well, thank you!"
        
        with patch("ai_query.generate_text", new=AsyncMock(return_value=mock_result)):
            with patch("ai_query.providers.google.google", return_value=MagicMock()):
                class MyBot(ChatAgent, InMemoryAgent):
                    initial_state = {}
                    system = "You are a helpful bot."
                
                async with MyBot("bot-1") as bot:
                    response = await bot.chat("How are you?")
                    
                    assert response == "I'm doing well, thank you!"
                    assert len(bot.messages) == 2
                    assert bot.messages[0].role == "user"
                    assert bot.messages[0].content == "How are you?"
                    assert bot.messages[1].role == "assistant"
                    assert bot.messages[1].content == "I'm doing well, thank you!"
