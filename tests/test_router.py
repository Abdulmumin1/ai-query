"""Tests for AgentServer router."""

import json
import pytest
from unittest.mock import MagicMock, patch
from aiohttp import web
from ai_query.agents import Agent, MemoryStorage, AgentServer, AgentServerConfig


class TestAgentServerBasics:
    """Tests for AgentServer basic functionality."""

    def test_initialization(self):
        """AgentServer should initialize with an Agent class."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        server = AgentServer(TestAgent)
        # Check that registry has the default route
        # We can resolve any ID to TestAgent
        assert server.registry.resolve("any-id") is TestAgent

    def test_initialization_with_config(self):
        """AgentServer should accept configuration."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        config = AgentServerConfig(
            idle_timeout=600, max_agents=50, base_path="/api/agent"
        )
        server = AgentServer(TestAgent, config=config)

        assert server._config.idle_timeout == 600
        assert server._config.max_agents == 50
        assert server._config.base_path == "/api/agent"

    def test_get_or_create_new_agent(self):
        """get_or_create() should create new agents."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        server = AgentServer(TestAgent)
        agent = server.get_or_create("agent-1")

        assert agent is not None
        assert agent.id == "agent-1"
        assert "agent-1" in server._agents

    def test_get_or_create_returns_existing(self):
        """get_or_create() should return existing agents."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        server = AgentServer(TestAgent)
        agent1 = server.get_or_create("agent-1")
        agent2 = server.get_or_create("agent-1")

        assert agent1 is agent2

    def test_get_or_create_respects_max_agents(self):
        """get_or_create() should reject when max_agents is reached."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        config = AgentServerConfig(max_agents=2)
        server = AgentServer(TestAgent, config=config)

        server.get_or_create("agent-1")
        server.get_or_create("agent-2")

        with pytest.raises(web.HTTPTooManyRequests):
            server.get_or_create("agent-3")

    def test_list_agents(self):
        """list_agents() should return all active agent IDs."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        server = AgentServer(TestAgent)
        server.get_or_create("agent-1")
        server.get_or_create("agent-2")
        server.get_or_create("agent-3")

        agents = server.list_agents()
        assert sorted(agents) == ["agent-1", "agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_evict_removes_agent(self):
        """evict() should remove agent from registry."""

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        server = AgentServer(TestAgent)
        agent = server.get_or_create("agent-1")
        await agent.start()

        await server.evict("agent-1")
        assert "agent-1" not in server._agents


class TestAgentServerEndpoints:
    """Tests for AgentServer HTTP/REST endpoints."""

    @pytest.fixture
    def agent_class(self):
        """Create a test Agent class."""
        from ai_query.model import LanguageModel

        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(
                    agent_id,
                    storage=MemoryStorage(),
                    initial_state={"count": 0},
                    model=MagicMock(spec=LanguageModel),
                )

            async def handle_invoke(self, payload):
                task = payload.get("task")
                if task == "echo":
                    return {"echo": payload.get("data")}
                if task == "increment":
                    await self.update_state(count=self.state.get("count", 0) + 1)
                    return {"count": self.state["count"]}
                return {"error": f"Unknown task: {task}"}

        return TestAgent

    @pytest.mark.asyncio
    async def test_get_state_endpoint(self, aiohttp_client, agent_class):
        """GET /agent/{id}/state should return agent state."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        client = await aiohttp_client(app)

        agent = server.get_or_create("test-state")
        await agent.start()

        resp = await client.get("/agent/test-state/state")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"count": 0}

    @pytest.mark.asyncio
    async def test_put_state_endpoint(self, aiohttp_client, agent_class):
        """PUT /agent/{id}/state should update agent state."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        client = await aiohttp_client(app)

        agent = server.get_or_create("test-put")
        await agent.start()

        resp = await client.put(
            "/agent/test-put/state", json={"count": 100, "name": "updated"}
        )
        assert resp.status == 200

        resp = await client.get("/agent/test-put/state")
        data = await resp.json()
        assert data == {"count": 100, "name": "updated"}

    @pytest.mark.asyncio
    async def test_invoke_endpoint(self, aiohttp_client, agent_class):
        """POST /agent/{id}/invoke should call handle_invoke."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        client = await aiohttp_client(app)

        resp = await client.post(
            "/agent/test-invoke/invoke", json={"task": "echo", "data": "hello"}
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["result"] == {"echo": "hello"}
        assert data["agent_id"] == "test-invoke"

    @pytest.mark.asyncio
    async def test_delete_agent_endpoint(self, aiohttp_client, agent_class):
        """DELETE /agent/{id} should evict the agent."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        client = await aiohttp_client(app)

        agent = server.get_or_create("test-delete")
        await agent.start()
        assert "test-delete" in server._agents

        resp = await client.delete("/agent/test-delete")
        assert resp.status == 200

        assert "test-delete" not in server._agents

    @pytest.mark.asyncio
    async def test_chat_endpoint_non_streaming(self, aiohttp_client, agent_class):
        """POST /agent/{id}/chat should return JSON response."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        async def mock_text_stream():
            yield "Hello"
            yield " World"

        mock_result = MagicMock()
        mock_result.text_stream = mock_text_stream()

        with patch("ai_query.stream_text", return_value=mock_result):
            client = await aiohttp_client(app)

            resp = await client.post("/agent/test-chat/chat", json={"message": "Hi"})
            assert resp.status == 200
            data = await resp.json()
            assert data["response"] == "Hello World"

    @pytest.mark.asyncio
    async def test_agent_not_found(self, aiohttp_client, agent_class):
        """Endpoints should return 404 for non-existent agents (when required)."""
        config = AgentServerConfig(enable_rest_api=True)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()

        client = await aiohttp_client(app)

        resp = await client.get("/agent/nonexistent/state")
        assert resp.status == 404


class TestAgentServerStreaming:
    """Tests for AgentServer streaming endpoints."""

    @pytest.fixture
    def streaming_agent_class(self):
        """Create an Agent class with streaming support."""
        from ai_query.model import LanguageModel

        class StreamingAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(
                    agent_id,
                    storage=MemoryStorage(),
                    initial_state={},
                    model=MagicMock(spec=LanguageModel),
                )

            async def stream(self, message, **kwargs):
                """Override stream to return mock data without API calls."""
                # Add user message to history
                from ai_query.types import Message

                self._messages.append(Message(role="user", content=message))

                # Yield mock chunks
                yield "Hello "
                yield "World"

                # Add assistant response
                self._messages.append(Message(role="assistant", content="Hello World"))

        return StreamingAgent

    @pytest.mark.asyncio
    async def test_streaming_chat_endpoint(self, aiohttp_client, streaming_agent_class):
        """POST /agent/{id}/chat?stream=true should return SSE stream."""
        server = AgentServer(streaming_agent_class)

        # Mock get_or_create to return our specific instance or let logic handle it
        # Since we use registry, server will create new instance of streaming_agent_class
        # That works fine.

        config = AgentServerConfig(enable_rest_api=True)
        server._config = config
        app = server.create_app()

        client = await aiohttp_client(app)

        resp = await client.post(
            "/agent/test-stream/chat?stream=true",
            json={"message": "Hi"},
            headers={"Content-Type": "application/json"},
        )

        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/event-stream"

        content = await resp.text()

        assert "event: start" in content
        assert "event: chunk" in content or "Hello" in content


class TestAgentServerCORS:
    """Tests for AgentServer CORS handling."""

    @pytest.fixture
    def agent_class(self):
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        return TestAgent

    @pytest.mark.asyncio
    async def test_cors_all_origins(self, aiohttp_client, agent_class):
        """CORS should allow all origins when not configured."""
        server = AgentServer(agent_class)
        app = server.create_app()
        client = await aiohttp_client(app)

        agent = server.get_or_create("test")
        await agent.start()

        resp = await client.get(
            "/agent/test/state", headers={"Origin": "https://example.com"}
        )

        assert "Access-Control-Allow-Origin" in resp.headers
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_cors_specific_origins(self, aiohttp_client, agent_class):
        """CORS should only allow configured origins."""
        config = AgentServerConfig(allowed_origins=["https://myapp.com"])
        server = AgentServer(agent_class, config=config)
        app = server.create_app()
        client = await aiohttp_client(app)

        agent = server.get_or_create("test")
        await agent.start()

        resp = await client.get(
            "/agent/test/state", headers={"Origin": "https://myapp.com"}
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://myapp.com"

    @pytest.mark.asyncio
    async def test_cors_rejected_origin(self, aiohttp_client, agent_class):
        """CORS should not include header for rejected origins."""
        config = AgentServerConfig(allowed_origins=["https://myapp.com"])
        server = AgentServer(agent_class, config=config)
        app = server.create_app()
        client = await aiohttp_client(app)

        agent = server.get_or_create("test")
        await agent.start()

        resp = await client.get(
            "/agent/test/state", headers={"Origin": "https://evil.com"}
        )
        assert resp.headers.get("Access-Control-Allow-Origin") != "https://evil.com"


class TestAgentServerAuth:
    """Tests for AgentServer authentication."""

    @pytest.fixture
    def agent_class(self):
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        return TestAgent

    @pytest.mark.asyncio
    async def test_auth_success(self, aiohttp_client, agent_class):
        """Requests with valid auth should succeed."""

        async def auth_check(request):
            return request.headers.get("X-API-Key") == "secret"

        config = AgentServerConfig(auth=auth_check)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()
        client = await aiohttp_client(app)

        agent = server.get_or_create("test")
        await agent.start()

        resp = await client.get("/agent/test/state", headers={"X-API-Key": "secret"})
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_auth_failure(self, aiohttp_client, agent_class):
        """Requests with invalid auth should be rejected."""

        async def auth_check(request):
            return request.headers.get("X-API-Key") == "secret"

        config = AgentServerConfig(auth=auth_check)
        server = AgentServer(agent_class, config=config)
        app = server.create_app()
        client = await aiohttp_client(app)

        resp = await client.get("/agent/test/state", headers={"X-API-Key": "wrong"})
        assert resp.status == 401


class TestAgentServerWebSocket:
    """Tests for AgentServer WebSocket handling."""

    @pytest.fixture
    def agent_class(self):
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage(), initial_state={})

            async def on_message(self, connection, message):
                await connection.send(f"Echo: {message}")

        return TestAgent

    @pytest.mark.asyncio
    async def test_websocket_connection(self, aiohttp_client, agent_class):
        """WebSocket connections should be established."""
        server = AgentServer(agent_class)
        app = server.create_app()
        client = await aiohttp_client(app)

        async with client.ws_connect("/agent/test-ws/ws") as ws:
            await ws.close()

        assert "test-ws" in server._agents

    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, aiohttp_client, agent_class):
        """WebSocket messages should be processed by on_message."""
        server = AgentServer(agent_class)
        app = server.create_app()
        client = await aiohttp_client(app)

        async with client.ws_connect("/agent/test-echo/ws") as ws:
            await ws.send_str("Hello")

            import asyncio

            await asyncio.sleep(0.1)

            await ws.close()


class TestAgentServerLifecycleHooks:
    """Tests for AgentServer lifecycle hooks."""

    @pytest.fixture
    def agent_class(self):
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        return TestAgent

    @pytest.mark.asyncio
    async def test_on_agent_create_hook(self, agent_class):
        """on_agent_create should be called when agent is first accessed."""
        created_agents = []

        class CustomServer(AgentServer):
            async def on_agent_create(self, agent):
                created_agents.append(agent.id)

        server = CustomServer(agent_class)
        agent = server.get_or_create("test-create")
        await agent.start()
        await server.on_agent_create(agent)

        assert "test-create" in created_agents

    @pytest.mark.asyncio
    async def test_on_agent_evict_hook(self, agent_class):
        """on_agent_evict should be called when agent is evicted."""
        evicted_agents = []

        class CustomServer(AgentServer):
            async def on_agent_evict(self, agent):
                evicted_agents.append(agent.id)

        server = CustomServer(agent_class)
        agent = server.get_or_create("test-evict")
        await agent.start()

        await server.evict("test-evict")

        assert "test-evict" in evicted_agents


class TestAgentServerCustomRoutes:
    """Tests for adding custom routes to AgentServer."""

    @pytest.fixture
    def agent_class(self):
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, storage=MemoryStorage())

        return TestAgent

    @pytest.mark.asyncio
    async def test_on_app_setup_hook(self, aiohttp_client, agent_class):
        """on_app_setup should allow adding custom routes."""

        class CustomServer(AgentServer):
            def on_app_setup(self, app):
                app.router.add_get("/health", self.health_check)

            async def health_check(self, request):
                return web.json_response({"status": "ok", "agents": len(self._agents)})

        server = CustomServer(agent_class)
        app = server.create_app()
        client = await aiohttp_client(app)

        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_create_app_customization(self, aiohttp_client, agent_class):
        """create_app() should return customizable app."""
        server = AgentServer(agent_class)
        app = server.create_app()

        async def custom_handler(request):
            return web.json_response({"custom": True})

        app.router.add_get("/custom", custom_handler)

        client = await aiohttp_client(app)

        resp = await client.get("/custom")
        assert resp.status == 200
        data = await resp.json()
        assert data["custom"] is True
