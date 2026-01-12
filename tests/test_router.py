"""Tests for AgentServer router."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import web
from ai_query.agents import Agent, ChatAgent, InMemoryAgent
from ai_query.agents.router import AgentServer, AgentServerConfig

@pytest.fixture(autouse=True)
def cleanup():
    """Clear storage before each test."""
    InMemoryAgent.clear_all()
    yield
    InMemoryAgent.clear_all()

@pytest.mark.asyncio
async def test_streaming_chat_endpoint(aiohttp_client):
    """Test POST /agent/{id}/chat?stream=true endpoint."""

    # Mock stream_chat to yield chunks
    async def mock_stream_chat(message):
        yield "Hello"
        yield " "
        yield "World"

    # Create a ChatAgent with mocked stream_chat
    class MyBot(ChatAgent, InMemoryAgent):
        initial_state = {}
        # We need to mock the method on the class or instance

    bot = MyBot("test-bot")
    bot.stream_chat = mock_stream_chat # type: ignore

    # Mock AgentServer.get_or_create to return our bot
    server = AgentServer(MyBot)
    server.get_or_create = MagicMock(return_value=bot) # type: ignore

    # Manually create app and router to avoid blocking serve_async
    config = AgentServerConfig()
    app = web.Application()
    base = config.base_path.rstrip("/")

    # Bind handlers
    app.router.add_post(f"{base}/{{agent_id}}/chat", server._handle_chat)

    client = await aiohttp_client(app)

    # Make request
    resp = await client.post(
        f"/agent/test-bot/chat?stream=true",
        json={"message": "Hi"},
        headers={"Content-Type": "application/json"}
    )

    assert resp.status == 200
    assert resp.headers["Content-Type"] == "text/event-stream"
    assert resp.headers["Cache-Control"] == "no-cache"

    # Read stream
    content = await resp.text()

    expected_chunks = [
        "event: start\ndata: \n\n",
        "event: chunk\ndata: Hello\n\n",
        "event: chunk\ndata:  \n\n",
        "event: chunk\ndata: World\n\n",
        'event: end\ndata: "Hello World"\n\n'
    ]

    for chunk in expected_chunks:
        assert chunk in content

@pytest.mark.asyncio
async def test_chat_endpoint_no_stream(aiohttp_client):
    """Test POST /agent/{id}/chat without streaming."""

    # Mock handle_request
    mock_result = {"agent_id": "test-bot", "response": "Hello World"}

    class MyBot(ChatAgent, InMemoryAgent):
        initial_state = {}

    bot = MyBot("test-bot")
    bot.handle_request = AsyncMock(return_value=mock_result) # type: ignore

    server = AgentServer(MyBot)
    server.get_or_create = MagicMock(return_value=bot) # type: ignore

    config = AgentServerConfig()
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_post(f"{base}/{{agent_id}}/chat", server._handle_chat)

    client = await aiohttp_client(app)

    resp = await client.post(
        f"/agent/test-bot/chat",
        json={"message": "Hi"},
    )

    assert resp.status == 200
    assert "application/json" in resp.headers["Content-Type"]
    data = await resp.json()
    assert data == mock_result

@pytest.mark.asyncio
async def test_get_state_endpoint(aiohttp_client):
    """Test GET /agent/{id}/state endpoint."""
    class MyBot(InMemoryAgent):
        initial_state = {"count": 42}

    server = AgentServer(MyBot)
    config = AgentServerConfig(enable_rest_api=True)
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_get(f"{base}/{{agent_id}}/state", server._handle_get_state)

    client = await aiohttp_client(app)

    # Initialize agent state first
    agent = server.get_or_create("test-state")
    await agent.start()

    resp = await client.get(f"/agent/test-state/state")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"count": 42}

@pytest.mark.asyncio
async def test_put_state_endpoint(aiohttp_client):
    """Test PUT /agent/{id}/state endpoint."""
    class MyBot(InMemoryAgent):
        initial_state = {"count": 0}

    server = AgentServer(MyBot)
    config = AgentServerConfig(enable_rest_api=True)
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_put(f"{base}/{{agent_id}}/state", server._handle_put_state)

    client = await aiohttp_client(app)

    # Create agent first
    server.get_or_create("test-state-put")

    # Put new state
    new_state = {"count": 100}
    resp = await client.put(
        f"/agent/test-state-put/state",
        json=new_state
    )
    assert resp.status == 200

    # Verify state was updated
    agent = server.get_or_create("test-state-put")
    assert agent.state == new_state

@pytest.mark.asyncio
async def test_invoke_endpoint(aiohttp_client):
    """Test POST /agent/{id}/invoke endpoint."""
    class MyBot(InMemoryAgent):
        async def handle_invoke(self, payload):
            if payload.get("task") == "echo":
                return {"echo": payload.get("data")}
            return {"error": "unknown"}

    server = AgentServer(MyBot)
    config = AgentServerConfig()
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_post(f"{base}/{{agent_id}}/invoke", server._handle_invoke)

    client = await aiohttp_client(app)

    payload = {"task": "echo", "data": "hello"}
    resp = await client.post(
        f"/agent/test-invoke/invoke",
        json=payload
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["result"] == {"echo": "hello"}
    assert data["agent_id"] == "test-invoke"

@pytest.mark.asyncio
async def test_delete_agent_endpoint(aiohttp_client):
    """Test DELETE /agent/{id} endpoint."""
    class MyBot(InMemoryAgent):
        pass

    server = AgentServer(MyBot)
    config = AgentServerConfig(enable_rest_api=True)
    app = web.Application()
    base = config.base_path.rstrip("/")
    app.router.add_delete(f"{base}/{{agent_id}}", server._handle_delete_agent)

    client = await aiohttp_client(app)

    # Create agent
    agent = server.get_or_create("test-delete")
    assert "test-delete" in server._agents

    # Delete agent
    resp = await client.delete(f"/agent/test-delete")
    assert resp.status == 200

    # Verify agent is gone
    assert "test-delete" not in server._agents
