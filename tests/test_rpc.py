from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from ai_query.agents import Agent, action, AgentServer, MemoryStorage


class RpcAgent(Agent):
    def __init__(self, id: str, **kwargs: Any):
        super().__init__(
            id,
            storage=MemoryStorage(),
            initial_state={"reputation": {}},
            **kwargs
        )

    @action
    async def get_id(self) -> str:
        return self.id

    @action
    async def get_state(self) -> dict[str, Any]:
        return self.state

    @action
    async def update_reputation(self, amount: int) -> int:
        assert self.context.connection is not None
        user_id = self.context.connection.state.get("user_id", "anonymous")
        reputation = self.state.setdefault("reputation", {})
        reputation[user_id] = reputation.get(user_id, 0) + amount
        await self.update_state(reputation=reputation)
        return reputation[user_id]

    async def on_connect(self, connection, ctx):
        connection.state["user_id"] = ctx.metadata.get("user_id", "anonymous")
        await super().on_connect(connection, ctx)


@pytest_asyncio.fixture
async def rpc_server():
    server = AgentServer(RpcAgent)
    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    yield
    await runner.cleanup()


@pytest.mark.asyncio
async def test_http_action_call(rpc_server):
    async with aiohttp.ClientSession() as session:
        # Test basic action
        async with session.post(
            "http://localhost:8080/agent/test-agent/action/get_id"
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["result"] == "test-agent"

        # Test action with no params
        async with session.post(
            "http://localhost:8080/agent/test-agent/action/get_state"
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["result"] == {"reputation": {}}


@pytest.mark.asyncio
async def test_websocket_action_call(rpc_server):
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            "http://localhost:8080/agent/ws-agent/ws?user_id=alice"
        ) as ws:
            # Call action with params
            call_id = "123"
            await ws.send_json(
                {
                    "type": "action",
                    "name": "update_reputation",
                    "params": {"amount": 10},
                    "call_id": call_id,
                }
            )

            # Wait for result, ignoring other messages until we get the one we want
            response = {}
            for _ in range(5): # Try up to 5 times
                response = await ws.receive_json()
                if response.get("type") == "action_result" and response.get("call_id") == call_id:
                    break
            
            assert response["type"] == "action_result"
            assert response["call_id"] == call_id
            assert response["result"] == 10

            # Check state
            async with session.get(
                "http://localhost:8080/agent/ws-agent/state"
            ) as resp:
                data = await resp.json()
                assert data["reputation"]["alice"] == 10
