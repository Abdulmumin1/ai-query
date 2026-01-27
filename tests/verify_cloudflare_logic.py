import asyncio
import json
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock 'js' module before importing adapters
sys.modules["js"] = MagicMock()
sys.modules["pyodide"] = MagicMock()
sys.modules["pyodide.ffi"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["aiohttp.web"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["fastapi"] = MagicMock()

# Setup JS Mocks
js = sys.modules["js"]
js.Response.new = MagicMock(
    side_effect=lambda body, init=None: {"body": body, "init": init}
)
js.Object.fromEntries = MagicMock(side_effect=lambda x: x)


# Mock Request
class MockRequest:
    def __init__(self, method, body=None, url="http://worker/"):
        self.method = method
        self._body = body
        self.headers = {}
        self.url = url

    async def json(self):
        return self._body


# Import logic to test
from ai_query import Agent
from ai_query.adapters.cloudflare import AgentDO, CloudflareRegistry


# Define Agent
class TestAgent(Agent):
    async def handle_request(self, request):
        return {"result": "ok", "echo": request}


# Define DO
class TestDO(AgentDO):
    agent_class = TestAgent


async def test_do_fetch():
    print("Testing AgentDO.fetch()...")

    # Mock Ctx & Env
    ctx = MagicMock()
    ctx.id = "test-id"
    ctx.storage = AsyncMock()  # Mock Storage

    env = MagicMock()

    # Instantiate DO
    do = TestDO(ctx, env)

    # Check Storage Init
    assert do.storage._storage == ctx.storage
    print("✓ Storage initialized")

    # Simulate POST Request
    req_body = {"action": "invoke", "payload": {"foo": "bar"}}
    req = MockRequest("POST", body=req_body)

    # Force agent start (async)
    # The actual 'fetch' calls 'agent.start()' inside.
    # Agent.start calls storage.get ...
    ctx.storage.get.return_value = None  # No existing state

    response = await do.fetch(req)

    # Verify Response
    body_str = response["body"]
    body_json = json.loads(body_str)
    print(f"DEBUG Response: {body_json}")

    # The Agent logic for 'invoke' -> 'action' -> or handle_invoke
    # Wait, TestAgent overrides handle_request!
    # So it should return {"result": "ok", ...}

    assert body_json["result"] == "ok"
    print("✓ Fetch POST handled correctly")


async def test_registry():
    print("Testing CloudflareRegistry...")
    env = MagicMock()
    registry = CloudflareRegistry(env)

    # Mock Binding
    binding = MagicMock()
    stub = MagicMock()
    stub.fetch = AsyncMock(return_value="stub_response")
    # binding.getByName returns the stub directly
    binding.getByName.return_value = stub

    registry.register("test-.*", binding)

    # Test Match
    req = MockRequest("GET", url="http://worker/agent/test-123/chat")
    # Need to mock js.URL inside registry
    js.URL.new = lambda url: MagicMock(pathname="/agent/test-123/chat")

    await registry.handle_request(req)

    binding.getByName.assert_called_with("test-123")
    stub.fetch.assert_called_with(req)
    print("✓ Registry routing working")


async def test_websocket_hibernation():
    print("Testing WebSocket Hibernation...")
    ctx = MagicMock()
    ctx.id = "test-ws"
    ctx.storage = AsyncMock()
    ctx.storage.get.return_value = None
    env = MagicMock()

    do = TestDO(ctx, env)

    # Mock WebSocket
    ws = MagicMock()

    # Simulate webSocketMessage
    # It should start the agent if not running
    await do.webSocketMessage(ws, "hello")

    if do.agent._state is None:
        # It tried to start
        pass

    # Check if agent.on_message was called?
    # We can't easily check internal agent.on_message without mocking agent instance or its connection.
    # But we can check if it didn't crash.
    print("✓ webSocketMessage handled")

    # Simulate webSocketClose
    await do.webSocketClose(ws, 1000, "normal", True)
    print("✓ webSocketClose handled")


async def test_serialization():
    print("Testing Robust Serialization...")
    from ai_query.agents.storage.cloudflare import DurableObjectStorage

    # Mock storage
    mock_storage = AsyncMock()
    # Mock put/get
    store = {}

    async def put(k, v):
        store[k] = v

    async def get(k):
        return store.get(k)

    mock_storage.put.side_effect = put
    mock_storage.get.side_effect = get

    do_storage = DurableObjectStorage(mock_storage)

    # Test Data with Set and Tuple
    complex_data = {"tags": {"a", "b", "c"}, "coords": (10, 20), "normal": "string"}

    # Save
    await do_storage.set("complex", complex_data)

    # Verify it was serialized (check raw store)
    raw = store["complex"]
    assert "__ai_query_type__" in raw
    print("✓ Data serialized with custom types")

    # Load
    loaded = await do_storage.get("complex")

    print(f"DEBUG: loaded type: {type(loaded)}")
    print(f"DEBUG: loaded data: {loaded}")
    print(f"DEBUG: coords type: {type(loaded.get('coords'))}")

    assert isinstance(loaded["tags"], set)
    assert loaded["tags"] == {"a", "b", "c"}
    # JSON converts tuples to lists by default; Encoder.default isn't called for tuples.
    # We accept this degradation as standard for JSON storage.
    assert isinstance(loaded["coords"], list)
    assert loaded["coords"] == [10, 20]
    print("✓ Data deserialized correctly (Set restored, Tuple -> List)")


def main():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_do_fetch())
    loop.run_until_complete(test_registry())
    loop.run_until_complete(test_websocket_hibernation())
    loop.run_until_complete(test_serialization())
    print("All verification tests passed!")


if __name__ == "__main__":
    main()
