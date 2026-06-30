from __future__ import annotations

import json

import httpx
import pytest

from ai_query.agents.remote import RemoteAgent
from ai_query.agents.transport.http import HTTPTransport
from ai_query.agents.turn import TextDelta, TurnStarted
from ai_query.agents.turn_codec import turn_event_to_sse
from ai_query.types import Message


@pytest.mark.asyncio
async def test_http_transport_reconstructs_typed_turn_events():
    wire_events = [
        TurnStarted(
            type="turn.started",
            turn_id="turn_1",
            agent_id="worker",
            message=Message(role="user", content="hello"),
            created_at=1.0,
        ),
        TextDelta(type="text.delta", text="hello"),
    ]
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_request["url"] = str(request.url)
        seen_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=": keepalive\n\n"
            + "".join(turn_event_to_sse(event) for event in wire_events),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HTTPTransport(base_url="https://example.test/agent", client=client)
    remote = RemoteAgent(transport, "worker")

    events = [event async for event in remote.events("hello")]

    assert events == wire_events
    assert seen_request == {
        "url": "https://example.test/agent/worker/chat?stream=events",
        "body": {"action": "chat", "message": "hello"},
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_http_transport_preserves_legacy_text_stream():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/agent/worker/chat?stream=true"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                ": keepalive\n\n"
                "event: start\ndata: \n\n"
                f"event: chunk\ndata: {json.dumps('Hello ')}\n\n"
                f"event: chunk\ndata: {json.dumps('world')}\n\n"
                f"event: end\ndata: {json.dumps('Hello world')}\n\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HTTPTransport(base_url="https://example.test/agent", client=client)

    chunks = [chunk async for chunk in transport.stream("worker", "hello")]

    assert chunks == ["Hello ", "world"]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_transport_rejects_mismatched_sse_event_name():
    event = TextDelta(type="text.delta", text="hello")

    async def handler(request: httpx.Request) -> httpx.Response:
        frame = turn_event_to_sse(event).replace(
            "event: text.delta", "event: reasoning.delta"
        )
        return httpx.Response(200, text=frame)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HTTPTransport(base_url="https://example.test/agent", client=client)

    with pytest.raises(ValueError, match="does not match"):
        _ = [event async for event in transport.stream_events("worker", "hello")]
    await client.aclose()
