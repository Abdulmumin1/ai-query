"""Tests for the FastAPI adapter helpers."""

import asyncio
import json
import logging

import pytest

import ai_query.adapters.fastapi as fastapi_adapter


class _StreamingAgent:
    id = "fastapi-stream"

    async def handle_request_stream(self, request):
        await asyncio.sleep(0.03)
        yield "event: chunk\ndata: done\n\n"


class _ErrorAgent:
    id = "fastapi-error"

    async def handle_request_stream(self, request):
        raise RuntimeError("stream broke")
        yield "unreachable"


class _EventModeAgent:
    id = "fastapi-events"

    async def handle_request_stream(self, request):
        yield f"event: request\ndata: {json.dumps(request)}\n\n"


@pytest.mark.asyncio
async def test_fastapi_stream_generator_sends_keepalive_while_idle(monkeypatch):
    """FastAPI streaming chat should keep SSE connections alive while idle."""
    monkeypatch.setattr(fastapi_adapter, "_SSE_KEEPALIVE_INTERVAL", 0.01)
    router = object.__new__(fastapi_adapter.AgentRouter)

    stream = router._stream_generator(_StreamingAgent(), "Hi")

    assert await anext(stream) == ": keepalive\n\n"
    chunks = []
    for _ in range(4):
        chunk = await anext(stream)
        chunks.append(chunk)
        if chunk == "event: chunk\ndata: done\n\n":
            break
    assert "event: chunk\ndata: done\n\n" in chunks


@pytest.mark.asyncio
async def test_fastapi_stream_generator_sends_json_error_and_logs_traceback(caplog):
    """FastAPI streaming chat errors should be structured and logged."""
    router = object.__new__(fastapi_adapter.AgentRouter)
    caplog.set_level(logging.ERROR, logger=fastapi_adapter.__name__)

    stream = router._stream_generator(_ErrorAgent(), "Hi")

    assert await anext(stream) == (
        f"event: error\ndata: {json.dumps({'error': 'stream broke', 'type': 'RuntimeError'})}\n\n"
    )
    assert "Streaming chat failed for agent fastapi-error" in caplog.text
    assert "Traceback" in caplog.text


@pytest.mark.asyncio
async def test_fastapi_stream_generator_passes_typed_event_mode_and_request_data():
    router = object.__new__(fastapi_adapter.AgentRouter)
    expected_request = {
        "message": "Hi",
        "metadata": {"source": "test"},
        "action": "chat",
        "stream": "events",
    }

    stream = router._stream_generator(
        _EventModeAgent(),
        "Hi",
        stream_mode="events",
        request_data={"message": "Hi", "metadata": {"source": "test"}},
    )

    assert await anext(stream) == (
        "event: request\ndata: "
        f"{json.dumps(expected_request)}"
        "\n\n"
    )
