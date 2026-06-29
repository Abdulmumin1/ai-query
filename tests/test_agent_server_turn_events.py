from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx
import pytest

from ai_query import step_count_is, tool
from ai_query.agents import Agent, AgentServer, AgentServerConfig, MemoryStorage
from ai_query.agents.remote import RemoteAgent
from ai_query.agents.transport.http import HTTPTransport
from ai_query.agents.turn import TextDelta, TurnFinished, TurnStarted
from ai_query.model import LanguageModel
from ai_query.providers.base import BaseProvider
from ai_query.types import (
    GenerateTextResult,
    Message,
    ProviderOptions,
    ReasoningEvent,
    StreamChunk,
    ToolCall,
    ToolSet,
    Usage,
)


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    frames = []
    for block in text.split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event_name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event_name and data_lines:
            frames.append((event_name, json.loads("\n".join(data_lines))))
    return frames


class _SequenceProvider(BaseProvider):
    name = "sequence"

    def __init__(self, chunks: list[list[StreamChunk]]):
        super().__init__(api_key="test")
        self.chunks = chunks
        self.stream_count = 0

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> GenerateTextResult:
        raise NotImplementedError

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: ToolSet | None = None,
        provider_options: ProviderOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        chunks = self.chunks[self.stream_count]
        self.stream_count += 1
        for chunk in chunks:
            yield chunk


@pytest.mark.asyncio
async def test_agent_server_streams_real_typed_turn_with_parallel_tools(aiohttp_client):
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}

    async def run_probe(name: str, delay: float) -> str:
        starts[name] = time.perf_counter()
        await asyncio.sleep(delay)
        finishes[name] = time.perf_counter()
        return f"{name}:ok"

    @tool(description="Slow probe")
    async def slow_probe() -> str:
        return await run_probe("slow_probe", 0.06)

    @tool(description="Medium probe")
    async def medium_probe() -> str:
        return await run_probe("medium_probe", 0.04)

    @tool(description="Fast probe")
    async def fast_probe() -> str:
        return await run_probe("fast_probe", 0.02)

    calls = [
        ToolCall(id="call_slow", name="slow_probe", arguments={}),
        ToolCall(id="call_medium", name="medium_probe", arguments={}),
        ToolCall(id="call_fast", name="fast_probe", arguments={}),
    ]
    provider = _SequenceProvider(
        [
            [
                StreamChunk(
                    reasoning_events=[
                        ReasoningEvent(
                            kind="summary",
                            provider="sequence",
                            text="Run all probes together",
                        )
                    ]
                ),
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=calls,
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
            ],
            [
                StreamChunk(text="All probes completed."),
                StreamChunk(
                    is_final=True,
                    finish_reason="stop",
                    usage=Usage(input_tokens=20, output_tokens=4, total_tokens=24),
                ),
            ],
        ]
    )
    model = LanguageModel(provider=provider, model_id="sequence-model")

    class EventAgent(Agent):
        def __init__(self, agent_id: str):
            super().__init__(
                agent_id,
                storage=MemoryStorage(),
                model=model,
                tools={
                    "slow_probe": slow_probe,
                    "medium_probe": medium_probe,
                    "fast_probe": fast_probe,
                },
                stop_when=step_count_is(3),
            )

    server = AgentServer(EventAgent, config=AgentServerConfig(enable_rest_api=True))
    client = await aiohttp_client(server.create_app())

    response = await client.post(
        "/agent/parallel/chat?stream=events",
        json={"message": "Run all probes"},
    )

    assert response.status == 200
    frames = _parse_sse(await response.text())
    event_types = [name for name, _ in frames]
    assert all(name == payload["type"] for name, payload in frames)
    assert event_types[0:2] == ["turn.started", "step.started"]
    assert event_types.count("reasoning.delta") == 1
    assert event_types.count("tool_call.started") == 3
    assert event_types.count("tool_call.delta") == 3
    assert event_types.count("tool_call.ready") == 3
    assert event_types.count("tool_execution.started") == 3
    assert event_types.count("tool_execution.finished") == 3
    assert event_types.count("tool_result") == 3
    assert event_types[-1] == "turn.finished"
    assert not {"start", "chunk", "end", "error"}.intersection(event_types)

    ready_payloads = [payload for name, payload in frames if name == "tool_call.ready"]
    assert [payload["tool_call"]["name"] for payload in ready_payloads] == [
        "slow_probe",
        "medium_probe",
        "fast_probe",
    ]
    finish_payloads = [
        payload for name, payload in frames if name == "tool_execution.finished"
    ]
    assert [payload["tool_call"]["name"] for payload in finish_payloads] == [
        "fast_probe",
        "medium_probe",
        "slow_probe",
    ]
    final = frames[-1][1]["result"]
    assert final["text"] == "All probes completed."
    assert final["usage"] == {
        "input_tokens": 20,
        "output_tokens": 4,
        "cached_tokens": 0,
        "total_tokens": 24,
    }
    assert len(final["steps"]) == 2

    first_start = min(starts.values())
    assert max(starts.values()) - first_start < 0.02
    assert max(finishes.values()) - first_start < 0.09
    assert [name for name, _ in sorted(finishes.items(), key=lambda item: item[1])] == [
        "fast_probe",
        "medium_probe",
        "slow_probe",
    ]


class _FailingProvider(BaseProvider):
    name = "failing"

    def __init__(self):
        super().__init__(api_key="test")

    async def generate(self, **kwargs: Any) -> GenerateTextResult:
        raise RuntimeError("provider failed")

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise RuntimeError("provider failed")
        yield StreamChunk()


@pytest.mark.asyncio
async def test_agent_server_streams_typed_turn_failure(aiohttp_client):
    model = LanguageModel(provider=_FailingProvider(), model_id="failing-model")

    class FailingAgent(Agent):
        def __init__(self, agent_id: str):
            super().__init__(agent_id, storage=MemoryStorage(), model=model)

    server = AgentServer(FailingAgent, config=AgentServerConfig(enable_rest_api=True))
    client = await aiohttp_client(server.create_app())

    response = await client.post(
        "/agent/failing/chat?stream=events", json={"message": "Fail"}
    )
    frames = _parse_sse(await response.text())

    assert [name for name, _ in frames][-1] == "turn.failed"
    assert frames[-1][1] == {
        "type": "turn.failed",
        "error": "provider failed",
        "error_type": "RuntimeError",
        "aborted": False,
    }


@pytest.mark.asyncio
async def test_remote_agent_round_trips_events_from_live_agent_server(aiohttp_server):
    provider = _SequenceProvider(
        [
            [
                StreamChunk(text="Hello remotely"),
                StreamChunk(
                    is_final=True,
                    finish_reason="stop",
                    usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
                ),
            ]
        ]
    )
    model = LanguageModel(provider=provider, model_id="remote-model")

    class RemoteEventAgent(Agent):
        def __init__(self, agent_id: str):
            super().__init__(agent_id, storage=MemoryStorage(), model=model)

    server = AgentServer(
        RemoteEventAgent, config=AgentServerConfig(enable_rest_api=True)
    )
    test_server = await aiohttp_server(server.create_app())
    base_url = str(test_server.make_url("/agent")).rstrip("/")
    http_client = httpx.AsyncClient()
    remote = RemoteAgent(
        HTTPTransport(base_url=base_url, client=http_client), "remote"
    )

    try:
        events = [event async for event in remote.events("Say hello")]
    finally:
        await remote.close()
        await http_client.aclose()

    assert [event.type for event in events] == [
        "turn.started",
        "step.started",
        "text.delta",
        "step.finished",
        "turn.finished",
    ]
    assert isinstance(events[0], TurnStarted)
    assert isinstance(events[2], TextDelta)
    assert events[2].text == "Hello remotely"
    assert isinstance(events[-1], TurnFinished)
    assert events[-1].result.usage == Usage(
        input_tokens=3, output_tokens=2, total_tokens=5
    )


@pytest.mark.asyncio
async def test_closing_turn_event_stream_aborts_running_turn():
    gate = asyncio.Event()

    class HangingProvider(_FailingProvider):
        async def stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
            await gate.wait()
            yield StreamChunk(is_final=True, finish_reason="stop")

    model = LanguageModel(provider=HangingProvider(), model_id="hanging-model")

    class CapturingAgent(Agent):
        last_turn = None

        def __init__(self, agent_id: str):
            super().__init__(agent_id, storage=MemoryStorage(), model=model)

        def turn(self, message, *, options=None):
            self.last_turn = super().turn(message, options=options)
            return self.last_turn

    agent = CapturingAgent("disconnect")
    await agent.start()
    stream = agent.handle_request_stream(
        {"action": "chat", "message": "Wait", "stream": "events"}
    )

    first_frame = await anext(stream)
    assert first_frame.startswith("event: turn.started\n")
    await stream.aclose()

    assert agent.last_turn is not None
    assert agent.last_turn._controller.signal.aborted
    assert agent.last_turn._controller.signal.reason == "Turn event stream disconnected"
    with pytest.raises(Exception, match="disconnected"):
        await agent.last_turn.result()
    await agent.stop()
