from __future__ import annotations

import pytest

from ai_query import (
    Agent,
    Field,
    generate_text,
    has_tool_call,
    step_count_is,
    stream_text,
    tool,
)
from ai_query.providers import FauxProvider, FauxResponse, faux
from ai_query.types import ReasoningEvent, ToolCall


@pytest.mark.asyncio
async def test_faux_generate_records_calls_and_fails_when_exhausted():
    model = faux(responses=[FauxResponse(text="hello")])
    provider = model.provider
    assert isinstance(provider, FauxProvider)

    result = await generate_text(model=model, prompt="hi")

    assert result.text == "hello"
    assert provider.call_count == 1
    assert provider.calls[0].method == "generate"
    assert provider.calls[0].messages[-1].content == "hi"
    provider.assert_exhausted()

    with pytest.raises(RuntimeError, match="No faux response queued"):
        await generate_text(model=model, prompt="again")


@pytest.mark.asyncio
async def test_faux_stream_preserves_chunks_reasoning_and_tools():
    @tool(description="Look up a value")
    def lookup(
        query: str = Field(description="Value to look up"),
    ) -> str:
        return f"result:{query}"

    model = faux(
        responses=[
            FauxResponse(
                text="ignored when chunks are explicit",
                chunks=["hel", "lo"],
                reasoning_events=[
                    ReasoningEvent(
                        kind="delta",
                        provider="faux",
                        text="thinking",
                    )
                ],
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup",
                        arguments={"query": "weather"},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )

    result = stream_text(
        model=model,
        prompt="search",
        tools={"lookup": lookup},
        stop_when=has_tool_call("lookup"),
    )
    events = [event async for event in result.event_stream]

    assert [event.type for event in events] == [
        "step.started",
        "reasoning.delta",
        "text.delta",
        "text.delta",
        "tool_call.started",
        "tool_call.delta",
        "tool_call.ready",
        "tool_execution.started",
        "tool_execution.finished",
        "tool_result",
        "step.finished",
        "stream.finished",
    ]
    assert await result.text == "hello"


@pytest.mark.asyncio
async def test_faux_drives_a_complete_agent_tool_turn():
    @tool(description="Look up a value")
    async def lookup(
        query: str = Field(description="Value to look up"),
    ) -> str:
        return f"result:{query}"

    def final_response(call):
        assert call.messages[-1].role == "tool"
        return FauxResponse(text="done", chunks=["do", "ne"])

    model = faux(
        responses=[
            FauxResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup",
                        arguments={"query": "chump"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            final_response,
        ]
    )
    agent = Agent(
        "test",
        model=model,
        tools={"lookup": lookup},
        stop_when=step_count_is(4),
    )

    async with agent:
        events = [event async for event in agent.turn("go").events()]

    assert events[-1].type == "turn.finished"
    assert events[-1].result.text == "done"
    assert [event.type for event in events].count("tool_result") == 1
    assert isinstance(model.provider, FauxProvider)
    assert model.provider.call_count == 2
    model.provider.assert_exhausted()


def test_faux_queue_can_be_replaced_and_extended():
    provider = FauxProvider([FauxResponse(text="old")])

    provider.set_responses([FauxResponse(text="first")])
    provider.append_responses([FauxResponse(text="second")])

    assert provider.pending_response_count == 2
