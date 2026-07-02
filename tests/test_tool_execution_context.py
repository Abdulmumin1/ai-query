from __future__ import annotations

import asyncio

import pytest

from ai_query import (
    AbortError,
    ToolExecutionContext,
    ToolExecutionProgressEvent,
    generate_text,
    step_count_is,
    stream_text,
    tool,
    turn_event_from_dict,
    turn_event_to_dict,
)
from ai_query.agents import Agent, TurnOptions
from ai_query.model import LanguageModel
from ai_query.types import StreamChunk, Tool, ToolCall
from tests.conftest import MockProvider, make_response, make_tool_call


@pytest.mark.asyncio
async def test_context_annotation_is_excluded_and_injected() -> None:
    observed: ToolExecutionContext | None = None

    @tool(description="Inspect runtime context")
    async def inspect(value: str, ctx: ToolExecutionContext) -> str:
        nonlocal observed
        observed = ctx
        await ctx.emit_progress("not observed on non-streaming API")
        return value

    provider = MockProvider(
        responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[
                    make_tool_call(
                        "inspect",
                        {"value": "ok", "ctx": "model-controlled"},
                        id="call_1",
                    )
                ],
            )
        ]
    )
    result = await generate_text(
        model=LanguageModel(provider=provider, model_id="mock"),
        prompt="inspect",
        tools={"inspect": inspect},
        metadata={"request": {"tags": ["a", "b"]}},
        stop_when=step_count_is(1),
    )

    assert inspect.parameters == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    assert result.tool_results[0].result == "ok"
    assert observed is not None
    assert observed.tool_call_id == "call_1"
    assert observed.tool_name == "inspect"
    assert observed.step_number == 1
    assert observed.turn_id is None
    assert observed.agent_id is None
    assert observed.signal.aborted is False
    assert observed.metadata["request"]["tags"] == ("a", "b")
    with pytest.raises(TypeError):
        observed.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        observed.metadata["request"]["new"] = True  # type: ignore[index]


@pytest.mark.asyncio
async def test_manual_tool_definition_injects_annotated_context() -> None:
    seen: list[str] = []

    async def execute(value: str, runtime: ToolExecutionContext) -> str:
        seen.append(runtime.tool_call_id)
        return value

    definition = Tool(
        description="Manual tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute,
    )
    provider = MockProvider(
        responses=[
            make_response(
                finish_reason="tool_use",
                tool_calls=[make_tool_call("manual", {"value": "ok"}, id="call_m")],
            )
        ]
    )

    await generate_text(
        model=LanguageModel(provider=provider, model_id="mock"),
        prompt="run",
        tools={"manual": definition},
        stop_when=step_count_is(1),
    )

    assert seen == ["call_m"]


@pytest.mark.asyncio
async def test_parallel_contexts_and_progress_are_correlated() -> None:
    contexts: dict[str, ToolExecutionContext] = {}
    release = asyncio.Event()

    @tool(description="Parallel worker")
    async def worker(name: str, ctx: ToolExecutionContext) -> str:
        contexts[name] = ctx
        await ctx.emit_progress("started", data={"name": name})
        if len(contexts) == 3:
            release.set()
        await release.wait()
        await ctx.emit_progress("finished", data={"name": name})
        return name

    calls = [
        ToolCall(id=f"call_{name}", name="worker", arguments={"name": name})
        for name in ("a", "b", "c")
    ]
    provider = MockProvider(
        stream_chunks=[
            [StreamChunk(is_final=True, finish_reason="tool_use", tool_calls=calls)]
        ]
    )
    result = stream_text(
        model=LanguageModel(provider=provider, model_id="mock"),
        prompt="run",
        tools={"worker": worker},
        metadata={"batch": "parallel"},
        stop_when=step_count_is(1),
    )

    events = [event async for event in result.event_stream]
    progress = [
        event for event in events if isinstance(event, ToolExecutionProgressEvent)
    ]

    assert set(contexts) == {"a", "b", "c"}
    assert len({id(context) for context in contexts.values()}) == 3
    assert {name: ctx.tool_call_id for name, ctx in contexts.items()} == {
        "a": "call_a",
        "b": "call_b",
        "c": "call_c",
    }
    assert all(ctx.metadata["batch"] == "parallel" for ctx in contexts.values())
    assert [(event.tool_call.id, event.message) for event in progress[:3]] == [
        ("call_a", "started"),
        ("call_b", "started"),
        ("call_c", "started"),
    ]
    assert {
        (event.tool_call.id, event.message) for event in progress[3:]
    } == {
        ("call_a", "finished"),
        ("call_b", "finished"),
        ("call_c", "finished"),
    }
    assert [event.data["name"] for event in progress[:3]] == ["a", "b", "c"]
    assert {event.data["name"] for event in progress[3:]} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_agent_turn_context_has_identity_progress_and_abort_signal() -> None:
    tool_started = asyncio.Event()
    observed: ToolExecutionContext | None = None

    @tool(description="Wait until aborted")
    async def wait_for_abort(ctx: ToolExecutionContext) -> str:
        nonlocal observed
        observed = ctx
        await ctx.emit_progress("waiting")
        tool_started.set()
        await ctx.signal.wait()
        ctx.signal.throw_if_aborted()
        return "unreachable"

    provider = MockProvider(
        stream_chunks=[
            [
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[ToolCall("call_abort", "wait_for_abort", {})],
                )
            ]
        ]
    )
    agent = Agent(
        "agent-ctx",
        model=LanguageModel(provider=provider, model_id="mock"),
        tools={"wait_for_abort": wait_for_abort},
    )
    await agent.start()
    turn = agent.turn(
        "wait",
        options=TurnOptions(metadata={"tenant": "acme"}),
    )

    async def consume_events():
        return [event async for event in turn.events()]

    event_task = asyncio.create_task(consume_events())
    await tool_started.wait()
    turn.abort("stop now")
    with pytest.raises(AbortError, match="stop now"):
        await turn.result()
    events = await event_task

    assert observed is not None
    assert observed.turn_id == turn.id
    assert observed.agent_id == "agent-ctx"
    assert observed.metadata["tenant"] == "acme"
    assert observed.signal.aborted is True
    progress = [
        event for event in events if isinstance(event, ToolExecutionProgressEvent)
    ]
    assert len(progress) == 1
    assert progress[0].tool_call.id == "call_abort"
    assert progress[0].message == "waiting"
    await agent.stop()


def test_progress_event_wire_codec_round_trip() -> None:
    event = ToolExecutionProgressEvent(
        type="tool_execution.progress",
        step_number=2,
        index=1,
        tool_call=ToolCall("call_1", "work", {"path": "x"}),
        message="halfway",
        data={"percent": 50, "raw": b"ok"},
    )

    assert turn_event_from_dict(turn_event_to_dict(event)) == event


def test_multiple_or_positional_context_parameters_fail_at_definition() -> None:
    with pytest.raises(TypeError, match="only one"):

        @tool
        def invalid(
            first: ToolExecutionContext,
            second: ToolExecutionContext,
        ) -> None:
            pass

    def positional(ctx: ToolExecutionContext, /) -> None:
        pass

    with pytest.raises(TypeError, match="positional-only"):
        tool(positional)


def test_context_stays_excluded_when_an_unrelated_hint_cannot_resolve() -> None:
    namespace: dict[str, object] = {"tool": tool}
    exec(
        "from __future__ import annotations\n"
        "@tool\n"
        "def inspect(value: MissingType, ctx: ToolExecutionContext):\n"
        "    return value\n",
        namespace,
    )

    definition = namespace["inspect"]
    assert isinstance(definition, Tool)
    assert definition.context_parameter == "ctx"
    assert "ctx" not in definition.parameters["properties"]
