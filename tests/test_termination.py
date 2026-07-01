from __future__ import annotations

import pytest

from ai_query import (
    AfterToolCallResult,
    StopDecision,
    StreamFinishedEvent,
    generate_text,
    step_count_is,
    stream_text,
    tool,
)
from ai_query.agents import Agent, MemoryStorage, TurnOptions
from ai_query.model import LanguageModel
from ai_query.types import (
    AbortController,
    AbortError,
    RetryPolicy,
    StepControl,
    StreamChunk,
    ToolCall,
)
from tests.conftest import MockProvider, make_response


async def _stream_finished(result) -> StreamFinishedEvent:
    events = [event async for event in result.event_stream]
    assert isinstance(events[-1], StreamFinishedEvent)
    return events[-1]


@pytest.mark.asyncio
async def test_normal_completion_has_matching_generate_and_stream_diagnostics():
    generate_provider = MockProvider(
        responses=[make_response(text="Done", finish_reason="stop")]
    )
    stream_provider = MockProvider(
        stream_chunks=[
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ]
    )

    generated = await generate_text(
        model=LanguageModel(generate_provider, "test"), prompt="Finish"
    )
    streamed = stream_text(
        model=LanguageModel(stream_provider, "test"), prompt="Finish"
    )
    finished = await _stream_finished(streamed)

    assert generated.termination == finished.termination
    assert await streamed.termination == finished.termination
    assert generated.termination is not None
    assert generated.termination.kind == "completed"
    assert generated.termination.provider_finish_reason == "stop"
    assert generated.termination.final_step_number == 1
    assert generated.termination.has_text
    assert not generated.termination.has_tool_calls


@pytest.mark.asyncio
async def test_step_limit_is_not_reported_as_generic_stop_condition():
    @tool(description="Continue")
    def continue_work() -> str:
        return "continued"

    calls = [
        ToolCall(id=f"call_{index}", name="continue_work", arguments={})
        for index in range(2)
    ]
    generate_provider = MockProvider(
        responses=[
            make_response(text="", finish_reason="tool_use", tool_calls=[call])
            for call in calls
        ]
    )
    stream_provider = MockProvider(
        stream_chunks=[
            [
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[call],
                )
            ]
            for call in calls
        ]
    )

    generated = await generate_text(
        model=LanguageModel(generate_provider, "test"),
        prompt="Continue",
        tools={"continue_work": continue_work},
        stop_when=step_count_is(2),
    )
    finished = await _stream_finished(
        stream_text(
            model=LanguageModel(stream_provider, "test"),
            prompt="Continue",
            tools={"continue_work": continue_work},
            stop_when=step_count_is(2),
        )
    )

    assert generated.termination == finished.termination
    assert generated.termination is not None
    assert generated.termination.kind == "step_limit"
    assert generated.termination.stop_condition == "step_count_is(2)"
    assert generated.termination.reason == "Reached the step limit of 2"
    assert generated.termination.final_step_number == 2
    assert generated.termination.has_tool_calls


@pytest.mark.asyncio
async def test_custom_stop_decision_exposes_name_and_reason():
    @tool(description="Work")
    def work() -> str:
        return "worked"

    def budget_reached(steps):
        return StopDecision(
            stop=bool(steps),
            name="token_budget",
            reason="Application token budget reached",
        )

    provider = MockProvider(
        responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[ToolCall(id="call_1", name="work", arguments={})],
            )
        ]
    )

    result = await generate_text(
        model=LanguageModel(provider, "test"),
        prompt="Work",
        tools={"work": work},
        stop_when=budget_reached,
    )

    assert result.termination is not None
    assert result.termination.kind == "stop_condition"
    assert result.termination.stop_condition == "token_budget"
    assert result.termination.reason == "Application token budget reached"


@pytest.mark.asyncio
async def test_before_step_hook_can_stop_before_first_provider_call():
    async def stop_before_step(_event):
        return StepControl(stop=True, stop_reason="Agent is in maintenance mode")

    generate_provider = MockProvider()
    stream_provider = MockProvider()

    generated = await generate_text(
        model=LanguageModel(generate_provider, "test"),
        prompt="Start",
        on_step_start=stop_before_step,
    )
    finished = await _stream_finished(
        stream_text(
            model=LanguageModel(stream_provider, "test"),
            prompt="Start",
            on_step_start=stop_before_step,
        )
    )

    assert generated.termination == finished.termination
    assert generated.text == ""
    assert generated.steps == []
    assert generate_provider.call_count == 0
    assert stream_provider.stream_call_count == 0
    assert generated.termination is not None
    assert generated.termination.kind == "hook_stop"
    assert generated.termination.reason == "Agent is in maintenance mode"
    assert generated.termination.final_step_number == 0


@pytest.mark.asyncio
async def test_after_tool_hook_termination_has_priority_and_reason():
    @tool(description="Finish")
    def finish() -> str:
        return "finished"

    async def terminate_after_tool(_event):
        return AfterToolCallResult(
            terminate=True,
            terminate_reason="The finish tool completed the workflow",
        )

    call = ToolCall(id="call_1", name="finish", arguments={})
    generate_provider = MockProvider(
        responses=[make_response(text="", finish_reason="tool_use", tool_calls=[call])]
    )
    stream_provider = MockProvider(
        stream_chunks=[
            [
                StreamChunk(
                    is_final=True,
                    finish_reason="tool_use",
                    tool_calls=[call],
                )
            ]
        ]
    )

    generated = await generate_text(
        model=LanguageModel(generate_provider, "test"),
        prompt="Finish",
        tools={"finish": finish},
        stop_when=step_count_is(1),
        after_tool_call=terminate_after_tool,
    )
    finished = await _stream_finished(
        stream_text(
            model=LanguageModel(stream_provider, "test"),
            prompt="Finish",
            tools={"finish": finish},
            stop_when=step_count_is(1),
            after_tool_call=terminate_after_tool,
        )
    )

    assert generated.termination == finished.termination
    assert generated.termination is not None
    assert generated.termination.kind == "tool_terminated"
    assert generated.termination.stop_condition is None
    assert generated.termination.tool_name == "finish"
    assert generated.termination.reason == "The finish tool completed the workflow"


@pytest.mark.asyncio
async def test_empty_tool_turn_reports_last_tool_error():
    call = ToolCall(id="call_1", name="missing_tool", arguments={})
    provider = MockProvider(
        responses=[make_response(text="", finish_reason="tool_use", tool_calls=[call])]
    )

    result = await generate_text(
        model=LanguageModel(provider, "test"),
        prompt="Call it",
        stop_when=step_count_is(1),
    )

    assert result.text == ""
    assert result.termination is not None
    assert result.termination.kind == "step_limit"
    assert result.termination.has_tool_calls
    assert result.termination.last_tool_error == "Error: Unknown tool 'missing_tool'"


@pytest.mark.asyncio
async def test_pre_aborted_requests_carry_aborted_termination():
    controller = AbortController()
    controller.abort("cancelled by user")
    model = LanguageModel(MockProvider(), "test")

    with pytest.raises(AbortError) as generate_error:
        await generate_text(
            model=model,
            prompt="Start",
            signal=controller.signal,
        )

    assert generate_error.value.termination is not None
    assert generate_error.value.termination.kind == "aborted"
    assert generate_error.value.termination.reason == "cancelled by user"
    assert generate_error.value.termination.final_step_number == 0

    streamed = stream_text(model=model, prompt="Start", signal=controller.signal)
    with pytest.raises(AbortError) as stream_error:
        await streamed.text

    assert stream_error.value.termination is not None
    assert stream_error.value.termination.kind == "aborted"
    assert stream_error.value.termination.reason == "cancelled by user"


@pytest.mark.asyncio
async def test_exhausted_retries_carry_failed_termination():
    class FailingProvider(MockProvider):
        async def generate(self, **kwargs):
            raise ConnectionError("provider unavailable")

        async def stream(self, **kwargs):
            raise ConnectionError("provider unavailable")
            yield StreamChunk()

    policy = RetryPolicy(max_attempts=2, initial_delay=0, jitter=False)

    with pytest.raises(ConnectionError) as generate_error:
        await generate_text(
            model=LanguageModel(FailingProvider(), "test"),
            prompt="Start",
            retry=policy,
        )

    assert generate_error.value.termination.kind == "failed"
    assert generate_error.value.termination.error_type == "ConnectionError"
    assert generate_error.value.termination.final_step_number == 1

    streamed = stream_text(
        model=LanguageModel(FailingProvider(), "test"),
        prompt="Start",
        retry=policy,
    )
    with pytest.raises(ConnectionError) as stream_error:
        await streamed.text

    assert stream_error.value.termination.kind == "failed"
    assert stream_error.value.termination.error_type == "ConnectionError"
    assert stream_error.value.termination.final_step_number == 1


@pytest.mark.asyncio
async def test_agent_turn_exposes_completion_and_failure_termination():
    completed_provider = MockProvider(
        stream_chunks=[
            [
                StreamChunk(text="Done"),
                StreamChunk(is_final=True, finish_reason="stop"),
            ]
        ]
    )
    completed_agent = Agent(
        "completed",
        storage=MemoryStorage(),
        model=LanguageModel(completed_provider, "test"),
    )
    await completed_agent.start()

    completed = await completed_agent.run("Finish")

    assert completed.termination is not None
    assert completed.termination.kind == "completed"
    await completed_agent.stop()

    class FailingProvider(MockProvider):
        async def stream(self, **kwargs):
            raise RuntimeError("model failed")
            yield StreamChunk()

    failed_agent = Agent(
        "failed",
        storage=MemoryStorage(),
        model=LanguageModel(FailingProvider(), "test"),
    )
    await failed_agent.start()
    turn = failed_agent.turn("Fail", options=TurnOptions())

    events = [event async for event in turn.events()]
    failed = events[-1]
    assert failed.type == "turn.failed"
    assert failed.termination is not None
    assert failed.termination.kind == "failed"
    assert failed.termination.error_type == "RuntimeError"
    with pytest.raises(RuntimeError) as error:
        await turn.result()
    assert error.value.termination == failed.termination
    await failed_agent.stop()


@pytest.mark.asyncio
async def test_agent_turn_exposes_hook_stop_and_abort_termination():
    class HookStopAgent(Agent):
        async def before_step(self, _ctx):
            return StepControl(stop=True, stop_reason="Paused by agent policy")

    hook_agent = HookStopAgent(
        "hook-stop",
        storage=MemoryStorage(),
        model=LanguageModel(MockProvider(), "test"),
    )
    await hook_agent.start()

    stopped = await hook_agent.run("Start")

    assert stopped.termination is not None
    assert stopped.termination.kind == "hook_stop"
    assert stopped.termination.reason == "Paused by agent policy"
    assert stopped.steps == []
    await hook_agent.stop()

    controller = AbortController()
    controller.abort("Stopped remotely")
    abort_agent = Agent(
        "aborted",
        storage=MemoryStorage(),
        model=LanguageModel(MockProvider(), "test"),
    )
    await abort_agent.start()
    turn = abort_agent.turn("Start", options=TurnOptions(signal=controller.signal))

    events = [event async for event in turn.events()]
    failed = events[-1]
    assert failed.type == "turn.failed"
    assert failed.termination is not None
    assert failed.termination.kind == "aborted"
    assert failed.termination.reason == "Stopped remotely"
    with pytest.raises(AbortError) as error:
        await turn.result()
    assert error.value.termination == failed.termination
    await abort_agent.stop()
