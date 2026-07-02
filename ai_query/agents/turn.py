from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal

from ai_query.agents.hooks import AgentHooks
from ai_query.types import (
    AbortController,
    AbortError,
    AbortSignal,
    AfterToolCallEvent,
    BeforeToolCallEvent,
    Message,
    ReasoningConfig,
    ReasoningEvent,
    RetryEvent,
    RetryPolicy,
    StepControl,
    StepFinishEvent,
    StepResult,
    StepStartEvent,
    StopCondition,
    ToolSet,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallReadyEvent,
    ToolCallStartedEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionProgressEvent,
    ToolExecutionStartedEvent,
    ToolResultEvent,
    TurnTermination,
    Usage,
    build_turn_termination,
)

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent, Content
    from ai_query.types import ProviderOptions


@dataclass
class TurnOptions:
    tools: ToolSet | None = None
    stop_when: StopCondition | list[StopCondition] | None = None
    reasoning: ReasoningConfig | None = None
    provider_options: "ProviderOptions | None" = None
    retry: RetryPolicy | None = None
    signal: AbortSignal | None = None
    hooks: AgentHooks | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    turn_id: str
    agent_id: str
    text: str
    finish_reason: str | None
    usage: Usage | None
    steps: list[StepResult]
    started_at: float
    ended_at: float
    output_message: Message
    termination: TurnTermination | None = None


@dataclass
class TurnStarted:
    type: Literal["turn.started"]
    turn_id: str
    agent_id: str
    message: Message
    created_at: float


@dataclass
class TextDelta:
    type: Literal["text.delta"]
    text: str


@dataclass
class ReasoningDelta:
    type: Literal["reasoning.delta"]
    event: ReasoningEvent


@dataclass
class StepStarted:
    type: Literal["step.started"]
    step_number: int


@dataclass
class StepFinished:
    type: Literal["step.finished"]
    step_number: int
    step: StepResult
    usage: Usage | None


@dataclass
class StepRetrying:
    type: Literal["step.retrying"]
    step_number: int
    attempt: int
    max_attempts: int
    delay: float
    error: str


@dataclass
class TurnFinished:
    type: Literal["turn.finished"]
    result: TurnResult


@dataclass
class TurnFailed:
    type: Literal["turn.failed"]
    error: str
    error_type: str | None = None
    aborted: bool = False
    termination: TurnTermination | None = None


TurnEvent = (
    TurnStarted
    | TextDelta
    | ReasoningDelta
    | StepStarted
    | StepRetrying
    | ToolCallStartedEvent
    | ToolCallDeltaEvent
    | ToolCallReadyEvent
    | ToolExecutionStartedEvent
    | ToolExecutionProgressEvent
    | ToolExecutionFinishedEvent
    | ToolResultEvent
    | StepFinished
    | TurnFinished
    | TurnFailed
)


class AgentTurn:
    def __init__(
        self,
        agent: "Agent[Any]",
        message: "Content",
        *,
        options: TurnOptions | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.agent_id = agent.id
        self.agent = agent
        self.options = options or TurnOptions()
        self.message = Message(role="user", content=message)
        self._controller = AbortController()
        self._events: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        self._steering: asyncio.Queue[Message] = asyncio.Queue()
        self._task: asyncio.Task[TurnResult] | None = None
        self._result: TurnResult | None = None
        self._started = False
        self._done = False
        self._started_at = 0.0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def started(self) -> bool:
        return self._started

    def abort(self, reason: str | None = None) -> None:
        self._controller.abort(reason)

    async def send(
        self,
        message: "Content",
        *,
        role: Literal["user", "system"] = "user",
    ) -> None:
        await self._steering.put(Message(role=role, content=message))

    async def steer(
        self,
        message: "Content",
        *,
        role: Literal["user", "system"] = "user",
    ) -> None:
        await self.send(message, role=role)

    async def events(self) -> AsyncIterator[TurnEvent]:
        self._ensure_started()
        while True:
            event = await self._events.get()
            if event is None:
                break
            yield event

    async def text_stream(self) -> AsyncIterator[str]:
        async for event in self.events():
            if event.type == "text.delta":
                yield event.text

    async def wait(self) -> TurnResult:
        return await self.result()

    async def result(self) -> TurnResult:
        self._ensure_started()
        if self._task is None:
            raise RuntimeError("Turn did not start")
        return await self._task

    def _ensure_started(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def _put_event(self, event: TurnEvent) -> None:
        await self._events.put(event)

    async def _run(self) -> TurnResult:
        from ai_query import stream_text
        from ai_query.core import _tool_execution_scope

        self._started = True
        self._started_at = time.time()
        full_response = ""
        persisted_step_count = 0
        observed_steps: list[StepResult] = []
        external_signal = self.options.signal
        signal = self._controller.signal

        (
            before_step_hook,
            after_step_hook,
            before_tool_call_hook,
            after_tool_call_hook,
            reasoning_handler,
        ) = self.agent._build_runtime_callbacks(
            turn=self,
            hooks=self.options.hooks,
            signal=signal,
        )

        await self._put_event(TurnStarted(
            type="turn.started",
            turn_id=self.id,
            agent_id=self.agent_id,
            message=self.message,
            created_at=self._started_at,
        ))

        self.agent.messages.append(self.message)
        await self.agent._persist_messages()

        async def on_step_start(event: StepStartEvent) -> StepControl | None:
            await self._put_event(StepStarted(type="step.started", step_number=event.step_number))
            injected: list[Message] = []
            while not self._steering.empty():
                injected.append(await self._steering.get())
            hook_control = await before_step_hook(event)
            if injected:
                self.agent.messages.extend(injected)
                await self.agent._persist_messages()
            if not hook_control and not injected:
                return None
            return StepControl(
                inject_messages=(hook_control.inject_messages if hook_control else []) + injected,
                stop=bool(hook_control.stop) if hook_control else False,
                stop_reason=hook_control.stop_reason if hook_control else None,
            )

        async def on_step_finish(event: StepFinishEvent) -> None:
            nonlocal persisted_step_count
            observed_steps.append(event.step)
            self.agent._append_step_message(event.step)
            await self.agent._persist_messages()
            persisted_step_count += 1
            await self._put_event(StepFinished(
                type="step.finished",
                step_number=event.step_number,
                step=event.step,
                usage=event.usage,
            ))
            await after_step_hook(event)

        async def before_tool_call(event: BeforeToolCallEvent):
            return await before_tool_call_hook(event)

        async def after_tool_call(event: AfterToolCallEvent):
            return await after_tool_call_hook(event)

        async def on_reasoning_event(event: ReasoningEvent) -> None:
            await reasoning_handler(event)
            await self._put_event(ReasoningDelta(type="reasoning.delta", event=event))

        async def on_retry(event: RetryEvent) -> None:
            await self._put_event(StepRetrying(
                type="step.retrying",
                step_number=event.step_number,
                attempt=event.attempt,
                max_attempts=event.max_attempts,
                delay=event.delay,
                error=event.error,
            ))

        try:
            if external_signal:
                external_signal.throw_if_aborted()
                external_signal.add_listener(lambda: self.abort(external_signal.reason))

            with _tool_execution_scope(turn_id=self.id, agent_id=self.agent_id):
                result = stream_text(
                    model=self.agent.model,
                    system=self.agent.system,
                    messages=self.agent.messages,
                    tools=self.options.tools if self.options.tools is not None else (self.agent.tools if self.agent.tools else None),
                    stop_when=self.options.stop_when if self.options.stop_when is not None else self.agent.stop_when,
                    provider_options=self.options.provider_options if self.options.provider_options is not None else self.agent.provider_options,
                    reasoning=self.options.reasoning if self.options.reasoning is not None else self.agent.reasoning,
                    retry=self.options.retry if self.options.retry is not None else self.agent.retry,
                    on_retry=on_retry,
                    signal=signal,
                    metadata=self.options.metadata,
                    on_reasoning_event=on_reasoning_event,
                    on_step_start=on_step_start,
                    on_step_finish=on_step_finish,
                    before_tool_call=before_tool_call,
                    after_tool_call=after_tool_call,
                )

            async for stream_event in result.event_stream:
                if isinstance(stream_event, TextDeltaEvent):
                    full_response += stream_event.text
                    await self._put_event(TextDelta(
                        type="text.delta",
                        text=stream_event.text,
                    ))
                elif isinstance(
                    stream_event,
                    (
                        ToolCallStartedEvent,
                        ToolCallDeltaEvent,
                        ToolCallReadyEvent,
                        ToolExecutionStartedEvent,
                        ToolExecutionProgressEvent,
                        ToolExecutionFinishedEvent,
                        ToolResultEvent,
                    ),
                ):
                    await self._put_event(stream_event)

            steps = await self.agent._get_result_steps(result) or []
            if persisted_step_count < len(steps):
                for step in steps[persisted_step_count:]:
                    self.agent._append_step_message(step)
                await self.agent._persist_messages()
                persisted_step_count = len(steps)

            output_message = self.agent.messages[-1] if self.agent.messages else Message(role="assistant", content=full_response)
            turn_result = TurnResult(
                turn_id=self.id,
                agent_id=self.agent_id,
                text=full_response,
                finish_reason=await result.finish_reason,
                usage=await result.usage,
                steps=steps,
                started_at=self._started_at,
                ended_at=time.time(),
                output_message=output_message,
                termination=await result.termination,
            )
            self._result = turn_result
            await self._put_event(TurnFinished(type="turn.finished", result=turn_result))
            return turn_result
        except AbortError as exc:
            abort_reason = exc.reason or signal.reason
            termination = exc.termination or build_turn_termination(
                "aborted",
                steps=observed_steps,
                text=full_response,
                reason=abort_reason,
                final_step_number=persisted_step_count,
                error_type=type(exc).__name__,
                message=abort_reason or "Operation aborted",
            )
            exc.termination = termination
            await self._put_event(TurnFailed(
                type="turn.failed",
                error=abort_reason or "Operation aborted",
                error_type=type(exc).__name__,
                aborted=True,
                termination=termination,
            ))
            raise
        except Exception as exc:
            termination = getattr(exc, "termination", None) or build_turn_termination(
                "failed",
                steps=observed_steps,
                text=full_response,
                final_step_number=persisted_step_count,
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )
            try:
                setattr(exc, "termination", termination)
            except Exception:
                pass
            await self._put_event(TurnFailed(
                type="turn.failed",
                error=str(exc),
                error_type=type(exc).__name__,
                termination=termination,
            ))
            raise
        finally:
            self._done = True
            await self._events.put(None)
