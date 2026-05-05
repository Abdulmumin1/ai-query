from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal

from ai_query.types import (
    AbortController,
    AbortError,
    AbortSignal,
    Message,
    ReasoningConfig,
    ReasoningEvent,
    StepControl,
    StepFinishEvent,
    StepResult,
    StepStartEvent,
    StopCondition,
    ToolSet,
    Usage,
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
    signal: AbortSignal | None = None
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
class TurnFinished:
    type: Literal["turn.finished"]
    result: TurnResult


@dataclass
class TurnFailed:
    type: Literal["turn.failed"]
    error: str


TurnEvent = TurnStarted | TextDelta | ReasoningDelta | StepStarted | StepFinished | TurnFinished | TurnFailed


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

        self._started = True
        self._started_at = time.time()
        full_response = ""
        external_signal = self.options.signal
        signal = self._controller.signal

        await self._put_event(TurnStarted(
            type="turn.started",
            turn_id=self.id,
            agent_id=self.agent_id,
            message=self.message,
            created_at=self._started_at,
        ))

        self.agent.messages.append(self.message)

        async def on_step_start(event: StepStartEvent) -> StepControl | None:
            await self._put_event(StepStarted(type="step.started", step_number=event.step_number))
            injected: list[Message] = []
            while not self._steering.empty():
                injected.append(await self._steering.get())
            if injected:
                return StepControl(inject_messages=injected)
            return None

        async def on_step_finish(event: StepFinishEvent) -> None:
            await self._put_event(StepFinished(
                type="step.finished",
                step_number=event.step_number,
                step=event.step,
                usage=event.usage,
            ))

        async def on_reasoning_event(event: ReasoningEvent) -> None:
            await self.agent._handle_reasoning_event(event)
            await self._put_event(ReasoningDelta(type="reasoning.delta", event=event))

        try:
            if external_signal:
                external_signal.throw_if_aborted()
                external_signal.add_listener(lambda: self.abort(external_signal.reason))

            result = stream_text(
                model=self.agent.model,
                system=self.agent.system,
                messages=self.agent.messages,
                tools=self.options.tools if self.options.tools is not None else (self.agent.tools if self.agent.tools else None),
                stop_when=self.options.stop_when if self.options.stop_when is not None else self.agent.stop_when,
                provider_options=self.options.provider_options if self.options.provider_options is not None else self.agent.provider_options,
                reasoning=self.options.reasoning if self.options.reasoning is not None else self.agent.reasoning,
                signal=signal,
                on_reasoning_event=on_reasoning_event,
                on_step_start=on_step_start,
                on_step_finish=on_step_finish,
            )

            async for chunk in result.text_stream:
                full_response += chunk
                await self._put_event(TextDelta(type="text.delta", text=chunk))

            steps = await self.agent._get_result_steps(result) or []
            self.agent._append_step_messages(steps, full_response)
            await self.agent._persist_messages()

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
            )
            self._result = turn_result
            await self._put_event(TurnFinished(type="turn.finished", result=turn_result))
            return turn_result
        except AbortError:
            if self.agent.messages and self.agent.messages[-1] is self.message:
                self.agent.messages.pop()
            await self._put_event(TurnFailed(type="turn.failed", error=signal.reason or "Operation aborted"))
            raise
        except Exception as exc:
            await self._put_event(TurnFailed(type="turn.failed", error=str(exc)))
            raise
        finally:
            self._done = True
            await self._events.put(None)
