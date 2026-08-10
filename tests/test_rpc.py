from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from ai_query import AbortController, AbortError
from ai_query.agents import Agent, Event, action, AgentServer, MemoryStorage
from ai_query.agents.agent import AgentCallProxy
from ai_query.agents.transport.base import AgentTransport


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


@pytest.mark.asyncio
async def test_local_agent_call_creates_target_and_routes_through_mailbox():
    calls: list[tuple[str, str]] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self, prompt: str) -> dict[str, str]:
            calls.append((self.id, prompt))
            return {"agent_id": self.id, "response": prompt.upper()}

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()

    result = await parent.call(
        "child",
        agent_cls=DelegatedAgent,
        timeout=None,
    ).delegate(prompt="investigate")

    assert result == {"agent_id": "child", "response": "INVESTIGATE"}
    assert calls == [("child", "investigate")]
    assert server.list_agents() == ["parent", "child"]
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_local_agent_call_observes_events_in_order_before_returning():
    first_observed = asyncio.Event()
    release_observer = asyncio.Event()
    observed: list[Event] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> str:
            await self.emit("work.started", {"step": 1})
            await self.emit("work.finished", {"step": 2})
            return "done"

    async def on_event(event: Event) -> None:
        observed.append(event)
        if event.type == "work.started":
            first_observed.set()
            await release_observer.wait()

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=None,
            on_event=on_event,
        ).delegate()
    )

    await first_observed.wait()
    assert not call.done()
    release_observer.set()

    assert await call == "done"
    assert [(event.id, event.type, event.data) for event in observed] == [
        (1, "work.started", {"step": 1}),
        (2, "work.finished", {"step": 2}),
    ]

    child = server.get_or_create("child")
    await child.emit("outside.call", {"step": 3})
    assert [event.type for event in observed] == ["work.started", "work.finished"]
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_local_agent_call_event_handler_failure_does_not_mask_result():
    observed: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> str:
            await self.emit("work.started", {})
            await self.emit("work.finished", {})
            return "primary result"

    async def on_event(event: Event) -> None:
        observed.append(event.type)
        if event.type == "work.started":
            raise RuntimeError("observer failed")

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()

    result = await parent.call(
        "child",
        agent_cls=DelegatedAgent,
        timeout=None,
        on_event=on_event,
    ).delegate()

    assert result == "primary result"
    assert observed == ["work.started", "work.finished"]
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_timed_out_local_call_deactivates_its_event_handler():
    observer_started = asyncio.Event()
    observer_finished = asyncio.Event()
    action_finished = asyncio.Event()
    observed: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> None:
            await self.emit("work.started", {})
            action_finished.set()

    async def on_event(event: Event) -> None:
        observed.append(event.type)
        observer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            observer_finished.set()

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()

    with pytest.raises(asyncio.TimeoutError):
        await parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=0.01,
            on_event=on_event,
        ).delegate()

    await observer_started.wait()
    await observer_finished.wait()
    await action_finished.wait()
    assert observed == ["work.started"]
    assert await parent.call(
        "child",
        agent_cls=DelegatedAgent,
        timeout=1.0,
    ).delegate() is None
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_abort_signal_bounds_an_uncooperative_local_call():
    started = asyncio.Event()
    observed: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> None:
            await self.emit("work.started", {})
            started.set()
            await asyncio.Event().wait()

    async def on_event(event: Event) -> None:
        observed.append(event.type)

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    controller = AbortController()
    call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=None,
            signal=controller.signal,
            on_event=on_event,
        ).delegate()
    )

    await started.wait()
    controller.abort("parent stopped")
    with pytest.raises(asyncio.TimeoutError, match="parent stopped"):
        await asyncio.wait_for(call, timeout=1.5)

    assert observed == ["work.started"]
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_abort_signal_cannot_be_overridden_by_late_target_success():
    started = asyncio.Event()

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> str:
            started.set()
            await asyncio.sleep(0.05)
            return "late success"

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    controller = AbortController()
    call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=None,
            signal=controller.signal,
        ).delegate()
    )

    await started.wait()
    controller.abort("parent stopped")
    with pytest.raises(asyncio.TimeoutError, match="parent stopped"):
        await call

    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_detached_task_keeps_its_originating_call_context():
    release_detached = asyncio.Event()
    detached_finished = asyncio.Event()
    release_second_call = asyncio.Event()
    observed_first: list[str] = []
    observed_second: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def start_detached(self) -> None:
            async def detached() -> None:
                await release_detached.wait()
                await self.emit("first.detached", {})
                detached_finished.set()

            asyncio.create_task(detached())

        @action
        async def run_second(self) -> None:
            await self.emit("second.started", {})
            await release_second_call.wait()

    async def observe_first(event: Event) -> None:
        observed_first.append(event.type)

    async def observe_second(event: Event) -> None:
        observed_second.append(event.type)

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    assert await parent.call(
        "child",
        agent_cls=DelegatedAgent,
        timeout=None,
        on_event=observe_first,
    ).start_detached() is None

    second_call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=None,
            on_event=observe_second,
        ).run_second()
    )
    while observed_second != ["second.started"]:
        await asyncio.sleep(0)
    release_detached.set()
    await detached_finished.wait()
    release_second_call.set()
    assert await second_call is None

    assert observed_first == []
    assert observed_second == ["second.started"]
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_immediate_detached_event_is_closed_at_target_action_boundary(
    caplog: pytest.LogCaptureFixture,
):
    detached_finished = asyncio.Event()
    observed: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def start_detached(self) -> str:
            async def detached() -> None:
                await self.emit("detached.immediate", {})
                detached_finished.set()

            asyncio.create_task(detached())
            return "done"

    async def on_event(event: Event) -> None:
        observed.append(event.type)

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    await parent.start()

    result = await parent.call(
        "child",
        agent_cls=DelegatedAgent,
        timeout=None,
        on_event=on_event,
    ).start_detached()
    await detached_finished.wait()

    assert result == "done"
    assert observed == []
    assert "Agent call event handler was cancelled" not in caplog.text
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_reentrant_target_emit_is_rejected_without_deadlocking(
    caplog: pytest.LogCaptureFixture,
):
    observed: list[str] = []

    class DelegatedAgent(Agent):
        def __init__(self, id: str):
            super().__init__(id, storage=MemoryStorage())

        @action
        async def delegate(self) -> str:
            await self.emit("work.started", {})
            return "done"

    server = AgentServer(DelegatedAgent)
    parent = server.get_or_create("parent")
    child = server.get_or_create("child")
    await parent.start()

    async def on_event(event: Event) -> None:
        observed.append(event.type)
        await child.emit("observer.reentered", {})

    result = await asyncio.wait_for(
        parent.call(
            "child",
            agent_cls=DelegatedAgent,
            timeout=None,
            on_event=on_event,
        ).delegate(),
        timeout=1.0,
    )

    assert result == "done"
    assert observed == ["work.started"]
    assert "cannot re-enter emit()" in caplog.text
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_legacy_transport_rejects_observation_without_raw_type_error():
    class LegacyTransport(AgentTransport):
        async def invoke(
            self,
            agent_id: str,
            payload: dict[str, Any],
            timeout: float | None = 30.0,
            signal=None,
        ) -> dict[str, Any]:
            return {"result": "unexpected"}

    async def on_event(_event: Event) -> None:
        pass

    proxy = AgentCallProxy(
        LegacyTransport(),
        "child",
        on_event=on_event,
    )
    with pytest.raises(
        NotImplementedError,
        match="LegacyTransport does not support observing invoke events",
    ):
        await proxy.delegate()


@pytest.mark.asyncio
async def test_local_agent_call_propagates_abort_signal_to_target_context():
    started = asyncio.Event()

    class CancellableAgent(Agent):
        def __init__(self, id: str):
            super().__init__(
                id,
                storage=MemoryStorage(),
                initial_state={"status": "idle"},
            )

        @action
        async def delegate(self) -> None:
            signal = self.context.signal
            assert signal is not None
            await self.update_state(status="running")
            started.set()
            try:
                await signal.wait()
                signal.throw_if_aborted()
            except AbortError:
                await self.update_state(status="aborted")
                raise

    server = AgentServer(CancellableAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    controller = AbortController()
    call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=CancellableAgent,
            timeout=1.0,
            signal=controller.signal,
        ).delegate()
    )

    await started.wait()
    controller.abort("parent turn aborted")

    with pytest.raises(RuntimeError, match="parent turn aborted"):
        await call
    assert server.get_or_create("child").state["status"] == "aborted"
    await server.evict("parent")
    await server.evict("child")


@pytest.mark.asyncio
async def test_cancelled_local_call_waits_for_target_abort_cleanup():
    started = asyncio.Event()
    observed: list[str] = []

    class CancellableAgent(Agent):
        def __init__(self, id: str):
            super().__init__(
                id,
                storage=MemoryStorage(),
                initial_state={"status": "idle"},
            )

        @action
        async def delegate(self) -> None:
            signal = self.context.signal
            assert signal is not None
            await self.update_state(status="running")
            await self.emit("delegate.started", {})
            started.set()
            try:
                await signal.wait()
                signal.throw_if_aborted()
            except AbortError:
                await self.update_state(status="aborted")
                await self.emit("delegate.aborted", {})
                raise

    async def on_event(event: Event) -> None:
        if event.type.startswith("delegate."):
            observed.append(event.type)

    server = AgentServer(CancellableAgent)
    parent = server.get_or_create("parent")
    await parent.start()
    controller = AbortController()
    call = asyncio.create_task(
        parent.call(
            "child",
            agent_cls=CancellableAgent,
            timeout=1.0,
            signal=controller.signal,
            on_event=on_event,
        ).delegate()
    )

    await started.wait()
    controller.abort("parent turn aborted")
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call
    child = server.get_or_create("child")
    assert child.state["status"] == "aborted"
    assert observed == ["delegate.started", "delegate.aborted"]
    await child.emit("delegate.outside", {})
    assert observed == ["delegate.started", "delegate.aborted"]
    await server.evict("parent")
    await server.evict("child")
