import pyodide_http

pyodide_http.patch_all()

from ai_query import Agent, action
from ai_query.adapters.cloudflare import AgentDO, CloudflareRegistry
from workers import DurableObject, Response, WorkerEntrypoint

# --- 1. Define the Agent ---


class CounterAgent(Agent):
    """A minimal agent that maintains a persistent counter.

    Demonstrates:
    1. Implicit state persistence via self.state
    2. WebSocket communication via self.emit()
    3. Action handling via @action
    """

    @action
    async def increment(self):
        # 1. State Logic
        count = self.state.get("count", 0)
        count += 1
        self.state["count"] = count  # Automatically saved to DO storage

        # 2. Real-time broadcast
        # This sends {"type": "count_updated", "data": {"count": 1}} to all connected clients
        await self.emit("count_updated", {"count": count})

        return count

    async def chat(self, message, *, signal=None):
        # Dynamic System Prompt: Inject the current count so the AI knows about it
        count = self.state.get("count", 0)
        self.system = f"""You are the Keeper of the Count. 
The current count is {count}. 
You are ancient, mystical, and very protective of your numbers.
If the user asks to increment, tell them they must use the proper channels (the increment button/action), 
but you can hint at how glorious the next number will be."""

        return await super().chat(message, signal=signal)

    @action
    async def get_count(self):
        return self.state.get("count", 0)

    async def on_connect(self, connection, ctx):
        # Send current count immediately on connection
        count = self.state.get("count", 0)
        await self.emit(
            "welcome", {"message": "Connected to CounterAgent", "count": count}
        )


# --- 2. Define the Durable Object Wrapper ---


class CounterDO(AgentDO):
    """The Cloudflare Durable Object class."""

    agent_class = CounterAgent


# --- 3. Define the Main Worker Handler ---
class Default(WorkerEntrypoint):
    async def fetch(self, request):
        registry = CloudflareRegistry(self.env)

        # Route requests for `/agent/counter-1` to the COUNTER binding
        registry.register("counter-.*", self.env.COUNTER)

        return await registry.handle_request(request)
