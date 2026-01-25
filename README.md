# ai-query

**The framework for building stateful, distributed AI agents.**

ai-query is a unified Python SDK that transforms AI models into stateful Actors. It provides a robust foundation for building agents that maintain memory, persist identity, and communicate via type-safe RPC.

## Key Features

- **Actor Model**: Sequential message processing to prevent race conditions.
- **Durable Identity**: Native support for SQLite, Redis, and Memory storage.
- **Durable Event Log**: Persist every event and replay automatically on reconnection.
- **Type-Safe RPC**: Call other agents fluently with full IDE autocompletion.
- **Unified Providers**: One interface for OpenAI, Anthropic, Google, DeepSeek, and more.
- **MCP Native**: Seamlessly use tools from any Model Context Protocol server.

## Installation

```bash
pip install ai-query
# with MCP support
pip install "ai-query[mcp]"
```

## Quick Start: The Stateful Agent

Create an agent that remembers context and persists history automatically.

```python
import asyncio
from ai_query.agents import Agent, SQLiteStorage
from ai_query.providers import openai

async def main():
    # Persistent agent with SQLite storage
    agent = Agent(
        "my-assistant",
        model=openai("gpt-4o"),
        storage=SQLiteStorage("agents.db")
    )

    async with agent:
        # Agent remembers conversation history automatically
        response = await agent.chat("Hi, I'm Alice!")
        print(response) # "Hello Alice! How can I help you today?"

        response = await agent.chat("What's my name?")
        print(response) # "Your name is Alice."

asyncio.run(main())
```

## Multi-User Routing

Host thousands of independent agent instances on a single server with automatic routing.

```python
from ai_query.agents import Agent, AgentServer
from ai_query.providers import google

class UserAssistant(Agent):
    def __init__(self, id):
        super().__init__(
            id,
            model=google("gemini-2.0-flash"),
            system="You are a personal assistant."
        )

# Start server - routes to /agent/{id}/ws and /agent/{id}/chat automatically
AgentServer(UserAssistant).serve(port=8080)
```

## Type-Safe RPC

Agents can expose structured **Actions** and call each other fluently.

```python
from ai_query.agents import Agent, action

class Researcher(Agent):
    @action
    async def get_summary(self, topic: str):
        return await self.chat(f"Summarize {topic}")

class Manager(Agent):
    async def handle_request(self, topic: str):
        # Call another agent with full type safety and autocompletion
        researcher = self.call("research-bot", agent_cls=Researcher)
        summary = await researcher.get_summary(topic=topic)
        return summary
```

## Real-time Events

Send custom feedback or status updates to connected clients using `emit`.

```python
class ResearchAgent(Agent):
    async def on_message(self, conn, msg):
        await self.emit("status", {"text": "Searching web..."})
        # ... logic ...
        await self.emit("status", {"text": "Synthesizing results..."})
```

## Durability & Replay

Enable the `enable_event_log` flag to persist every event. If a client disconnects, they can reconnect with their `last_event_id` and the agent will automatically replay missed events.

```python
class MyAgent(Agent):
    enable_event_log = True  # Persists events for automatic replay
    
    async def on_start(self):
        await self.emit("ready", {"timestamp": "..."})
```

## Core Generation

If you don't need state, use the core functions directly for one-off tasks.

```python
from ai_query import generate_text, stream_text
from ai_query.providers import anthropic

# Complete response
result = await generate_text(
    model=anthropic("claude-3-5-sonnet-latest"),
    prompt="Write a poem about agents."
)

# Real-time streaming
result = stream_text(
    model=anthropic("claude-3-5-sonnet-latest"),
    prompt="Explain quantum physics."
)
async for chunk in result.text_stream:
    print(chunk, end="", flush=True)
```

## Modular Imports

The library is strictly divided for a clean developer experience:

- `ai_query`: Core generation (`generate_text`, `stream_text`, `embed`).
- `ai_query.agents`: Stateful orchestration (`Agent`, `AgentServer`, `Storage`).
- `ai_query.providers`: Model gateways (`openai`, `anthropic`, `google`, etc.).
- `ai_query.mcp`: Model Context Protocol integration.

## License

MIT
