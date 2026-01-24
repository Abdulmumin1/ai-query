# ai-query

A unified Python SDK for querying AI models from multiple providers with a consistent interface.

## Installation

```bash
uv add ai-query
# or
pip install ai-query
```

For MCP (Model Context Protocol) support:

```bash
uv add ai-query[mcp]
# or
pip install ai-query[mcp]
```

## Quick Start

```python
import asyncio
from ai_query import generate_text
from ai_query.providers import openai

async def main():
    result = await generate_text(
        model=openai("gpt-4o"),
        prompt="What is the capital of France?"
    )
    print(result.text)

asyncio.run(main())
```

## Streaming

```python
from ai_query import stream_text
from ai_query.providers import google

async def main():
    result = stream_text(
        model=google("gemini-2.0-flash"),
        prompt="Write a short story."
    )

    async for chunk in result.text_stream:
        print(chunk, end="", flush=True)

    usage = await result.usage
    print(f"\nTokens: {usage.total_tokens}")
```

## Stateful Agents

ai-query provides a powerful Actor-based `Agent` class for building stateful AI applications that maintain memory and identity.

```python
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
        print(response)

        response = await agent.chat("What's my name?")
        print(response) # "Your name is Alice."

asyncio.run(main())
```

## Tool Calling

Define tools using the `@tool` decorator. The library handles the execution loop automatically.

```python
from ai_query import generate_text, tool, Field
from ai_query.providers import google

@tool(description="Get the current weather for a location")
async def get_weather(
    location: str = Field(description="City name")
) -> str:
    return f"Weather in {location}: 72°F, Sunny"

async def main():
    result = await generate_text(
        model=google("gemini-2.0-flash"),
        prompt="What's the weather in Paris?",
        tools={"weather": get_weather},
    )
    print(result.text)

asyncio.run(main())
```

## MCP (Model Context Protocol) Support

Connect to any MCP server and use its tools seamlessly.

```python
from ai_query import generate_text
from ai_query.providers import google
from ai_query.mcp import mcp

async def main():
    async with mcp("npx", "-y", "@modelcontextprotocol/server-fetch") as server:
        result = await generate_text(
            model=google("gemini-2.0-flash"),
            prompt="Fetch and summarize https://example.com",
            tools=server.tools,
        )
        print(result.text)

asyncio.run(main())
```

## Modular Imports

The library is divided into clean modules:

- `ai_query`: Core generation functions (`generate_text`, `stream_text`, `embed`, `tool`, `Field`).
- `ai_query.agents`: Stateful orchestration (`Agent`, `AgentServer`, `MemoryStorage`, `SQLiteStorage`).
- `ai_query.providers`: Model gateways (`openai`, `anthropic`, `google`, `deepseek`, `groq`, etc.).
- `ai_query.mcp`: Model Context Protocol integration.
- `ai_query.types`: Data models and types.

## License

MIT
