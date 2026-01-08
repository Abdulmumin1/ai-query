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
from ai_query import generate_text, openai

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
from ai_query import stream_text, google

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

## Tool Calling

Define tools and let the model use them automatically. The library handles the execution loop. Tools can be defined using the `@tool` decorator with type hints and the `Field` class for descriptions.

```python
from ai_query import generate_text, google, tool, Field, step_count_is

# Define tools using decorators
@tool(description="Get the current weather for a location")
async def get_weather(
    location: str = Field(description="City name")
) -> str:
    # Function implementation
    return f"Weather in {location}: 72°F, Sunny"

@tool(description="Perform math calculations")
def calculate(
    expression: str = Field(description="Math expression")
) -> str:
    return str(eval(expression))

async def main():
    result = await generate_text(
        model=google("gemini-2.0-flash"),
        prompt="What's the weather in Paris? Also, what is 25 * 4?",
        tools={
            "weather": get_weather,
            "calculator": calculate
        },
        stop_when=step_count_is(5),  # Max 5 model calls
    )
    print(result.text)
    print(f"Steps: {len(result.response['steps'])}")
```

### Stop Conditions

Control when the tool execution loop stops:

```python
from ai_query import step_count_is, has_tool_call

# Stop after N model calls
stop_when=step_count_is(5)

# Stop when a specific tool is called
stop_when=has_tool_call("final_answer")

# Multiple conditions (stops when any is true)
stop_when=[step_count_is(10), has_tool_call("done")]
```

## MCP (Model Context Protocol) Support

ai-query supports [MCP](https://modelcontextprotocol.io/) - a standard for AI tool integration. Connect to any MCP server and use its tools seamlessly.

### Transports

| Transport | Function | Use Case |
|-----------|----------|----------|
| **stdio** | `mcp()` | Local servers (python, node, npx) |
| **SSE** | `mcp_sse()` | Remote servers (legacy) |
| **Streamable HTTP** | `mcp_http()` | Remote servers (recommended) |

### Local MCP Server (stdio)

```python
from ai_query import generate_text, google, mcp

async def main():
    # Connect to a local Python MCP server
    async with mcp("python", "my_server.py") as server:
        print(f"Available tools: {list(server.tools.keys())}")

        result = await generate_text(
            model=google("gemini-2.0-flash"),
            prompt="Calculate 25 * 4",
            tools=server.tools,
        )
        print(result.text)
```

### Using npx for npm MCP packages

```python
from ai_query import generate_text, openai, mcp

async with mcp("npx", "-y", "@modelcontextprotocol/server-fetch") as server:
    result = await generate_text(
        model=openai("gpt-4o"),
        prompt="Fetch and summarize https://example.com",
        tools=server.tools,
    )
```

### Remote MCP Server (SSE)

```python
from ai_query import generate_text, openai, mcp_sse

async with mcp_sse("http://localhost:8000/sse") as server:
    result = await generate_text(
        model=openai("gpt-4o"),
        prompt="Hello!",
        tools=server.tools,
    )

# With authentication
async with mcp_sse(
    "https://api.example.com/mcp/sse",
    headers={"Authorization": "Bearer token123"}
) as server:
    ...
```

### Remote MCP Server (Streamable HTTP)

```python
from ai_query import generate_text, openai, mcp_http

async with mcp_http("http://localhost:8000/mcp") as server:
    result = await generate_text(
        model=openai("gpt-4o"),
        prompt="Hello!",
        tools=server.tools,
    )
```

### Combining Multiple Tool Sources

Use `merge_tools` to combine tools from multiple MCP servers or mix with local tools:

```python
from ai_query import generate_text, openai, mcp, merge_tools, tool, Field

@tool(description="Calculate math expressions")
def calculate(expr: str = Field(description="Expression")) -> str:
    return str(eval(expr))

async with mcp("python", "weather_server.py") as weather:
    async with mcp("python", "search_server.py") as search:
        all_tools = merge_tools(
            {"calculator": calculate},  # Local tool
            weather,                      # MCP server
            search,                       # Another MCP server
        )

        result = await generate_text(
            model=openai("gpt-4o"),
            prompt="What's the weather in Tokyo, search for news, and calculate 100/4",
            tools=all_tools,
        )
```

### Manual Connection Management

For long-lived connections, use `connect_mcp`, `connect_mcp_sse`, or `connect_mcp_http`:

```python
from ai_query import connect_mcp

server = await connect_mcp("python", "server.py")
try:
    # Use server.tools for multiple requests...
    result = await generate_text(...)
finally:
    await server.close()
```

## Step Callbacks

Monitor and react to each step in the execution loop with `on_step_start` and `on_step_finish`.

```python
from ai_query import generate_text, google, StepStartEvent, StepFinishEvent

def on_start(event: StepStartEvent):
    print(f"Step {event.step_number} starting...")
    print(f"  Messages: {len(event.messages)}")
    # event.messages can be modified before the model call

def on_finish(event: StepFinishEvent):
    print(f"Step {event.step_number} finished")

    # Current step details
    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            print(f"  Called: {tc.name}({tc.arguments})")
    if event.step.tool_results:
        for tr in event.step.tool_results:
            print(f"  Result: {tr.result}")

    # Accumulated state
    print(f"  Total tokens: {event.usage.total_tokens}")
    print(f"  Text so far: {event.text[:50]}...")

result = await generate_text(
    model=google("gemini-2.0-flash"),
    prompt="What's the weather in Tokyo?",
    tools={"weather": get_weather},
    on_step_start=on_start,
    on_step_finish=on_finish,
)
```

### StepStartEvent

| Field | Type | Description |
|-------|------|-------------|
| `step_number` | `int` | 1-indexed step number |
| `messages` | `list[Message]` | Conversation history (modifiable) |
| `tools` | `ToolSet \| None` | Available tools |

### StepFinishEvent

| Field | Type | Description |
|-------|------|-------------|
| `step_number` | `int` | 1-indexed step number |
| `step` | `StepResult` | Current step (text, tool_calls, tool_results) |
| `text` | `str` | Accumulated text from all steps |
| `usage` | `Usage` | Accumulated token usage |
| `steps` | `list[StepResult]` | All completed steps |

Both callbacks support sync and async functions.

## Providers

Built-in support for:

- **OpenAI**: `openai("gpt-4o")` - uses `OPENAI_API_KEY`
- **Anthropic**: `anthropic("claude-sonnet-4-20250514")` - uses `ANTHROPIC_API_KEY`
- **Google**: `google("gemini-2.0-flash")` - uses `GOOGLE_API_KEY`

Pass API keys directly if needed:

```python
model = google("gemini-2.0-flash", api_key="your_key")
```

## Provider Options

Pass provider-specific parameters:

```python
result = await generate_text(
    model=google("gemini-2.0-flash"),
    prompt="Tell me a story",
    provider_options={
        "google": {
            "safety_settings": {"HARM_CATEGORY_VIOLENCE": "BLOCK_NONE"}
        }
    }
)
```

## Examples

See the [examples/](examples/) folder for agent implementations.
