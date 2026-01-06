# ai-query

A unified Python SDK for querying AI models from multiple providers with a consistent interface.

## Installation

```bash
uv add ai-query
# or
pip install ai-query
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

Define tools and let the model use them automatically. The library handles the execution loop.

```python
from ai_query import generate_text, google, tool, step_count_is

# Define tools
weather_tool = tool(
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    },
    execute=lambda location: {"temp": 72, "condition": "sunny"}
)

calculator_tool = tool(
    description="Perform math calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    },
    execute=lambda expression: {"result": eval(expression)}
)

async def main():
    result = await generate_text(
        model=google("gemini-2.0-flash"),
        prompt="What's the weather in Paris? Also, what is 25 * 4?",
        tools={
            "weather": weather_tool,
            "calculator": calculator_tool
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
    tools={"weather": weather_tool},
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

See the [examples/](examples/) folder for agent implementations