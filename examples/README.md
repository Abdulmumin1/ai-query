# ai-query Examples

Examples are grouped by API depth.

## Agent APIs

Use these when you want state, message history, tools, or app/runtime behavior.

| Example | Description | Features |
| --- | --- | --- |
| [multi_agent_research.py](multi_agent_research.py) | Per-user research assistants | `Agent`, tools, state, events, `AgentServer` |
| [agent_direct.py](agent_direct.py) | Custom agent runtime | `Agent`, manual streaming, custom events |
| [cloudflare_worker.py](cloudflare_worker.py) | Worker deployment | Cloudflare adapter, actions |
| [cloudflare-counter](cloudflare-counter/) | Stateful counter agent | WebSocket/SSE, state, actions |
| [rpc](rpc/) | Interactive chat room | RPC, remote agents, WebSocket/SSE |

## Core Generation APIs

Use these when you want direct model calls without agent state.

| Example | Description | Features |
| --- | --- | --- |
| [wikipedia_agent.py](wikipedia_agent.py) | Wikipedia research loop | `generate_text`, async tools, `on_step_finish` |
| [code_executor.py](code_executor.py) | Python code executor | tools, step callbacks, sandboxed execution |
| [hackernews_agent.py](hackernews_agent.py) | Hacker News summarizer | `stream_text`, tools, API calls |
| [task_planner.py](task_planner.py) | Multi-step task executor | stop conditions, task logging |
| [country_explorer.py](country_explorer.py) | Geography data explorer | REST tools, step callbacks |
| [unit_converter.py](unit_converter.py) | Unit conversion assistant | sync tools |
| [multi_provider.py](multi_provider.py) | Provider comparison | OpenAI, Anthropic, Google |
| [openai_reasoning.py](openai_reasoning.py) | Reasoning controls | reasoning config, streaming |

## Running Examples

Set the provider keys required by the example you are running:

```bash
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Run from the repository root:

```bash
uv run examples/wikipedia_agent.py
uv run examples/hackernews_agent.py
uv run examples/rpc/server.py
```

## Common Patterns

### Simple Agent

```python
from ai_query.agents import Agent, SQLiteStorage
from ai_query.providers import openai

agent = Agent(
    "assistant",
    model=openai("gpt-4o"),
    storage=SQLiteStorage("agents.sqlite3"),
)

text = await agent.chat("Hello")
```

### Structured Turn

```python
result = await agent.run("Investigate the issue")

print(result.text)
print(result.usage)
print(result.steps)
```

### Live Turn Events

```python
turn = agent.turn("Fix the failing test")

async for event in turn.events():
    if event.type == "text.delta":
        print(event.text, end="")
    elif event.type == "step.started":
        print(f"step {event.step_number}")

result = await turn.result()
```

### Mid-Run Steering

```python
turn = agent.turn("Investigate the bug")

asyncio.create_task(turn.result())

await turn.send("Change direction: inspect migrations first.")
```

### Tool Calling

```python
from ai_query import Field, tool


@tool(description="Get the current weather for a location")
async def get_weather(
    location: str = Field(description="City name"),
) -> str:
    return f"Weather in {location}: 72F, Sunny"


agent = Agent(
    "weather",
    model=model,
    tools={"get_weather": get_weather},
)

result = await agent.run("What's the weather in Tokyo?")
```

### Core Streaming

```python
result = stream_text(
    model=model,
    prompt="Summarize the top Hacker News stories",
    tools={"get_stories": get_stories},
)

async for chunk in result.text_stream:
    print(chunk, end="", flush=True)
```

### Stop Conditions

```python
from ai_query import has_tool_call, step_count_is

stop_when=step_count_is(5)
stop_when=has_tool_call("complete_task")
stop_when=[has_tool_call("done"), step_count_is(10)]
```
