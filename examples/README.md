# ai-query Examples

Real-world agent examples demonstrating different features of ai-query.

## Examples

| Example | Description | Features |
|---------|-------------|----------|
| [wikipedia_agent.py](wikipedia_agent.py) | Research agent using Wikipedia API | Async tools, multiple tools, `on_step_finish` |
| [code_executor.py](code_executor.py) | Agent that executes Python code | Sync tools, sandboxed execution, `on_step_start` |
| [hackernews_agent.py](hackernews_agent.py) | Hacker News reader and summarizer | `stream_text` with tools, multiple API calls |
| [task_planner.py](task_planner.py) | Multi-step task executor | `has_tool_call` stop condition, task logging |
| [multi_provider.py](multi_provider.py) | Compare OpenAI, Anthropic, Google | Multiple providers, streaming comparison |
| [country_explorer.py](country_explorer.py) | Geography data explorer | REST API integration, `on_step_start` |
| [unit_converter.py](unit_converter.py) | Unit conversion assistant | Simple sync tools, multiple conversions |

## Running Examples

Make sure you have the required API keys set:

```bash
export GOOGLE_API_KEY="your-key"      # Required for most examples
export OPENAI_API_KEY="your-key"      # Required for multi_provider.py
export ANTHROPIC_API_KEY="your-key"   # Required for multi_provider.py
```

Run any example:

```bash
cd examples
uv run wikipedia_agent.py
uv run hackernews_agent.py
uv run task_planner.py
```

## Feature Highlights

### Tool Calling with Execution Loop
```python
@tool(description="Get the current weather for a location")
async def get_weather(
    location: str = Field(description="City name")
) -> str:
    return f"Weather in {location}: 72°F, Sunny"

result = await generate_text(
    model=google("gemini-2.0-flash"),
    prompt="What's the weather in Tokyo?",
    tools={"get_weather": get_weather},
    stop_when=step_count_is(5),
)
```

### Streaming with Tools
```python
@tool(description="Get top stories from Hacker News")
async def get_stories(limit: int = Field(default=5)) -> str:
    # ... implementation ...
    return "Top stories: ..."

result = stream_text(
    model=google("gemini-2.0-flash"),
    prompt="Summarize the top Hacker News stories",
    tools={"get_stories": get_stories},
)
async for chunk in result.text_stream:
    print(chunk, end="", flush=True)
```

### Step Callbacks
```python
def on_finish(event: StepFinishEvent):
    print(f"Step {event.step_number}: {len(event.step.tool_calls)} tool calls")
    print(f"Tokens so far: {event.usage.total_tokens}")

result = await generate_text(
    model=model,
    prompt=prompt,
    tools=tools,
    on_step_finish=on_finish,
)
```

### Stop Conditions
```python
# Stop when a specific tool is called
stop_when=has_tool_call("complete_task")

# Stop after N steps
stop_when=step_count_is(5)

# Multiple conditions (stops when any is true)
stop_when=[has_tool_call("done"), step_count_is(10)]
```
