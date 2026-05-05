# ai-query

Python SDK for model calls, tools, and stateful agents.

## Install

```bash
pip install ai-query
```

## Docs

The full documentation lives in [`docs/`](docs/).

Start here:

- [`docs/index.mdx`](docs/index.mdx)
- [`docs/tutorials/quickstart.mdx`](docs/tutorials/quickstart.mdx)
- [`docs/reference/agent.mdx`](docs/reference/agent.mdx)
- [`docs/reference/agent-turn.mdx`](docs/reference/agent-turn.mdx)

## Quick Example

```python
from ai_query.agents import Agent
from ai_query.providers import openai

agent = Agent("assistant", model=openai("gpt-4o"))

async with agent:
    print(await agent.chat("Hello"))
```

## License

MIT
