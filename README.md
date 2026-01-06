# ai-query

A unified Python SDK for querying AI models from multiple providers (OpenAI, Anthropic, Google) with a consistent interface.

## Installation

Install using `uv` or `pip`:

```bash
# Using uv
uv add ai-query

# Using pip
pip install ai-query
```

## Usage

### Basic Text Generation

The library provides a simple `generate_text` function for non-streaming requests.

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

### Streaming Text

For real-time responses, use `stream_text`. It returns a `TextStreamResult` that can be iterated directly.

```python
import asyncio
from ai_query import stream_text, google

async def main():
    result = stream_text(
        model=google("gemini-2.0-flash"),
        prompt="Write a short story about a robot."
    )

    # Stream the text chunks
    async for chunk in result.text_stream:
        print(chunk, end="", flush=True)

    # Access usage statistics after streaming finishes
    usage = await result.usage
    print(f"\nTotal tokens: {usage.total_tokens}")

asyncio.run(main())
```

### Multi-modal Messages

You can pass complex message structures, including files and images.

```python
import asyncio
from ai_query import stream_text, google

async def main():
    result = stream_text(
        model=google("gemini-2.0-flash"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this PDF:"},
                    {
                        "type": "file",
                        "data": "https://example.com/document.pdf",
                        "media_type": "application/pdf"
                    }
                ]
            }
        ]
    )
    async for chunk in result.text_stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Supported Providers

The library exports factory functions for popular providers:

- `openai("model-name")` (Requires `OPENAI_API_KEY`)
- `anthropic("model-name")` (Requires `ANTHROPIC_API_KEY`)
- `google("model-name")` (Requires `GOOGLE_GENERATIVE_AI_API_KEY`)

### Provider-Specific Options

You can pass specific parameters to the underlying provider using `provider_options`.

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
