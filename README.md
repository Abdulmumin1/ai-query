# ai-query

A unified Python SDK for querying AI models from various providers with a consistent interface.

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

### Providers

The library supports various AI providers through a unified interface. Built-in support is available for:

*   **OpenAI**: `openai("model-name")`
*   **Anthropic**: `anthropic("model-name")`
*   **Google**: `google("model-name")`

#### Configuration

You can configure providers using environment variables (recommended) or by passing credentials directly.

**Using Environment Variables:**

By default, providers look for standard environment variables:
*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`

**Passing API Keys Dynamically:**

You can explicitly pass the API key (and other parameters) when initializing a provider.

```python
from ai_query import generate_text, google

async def main():
    # Pass API key directly
    model = google("gemini-2.0-flash", api_key="your_api_key_here")

    result = await generate_text(
        model=model,
        prompt="Explain quantum computing"
    )
    print(result.text)
```

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
