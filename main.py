"""Example usage of ai-query library."""

import asyncio

from ai_query import generate_text, stream_text, openai, anthropic, google


async def main():
    # Example 1: Simple prompt with OpenAI
    # result = await generate_text(
    #     model=openai("gpt-4o"),
    #     prompt="What is the capital of France?"
    # )
    # print(f"OpenAI: {result.text}")

    # Example 2: System prompt + user prompt with Anthropic
    # result = await generate_text(
    #     model=anthropic("claude-sonnet-4-20250514"),
    #     system="You are a helpful assistant. Be concise.",
    #     prompt="Explain quantum computing in one sentence."
    # )
    # print(f"Anthropic: {result.text}")

    # Example 3: Streaming with Google + usage access
    print("Streaming from Google Gemini:")
    result = stream_text(
        model=google("gemini-2.0-flash"),
        system="You are a poet.",
        messages=[
            {"role": "assistant", "content": "hi there"},
            
            {"role": "user", "content": [
                {"type": "text", "text": "give me the test peom from this"},
                {"type": "file", "data": "https://www.arvindguptatoys.com/arvindgupta/hundred-poems.pdf", "media_type": "application/pdf"}
            ]}
        ]
    )

    # Stream the text
    async for chunk in result.text_stream:
        print(chunk, end="", flush=True)
    print("\n")

    # Access usage after streaming completes
    usage = await result.usage
    if usage:
        print(f"Usage: {usage.input_tokens} input, {usage.output_tokens} output, {usage.total_tokens} total")

    finish_reason = await result.finish_reason
    print(f"Finish reason: {finish_reason}")

    # Example 4: Direct iteration (simpler, but no usage access)
    # print("\nDirect streaming:")
    # async for chunk in stream_text(
    #     model=google("gemini-2.0-flash"),
    #     prompt="Say hello in 3 languages."
    # ):
    #     print(chunk, end="", flush=True)
    # print("\n")


if __name__ == "__main__":
    asyncio.run(main())
