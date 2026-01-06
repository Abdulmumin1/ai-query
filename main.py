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

    # Example 3: Full messages with Google
    # result = await generate_text(
    #     model=google("gemini-2.0-flash"),
    #     messages=[
    #         {"role": "system", "content": "You are a poet."},
    #         {"role": "user", "content": "Write a haiku about Python."}
    #     ]
    # )
    # print(f"Google: {result.text}")

    # Example 4: Streaming with Google
    print("Streaming from Google Gemini:")
    async for chunk in stream_text(
        model=google("gemini-2.0-flash"),
        system="You are a poet.",
        prompt="Write a 1000 words poem about finding purpose"
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # Example 5: Streaming with OpenAI
    # print("Streaming from OpenAI:")
    # async for chunk in stream_text(
    #     model=openai("gpt-4o"),
    #     prompt="Write a short story about a robot."
    # ):
    #     print(chunk, end="", flush=True)
    # print("\n")

    # Example 6: Streaming with Anthropic
    # print("Streaming from Anthropic:")
    # async for chunk in stream_text(
    #     model=anthropic("claude-sonnet-4-20250514"),
    #     prompt="Explain the meaning of life briefly."
    # ):
    #     print(chunk, end="", flush=True)
    # print("\n")


if __name__ == "__main__":
    asyncio.run(main())
