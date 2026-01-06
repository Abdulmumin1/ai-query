"""Example usage of ai-query library."""

import asyncio

from ai_query import generate_text


async def main():
    # Example 1: Simple prompt with OpenAI
    # result = await generate_text(
    #     model="openai/gpt-4",
    #     prompt="What is the capital of France?"
    # )
    # print(f"OpenAI: {result.text}")

    # Example 2: System prompt + user prompt with Anthropic
    # result = await generate_text(
    #     model="anthropic/claude-sonnet-4-20250514",
    #     system="You are a helpful assistant. Be concise.",
    #     prompt="Explain quantum computing in one sentence."
    # )
    # print(f"Anthropic: {result.text}")

    # Example 3: Full messages with Google
    # result = await generate_text(
    #     model="google/gemini-2.0-flash",
    #     messages=[
    #         {"role": "system", "content": "You are a poet."},
    #         {"role": "user", "content": "Write a haiku about Python."}
    #     ]
    # )
    # print(f"Google: {result.text}")

    # Example 4: With provider-specific options
    # result = await generate_text(
    #     model="google/gemini-2.0-flash",
    #     prompt="Tell me a story",
    #     max_tokens=100,
    #     temperature=0.7,
    #     provider_options={
    #         "google": {
    #             "safety_settings": {
    #                 "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"
    #             }
    #         }
    #     }
    # )
    # print(f"Google with options: {result.text}")

    print("Uncomment examples above and set API keys to test!")
    print()
    print("Usage:")
    print('  export OPENAI_API_KEY="your-key"')
    print('  export ANTHROPIC_API_KEY="your-key"')
    print('  export GOOGLE_API_KEY="your-key"')


if __name__ == "__main__":
    asyncio.run(main())
