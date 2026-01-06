"""Multi-Provider Comparison

Demonstrates:
- Using different providers (OpenAI, Anthropic, Google) with the same interface
- Comparing responses from different models
- No tool calling - just showing provider flexibility
"""

import asyncio
import time
from ai_query import generate_text, stream_text, openai, anthropic, google


async def compare_providers():
    """Compare responses from different AI providers."""

    prompt = "Explain the concept of recursion in programming in exactly 2 sentences."

    providers = [
        ("OpenAI GPT-4o-mini", openai("gpt-4o-mini")),
        ("Anthropic Claude Haiku", anthropic("claude-3-5-haiku-latest")),
        ("Google Gemini Flash", google("gemini-2.0-flash")),
    ]

    print("Multi-Provider Comparison")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)

    results = []

    for name, model in providers:
        print(f"\n{name}:")
        print("-" * 40)

        start = time.time()
        try:
            result = await generate_text(
                model=model,
                prompt=prompt,
                max_tokens=150,
            )
            elapsed = time.time() - start

            print(result.text)
            print(f"\n[{elapsed:.2f}s | {result.usage.total_tokens} tokens]")

            results.append({
                "name": name,
                "text": result.text,
                "tokens": result.usage.total_tokens,
                "time": elapsed,
            })

        except Exception as e:
            print(f"Error: {e}")
            print("(Make sure the API key is set)")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"{r['name']}: {r['time']:.2f}s, {r['tokens']} tokens")


async def streaming_comparison():
    """Compare streaming from different providers."""

    prompt = "Write a haiku about programming."

    providers = [
        ("OpenAI", openai("gpt-4o-mini")),
        ("Anthropic", anthropic("claude-3-5-haiku-latest")),
        ("Google", google("gemini-2.0-flash")),
    ]

    print("\n" + "=" * 60)
    print("STREAMING COMPARISON")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)

    for name, model in providers:
        print(f"\n{name} (streaming):")
        print("-" * 40)

        try:
            result = stream_text(
                model=model,
                prompt=prompt,
                max_tokens=100,
            )

            async for chunk in result.text_stream:
                print(chunk, end="", flush=True)

            usage = await result.usage
            print(f"\n[{usage.total_tokens} tokens]")

        except Exception as e:
            print(f"Error: {e}")


async def main():
    print("Note: This example requires API keys for all providers:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY")
    print()

    # Run comparisons
    await compare_providers()
    await streaming_comparison()


if __name__ == "__main__":
    asyncio.run(main())
