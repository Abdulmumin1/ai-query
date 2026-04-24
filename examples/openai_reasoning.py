"""OpenAI Reasoning Example

Demonstrates:
- Using the normalized `reasoning` parameter with the OpenAI provider
- Comparing different reasoning effort levels
- Streaming a reasoning-capable response
"""

import asyncio

from ai_query import generate_text, stream_text
from ai_query.providers import openai


async def compare_effort_levels() -> None:
    """Compare low and high reasoning effort on the same task."""
    prompt = """
You have three boxes:
- One contains only apples
- One contains only oranges
- One contains both apples and oranges

All three labels are wrong.
You may pick one fruit from one box without looking inside.
Explain how to relabel all boxes correctly in the fewest steps.
""".strip()

    model = openai("gpt-5.4")

    print("OpenAI Reasoning Example")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    try:
        for effort in ["low", "high"]:
            print(f"\nReasoning effort: {effort}")
            print("-" * 40)

            result = await generate_text(
                model=model,
                prompt=prompt,
                reasoning={"effort": effort},
            )

            print(result.text)
            if result.usage:
                print(f"\n[{result.usage.total_tokens} total tokens]")
    finally:
        await model.provider.transport.close()


async def streaming_example() -> None:
    """Stream a reasoning-capable answer from OpenAI."""
    prompt = "Explain why binary search is O(log n) in 4 short bullet points."
    model = openai("gpt-5.4")

    print("\n" + "=" * 60)
    print("STREAMING")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 40)

    try:
        result = stream_text(
            model=model,
            prompt=prompt,
            reasoning={"effort": "medium"},
            max_completion_tokens=250,
        )

        async for chunk in result.text_stream:
            print(chunk, end="", flush=True)

        usage = await result.usage
        if usage:
            print(f"\n\n[{usage.total_tokens} total tokens]")
    finally:
        await model.provider.transport.close()


async def main() -> None:
    print("This example requires OPENAI_API_KEY to be set.")
    print()

    await compare_effort_levels()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
