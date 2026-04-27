"""OpenAI Reasoning Example

Demonstrates:
- Using the normalized `reasoning` parameter with the OpenAI provider
- Comparing different reasoning effort levels
- Streaming a reasoning-capable response
- Why OpenAI streaming may not emit reasoning events with this provider
- Receiving reasoning/thinking events from OpenAI-compatible providers when they
  stream fields like `reasoning_content`, `reasoning`, or `thinking`
- Receiving Gemini thought parts as separate reasoning events
"""

import asyncio

from ai_query import generate_text, stream_text
from ai_query.providers import google, openai


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

    model = openai("gpt-5.4-mini")

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
    """Stream a Gemini answer and print reasoning events if available.

    Gemini can return thought text as parts marked with `thought: true`. ai-query
    forwards those parts through on_reasoning_event instead of text_stream.
    """
    prompt = "Explain why binary search is O(log n) in 4 short bullet points."
    model = google("gemini-3.1-pro-preview")

    print("\n" + "=" * 60)
    print("GOOGLE THINKING STREAM")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 40)

    reasoning_events: list = []

    def on_reasoning(ev):
        reasoning_events.append(ev)
        print(ev.text or ev.data, end="", flush=True)

    try:
        result = stream_text(
            model=model,
            prompt=prompt,
            provider_options={
                "google": {
                    "thinking_config": {
                        "thinking_budget": 2000,
                        "include_thoughts": True,
                    }
                }
            },
            on_reasoning_event=on_reasoning,
        )

        async for chunk in result.text_stream:
            print(chunk, end="", flush=True)

        if not reasoning_events:
            print(
                "\n(no reasoning fields were present in this stream; "
                "text streaming still worked)"
            )

        usage = await result.usage
        if usage:
            print(f"\n\n[{usage.total_tokens} total tokens]")
    finally:
        await model.provider.transport.close()


async def main() -> None:
    print("This example requires OPENAI_API_KEY for the OpenAI calls.")
    print("Set GOOGLE_API_KEY for the Google thinking stream.")
    print()

    # await compare_effort_levels()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
