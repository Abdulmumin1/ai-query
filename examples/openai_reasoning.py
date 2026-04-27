"""OpenAI Reasoning Example

Demonstrates:
- Using the normalized `reasoning` parameter with the OpenAI provider
- Confirming OpenAI reasoning + tools works through the Responses API path
- Comparing different reasoning effort levels
- Streaming a reasoning-capable response
- Why OpenAI streaming may not emit reasoning events with this provider
- Receiving reasoning/thinking events from OpenAI-compatible providers when they
  stream fields like `reasoning_content`, `reasoning`, or `thinking`
- Receiving Gemini thought parts as separate reasoning events
"""

import asyncio

from ai_query import Field, generate_text, step_count_is, stream_text, tool
from ai_query.providers import google, openai


@tool(description="Add two integers exactly")
def add_numbers(
    a: int = Field(description="First integer"),
    b: int = Field(description="Second integer"),
) -> int:
    return a + b


async def openai_reasoning_with_tools_smoke_test() -> None:
    """Confirm OpenAI reasoning + tools works.

    This exercises the `/v1/responses` fallback because OpenAI rejects
    `reasoning_effort` with function tools on `/v1/chat/completions` for GPT-5
    reasoning models.
    """
    model = openai("gpt-5.4")

    print("OpenAI Reasoning + Tools Smoke Test")
    print("=" * 60)

    try:
        result = await generate_text(
            model=model,
            system="You must use the add_numbers tool before answering.",
            prompt="Use the tool to add 17 and 25, then answer with only the sum.",
            tools={"add_numbers": add_numbers},
            reasoning={"effort": "high"},
            stop_when=step_count_is(5),
            max_tokens=500,
        )

        if not result.tool_calls:
            raise AssertionError("Expected OpenAI to call add_numbers, but no tool call was returned")

        print(result.text)
        print(f"\nPASS: reasoning + tools worked ({len(result.tool_calls)} tool call(s))")
        if result.usage:
            print(f"[{result.usage.total_tokens} total tokens]")
    finally:
        await model.provider.transport.close()


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

    await openai_reasoning_with_tools_smoke_test()
    # await compare_effort_levels()
    # await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
