"""Example usage of ai-query library."""

import asyncio

from ai_query import generate_text, stream_text, openai, anthropic, google, tool, step_count_is, has_tool_call, StepStartEvent, StepFinishEvent


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

    # Example 3: Tool calling with execution loop
    print("Tool calling example:")

    # Define a weather tool
    weather_tool = tool(
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris' or 'New York'"
                }
            },
            "required": ["location"]
        },
        execute=lambda location: {
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45
        }
    )

    # Define a calculator tool
    calculator_tool = tool(
        description="Perform basic math calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '2 + 2'"
                }
            },
            "required": ["expression"]
        },
        execute=lambda expression: {"result": eval(expression)}
    )


    def on_start(event: StepStartEvent):
        
      print(f"Step {event.step_number} starting with {event.messages} messages")

    def on_finish(event: StepFinishEvent):
      print(f"Step {event.step_number} finished")
      if event.step.tool_calls:
          for tc in event.step.tool_calls:
              print(f"  Tool called: {tc.name}({tc.arguments})")
      if event.step.tool_results:
          for tr in event.step.tool_results:
              print(f"  Tool result: {tr.result}")
      print(f"  Usage so far: {event.usage.total_tokens} tokens")
      

    result = await generate_text(
        model=google("gemini-3-pro-preview"),
        prompt="What's the weather in Paris?",
        tools={
            "weather": weather_tool,
            "calculator": calculator_tool
        },
        on_step_finish=on_finish,
        on_step_start=on_start,
        stop_when=step_count_is(3),  # Max 3 iterations
    )

    print(f"Result: {result.text}")
    print(f"Steps: {len(result.response.get('steps', []))}")
    if result.usage:
        print(f"Total usage: {result.usage.total_tokens} tokens")

    # Example 4: Streaming with Google + usage access
    # print("Streaming from Google Gemini:")
    # result = stream_text(
    #     model=google("gemini-2.0-flash"),
    #     system="You are a poet.",
    #     messages=[
    #         {"role": "assistant", "content": "hi there"},
    #         {"role": "user", "content": [
    #             {"type": "text", "text": "give me the test poem from this"},
    #             {"type": "file", "data": "https://example.com/poem.pdf", "media_type": "application/pdf"}
    #         ]}
    #     ]
    # )
    #
    # async for chunk in result.text_stream:
    #     print(chunk, end="", flush=True)
    # print("\n")
    #
    # usage = await result.usage
    # if usage:
    #     print(f"Usage: {usage.input_tokens} input, {usage.output_tokens} output")


if __name__ == "__main__":
    asyncio.run(main())
