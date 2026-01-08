"""Example of MCP (Model Context Protocol) integration with ai-query."""

import asyncio
import json
from ai_query import generate_text, google, openai, step_count_is, StepFinishEvent, mcp


def on_step_finish(event: StepFinishEvent):
    """Log the progress of each step."""
    print(f"\n--- Step {event.step_number} ---")

    if event.step.text:
        print(f"Response: {event.step.text.strip()[:300]}")

    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            args = json.dumps(tc.arguments)
            print(f"Tool Call: {tc.name}({args})")

    if event.step.tool_results:
        for tr in event.step.tool_results:
            print(f"Result: {tr.result}")


async def main():
    """
    Example using a local Python MCP server.

    The test_mcp_server.py provides these tools:
    - calculator: Perform math calculations
    - get_weather: Get weather for a city
    - echo: Echo back a message
    - random_number: Generate random numbers
    - lookup_user: Look up user profiles
    """
    print("=" * 60)
    print("MCP Integration Example - Using local test_mcp_server.py")
    print("=" * 60)

    # Connect to our local Python MCP server
    async with mcp("python", "test_mcp_server.py") as server:

        # Show discovered tools
        print(f"\nDiscovered {len(server.tools)} tool(s) from MCP server:")
        for name, tool in server.tools.items():
            print(f"  - {name}: {tool.description}")

        print("\n" + "-" * 60)
        print("Sending prompt to model with MCP tools...")
        print("-" * 60)

        result = await generate_text(
            model=google("gemini-2.0-flash"),  # or openai("gpt-4o")
            system="You are a helpful assistant. Use the available tools to answer questions.",
            prompt="What's 125 * 8? Also, what's the weather like in Tokyo? And look up user 1.",
            tools=server.tools,
            on_step_finish=on_step_finish,
            stop_when=step_count_is(5),
        )

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result.text)
        print("=" * 60)

        if result.usage:
            print(f"\nTotal tokens used: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
