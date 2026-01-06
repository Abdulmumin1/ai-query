"""Task Planner Agent with has_tool_call Stop Condition

Demonstrates:
- Using has_tool_call("complete_task") to stop the loop
- Agent that plans and executes multi-step tasks
- Tool for marking task completion
"""

import asyncio
import aiohttp
from datetime import datetime
from ai_query import generate_text, google, tool, has_tool_call, step_count_is, StepFinishEvent


# --- Task State ---

task_log = []


# --- Tools ---

def add_note(content: str):
    """Add a note to the task log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {content}"
    task_log.append(entry)
    print(f"  [Note] {entry}")
    return f"Note added: {content}"


async def fetch_data(url: str):
    """Fetch data from a URL."""
    print(f"  [Fetch] {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return f"Error: HTTP {resp.status}"
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    data = await resp.json()
                    return f"JSON Response: {str(data)[:500]}"
                else:
                    text = await resp.text()
                    return f"Text Response: {text[:500]}"
    except Exception as e:
        return f"Error fetching URL: {e}"


def calculate(expression: str):
    """Perform a calculation."""
    print(f"  [Calculate] {expression}")
    try:
        # Safe evaluation
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def complete_task(summary: str):
    """Mark the task as complete with a summary."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    task_log.append(f"[{timestamp}] COMPLETED: {summary}")
    print(f"  [Complete] Task finished!")
    return f"Task completed. Summary: {summary}"


note_tool = tool(
    description="Add a note or observation to the task log.",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The note content"}
        },
        "required": ["content"]
    },
    execute=add_note
)

fetch_tool = tool(
    description="Fetch data from a URL (API endpoint or webpage).",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch"}
        },
        "required": ["url"]
    },
    execute=fetch_data
)

calc_tool = tool(
    description="Perform a mathematical calculation.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"]
    },
    execute=calculate
)

complete_tool = tool(
    description="Mark the task as complete. Call this when you have finished the task and have all the information needed.",
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "A summary of what was accomplished"}
        },
        "required": ["summary"]
    },
    execute=complete_task
)


# --- Callback ---

def log_progress(event: StepFinishEvent):
    print(f"\n--- Step {event.step_number} (tokens: {event.usage.total_tokens}) ---")
    for tc in event.step.tool_calls:
        print(f"  Action: {tc.name}")


# --- Main ---

async def main():
    print("Task Planner Agent")
    print("=" * 50)

    task = (
        "I need you to do the following:\n"
        "1. Get the current Bitcoin price from CoinGecko API\n"
        "2. Get the current Ethereum price as well\n"
        "3. Calculate what percentage Bitcoin's price is of Ethereum's price\n"
        "4. Add notes about your findings\n"
        "5. Complete the task with a summary\n"
    )

    print(f"Task:\n{task}")
    print("-" * 50)

    result = await generate_text(
        model=google("gemini-2.0-flash"),
        system=(
            "You are a task execution agent. Follow the user's instructions step by step. "
            "Use the tools provided to gather data, make calculations, and take notes. "
            "When you have completed all steps, call the complete_task tool with a summary."
        ),
        prompt=task,
        tools={
            "add_note": note_tool,
            "fetch_data": fetch_tool,
            "calculate": calc_tool,
            "complete_task": complete_tool,
        },
        on_step_finish=log_progress,
        # Stop when complete_task is called OR after 10 steps (safety limit)
        stop_when=[has_tool_call("complete_task"), step_count_is(10)],
    )

    print("\n" + "=" * 50)
    print("FINAL RESPONSE")
    print("=" * 50)
    print(result.text)

    print("\n" + "=" * 50)
    print("TASK LOG")
    print("=" * 50)
    for entry in task_log:
        print(entry)

    print(f"\nTotal steps: {len(result.response.get('steps', []))}")
    print(f"Total tokens: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
