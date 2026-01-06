"""Code Executor Agent

Demonstrates:
- Executing Python code in a sandboxed environment
- Tool with potentially dangerous operations (handled safely)
- Step callbacks for monitoring execution
"""

import asyncio
import io
from contextlib import redirect_stdout, redirect_stderr
from ai_query import generate_text, google, tool, Field, step_count_is, StepStartEvent, StepFinishEvent


# --- Code Execution Tool ---

@tool(description=(
    "Execute Python code and return the output. "
    "Use print() to display results. "
    "Available: math module, basic builtins (len, range, list, dict, sum, max, min, sorted, etc.)"
))
def execute_python(
    code: str = Field(description="The Python code to execute")
) -> str:
    """Execute Python code and return the output."""
    print(f"  [Executor] Running code...")

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restricted globals for safety
    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "all": all,
            "any": any,
            "isinstance": isinstance,
            "type": type,
            "True": True,
            "False": False,
            "None": None,
        },
        "math": __import__("math"),
    }

    local_vars = {}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals, local_vars)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        result = ""
        if stdout:
            result += f"Output:\n{stdout}"
        if stderr:
            result += f"\nStderr:\n{stderr}"
        if not result:
            # Check if there's a result variable or last expression
            if "result" in local_vars:
                result = f"Result: {local_vars['result']}"
            else:
                result = "Code executed successfully (no output)"

        return result.strip()

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# --- Callbacks ---

def on_start(event: StepStartEvent):
    print(f"\n[Step {event.step_number}] Starting...")


def on_finish(event: StepFinishEvent):
    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            print(f"[Step {event.step_number}] Executing code:")
            code = tc.arguments.get("code", "")
            for line in code.split("\n"):
                print(f"    {line}")

    if event.step.tool_results:
        for tr in event.step.tool_results:
            print(f"[Step {event.step_number}] Result: {tr.result}")


# --- Main ---

async def main():
    print("Python Code Executor Agent")
    print("=" * 50)

    problems = [
        "Calculate the first 10 Fibonacci numbers",
        "Find all prime numbers between 1 and 50",
        "Calculate the factorial of 12",
    ]

    for problem in problems:
        print(f"\nProblem: {problem}")
        print("-" * 40)

        result = await generate_text(
            model=google("gemini-2.0-flash"),
            system=(
                "You are a Python coding assistant. When asked to solve problems, "
                "write and execute Python code to find the answer. "
                "Always use print() to display your results clearly."
            ),
            prompt=problem,
            tools={"execute_python": execute_python},
            on_step_start=on_start,
            on_step_finish=on_finish,
            stop_when=step_count_is(4),
        )

        print(f"\nAnswer: {result.text}")
        print(f"Tokens: {result.usage.total_tokens}")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
