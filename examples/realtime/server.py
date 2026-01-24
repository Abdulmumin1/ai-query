"""Real-time Task Assistant with AI.

Usage:
    uv run examples/realtime/server.py

Then connect with the client:
    uv run examples/realtime/client.py --user alice

Or use curl:
    curl http://localhost:8080/events  # SSE stream
    curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "Add a task: buy groceries"}'
"""

from ai_query import tool, Field
from ai_query.agents import Agent, MemoryStorage
from ai_query.providers import google


class TaskAssistant(Agent):
    """Real-time task assistant with AI support."""

    enable_event_log = True  # Enable event replay on reconnect

    def __init__(self):
        @tool(description="Add a new task to the list")
        async def add_task(task: str = Field(description="Task to add")) -> str:
            tasks = self.state.get("tasks", []) + [task]
            await self.update_state(tasks=tasks)
            await self.emit("task_added", {"task": task, "count": len(tasks)})
            return f"Added task #{len(tasks)}: {task}"

        @tool(description="Mark a task as complete")
        async def complete_task(task_number: int = Field(description="Task number to complete")) -> str:
            tasks = self.state.get("tasks", [])
            if task_number < 1 or task_number > len(tasks):
                return f"Invalid task number. You have {len(tasks)} tasks."
            task = tasks[task_number - 1]
            remaining = [t for i, t in enumerate(tasks) if i != task_number - 1]
            completed = self.state.get("completed", []) + [task]
            await self.update_state(tasks=remaining, completed=completed)
            await self.emit("task_completed", {"task": task, "remaining": len(remaining)})
            return f"Completed: {task}"

        @tool(description="List all current tasks")
        async def list_tasks() -> str:
            tasks = self.state.get("tasks", [])
            if not tasks:
                return "No pending tasks."
            return "Tasks:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)])

        super().__init__(
            "task-assistant",
            model=google("gemini-2.0-flash"),
            system="""You are a task assistant. Help users plan and complete tasks.
            Use the available tools to track tasks and progress.""",
            storage=MemoryStorage(),
            initial_state={"tasks": [], "completed": []},
            tools={"add_task": add_task, "complete_task": complete_task, "list_tasks": list_tasks},
        )

    async def on_start(self):
        task_count = len(self.state.get("tasks", []))
        print(f"Task Assistant started. Pending tasks: {task_count}")

    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        task_count = len(self.state.get("tasks", []))
        if task_count > 0:
            await connection.send(f"Welcome back! You have {task_count} pending tasks.")
        else:
            await connection.send("Hello! I'm your task assistant. How can I help?")

    async def on_message(self, connection, message):
        """Handle WebSocket messages with streaming."""
        await self.emit("chat_start", {"message": message})

        async for chunk in self.stream(message):
            await self.emit("chunk", {"content": chunk})
            await connection.send(chunk)

        await self.emit("chat_complete", {})


if __name__ == "__main__":
    print("Task Assistant Server")
    print("=" * 40)
    print("Endpoints:")
    print("  POST /chat   - Send message")
    print("  GET  /events - SSE stream")
    print("  WS   /ws     - WebSocket")
    print("  GET  /state  - Agent state")
    print()
    TaskAssistant().serve(host="localhost", port=8080)
