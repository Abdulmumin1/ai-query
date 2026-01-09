"""Real-time Task Assistant with AI.

Usage:
    uv run examples/realtime/server.py

Then connect with the client:
    uv run examples/realtime/client.py --user alice
"""

from ai_query.agents import ChatAgent, InMemoryAgent
from ai_query import tool, Field
from ai_query.providers.google import google


class TaskAssistant(ChatAgent, InMemoryAgent):
    """Real-time task assistant with AI support."""
    
    model = google("gemini-2.0-flash")
    system = """You are a task assistant. Help users plan and complete tasks.
    Use the available tools to track tasks and progress."""
    
    initial_state = {
        "tasks": [],
        "completed": [],
        "messages_handled": 0,
    }
    
    @property
    def tools(self):
        @tool(description="Add a new task to the list")
        async def add_task(task: str = Field(description="Task to add")) -> str:
            tasks = self.state["tasks"] + [task]
            await self.set_state({**self.state, "tasks": tasks})
            return f"Added task #{len(tasks)}: {task}"
        
        @tool(description="Mark a task as complete")
        async def complete_task(task_number: int = Field(description="Task number to complete")) -> str:
            if task_number < 1 or task_number > len(self.state["tasks"]):
                return f"Invalid task number. You have {len(self.state['tasks'])} tasks."
            task = self.state["tasks"][task_number - 1]
            remaining = [t for i, t in enumerate(self.state["tasks"]) if i != task_number - 1]
            completed = self.state["completed"] + [task]
            await self.set_state({**self.state, "tasks": remaining, "completed": completed})
            return f"Completed: {task}"
        
        @tool(description="List all current tasks")
        async def list_tasks() -> str:
            if not self.state["tasks"]:
                return "No pending tasks."
            return "Tasks:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.state["tasks"])])
        
        return {"add_task": add_task, "complete_task": complete_task, "list_tasks": list_tasks}
    
    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        
        user = ctx.metadata.get("user", "anonymous")
        connection.user = user
        
        task_count = len(self.state["tasks"])
        if task_count > 0:
            await connection.send(f"Welcome back! You have {task_count} pending tasks.")
        else:
            await connection.send("Hello! I'm your task assistant. How can I help?")
        print(f"+ {user} connected")
    
    async def on_message(self, connection, message):
        user = getattr(connection, "user", "anonymous")
        
        # Track message count
        await self.set_state({
            **self.state,
            "messages_handled": self.state["messages_handled"] + 1
        })
        
        print(f"  [{user}] {message[:50]}...")
        
        # Use SSE for efficient AI streaming
        response = await self.stream_chat_sse(message)
        await connection.send(response)
    
    async def on_close(self, connection, code, reason):
        user = getattr(connection, "user", "anonymous")
        await super().on_close(connection, code, reason)
        print(f"- {user} disconnected")


if __name__ == "__main__":
    print("Task Assistant Server")
    print("=" * 40)
    print("Connect with: uv run examples/realtime/client.py --user YourName")
    print()
    TaskAssistant("assistant-1").serve(host="localhost", port=8080, path="/ws")
